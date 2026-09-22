import math

import numpy as np
import pandas as pd

from src.data.folds import assign_day_index
from src.data.schema import FORECAST_COLUMNS, validate_demand_df

MODEL_ID = "synthetic"

# Locked structural grids (docs/OPEN_DECISIONS_CLOSED.md §A3-A5).
BIAS_FRACTIONS = (-1.0, -0.5, 0.0, 0.5, 1.0)
PHI_GRID = (0.0, 0.5, 0.8)
LAG_FRACTIONS = (0.0, 0.25, 0.5, 1.0)

# TODO: the "low/medium/high" target-RMSE grid (§A5) must be calibrated from
# empirical validation-day RMSE levels of the real forecasting models, once real
# trace data is available. No numeric default is set here -- target_rmse (m) is
# a required, caller-supplied parameter everywhere in this module.


def round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


def lag_steps(lag_fraction: float, horizon_steps: int) -> int:
    """lambda = floor(c_lambda*H + 0.5), round-half-up (docs/OPEN_DECISIONS_CLOSED.md §A4).

    c_lambda=0 -> lambda=0 (Oracle-like: anchored at D_{t+H} before noise).
    c_lambda=1 -> lambda=H (Persistence-like: anchored at D_t).
    """
    return round_half_up(lag_fraction * horizon_steps)


def burn_in_steps(phi: float) -> int:
    """max(50, 10/(1-phi)), rounded up (§A1). Discarded before use, never exported.

    Subtracts a tiny epsilon before ceil() so float representation noise (e.g.
    1-0.8 != 0.2 exactly) doesn't push an intended-exact value like 10/0.2=50 up
    to 51.
    """
    if phi >= 1:
        raise ValueError(f"phi must be < 1 for a stationary AR(1) process, got {phi}")
    return max(50, math.ceil(10 / (1 - phi) - 1e-9))


def _unit_ar1_process(n: int, phi: float, burn_in: int, rng: np.random.Generator) -> np.ndarray:
    """AR(1) with sigma_eps=1, mean 0, after discarding burn_in steps: length n.

    r_t - mu = sigma_eps * w_t exactly, by linearity of the AR(1) recursion in
    eps_t (same z-draws, scaled). This lets sigma_eps be calibrated by a closed-form
    solve (see _solve_sigma) instead of resimulating the AR path per candidate sigma.
    """
    total = n + burn_in
    z = rng.standard_normal(total)
    w = np.empty(total)
    w[0] = z[0]
    for i in range(1, total):
        w[i] = phi * w[i - 1] + z[i]
    return w[burn_in:]


def simulate_residual(n: int, phi: float, sigma_eps: float, mu: float, seed: int) -> np.ndarray:
    """r_t = mu + phi*(r_{t-1}-mu) + eps_t, eps_t ~ N(0, sigma_eps^2) (§A1)."""
    burn_in = burn_in_steps(phi)
    rng = np.random.default_rng(seed)
    w = _unit_ar1_process(n, phi, burn_in, rng)
    return mu + sigma_eps * w


def raw_synthetic_forecast(
    demand: pd.Series,
    horizon_steps: int,
    lag_fraction: float,
    mu: float,
    phi: float,
    sigma_eps: float,
    seed: int,
) -> pd.Series:
    """f_{t,H} = D_{t+H-lambda} - r_t (§A1), given mu/sigma_eps directly (no RMSE
    calibration). Origin-t indexed: row t holds the forecast MADE at t, not yet
    realigned to the target row t+H (callers shift by horizon_steps to export,
    matching persistence/ewma/autoregressive/lightgbm's convention).
    """
    lam = lag_steps(lag_fraction, horizon_steps)
    if not 0 <= lam <= horizon_steps:
        raise ValueError(f"lag_steps={lam} out of [0, horizon_steps={horizon_steps}]")

    r = simulate_residual(len(demand), phi, sigma_eps, mu, seed)
    shift_amount = horizon_steps - lam
    anchor = demand.shift(-shift_amount) if shift_amount else demand.copy()
    return anchor - pd.Series(r, index=demand.index)


def _solve_sigma(a: np.ndarray, w: np.ndarray, target_rmse: float):
    """sigma_eps>=0 s.t. RMSE(a + sigma_eps*w) == target_rmse, exactly (§A2).

    e_total = a + sigma_eps*w is linear in sigma_eps (a is fixed: lag_term + mu;
    w is the fixed unit-sigma AR(1) draw), so RMSE(e_total)^2 is an exact upward
    parabola in sigma_eps: A*sigma^2 + B*sigma + C, A=mean(w^2)>=0. Solved by the
    quadratic formula rather than iterative search, so the result is exact.
    Returns (sigma_eps or None, infeasible: bool).
    """
    A = float(np.mean(w**2))
    B = float(2 * np.mean(a * w))
    C = float(np.mean(a**2))
    target_sq = target_rmse**2

    if A <= 0:
        return (0.0, C > target_sq)

    disc = B**2 - 4 * A * (C - target_sq)
    if disc < 0:
        return (None, True)  # target unreachable at any real sigma

    sqrt_disc = math.sqrt(disc)
    roots = sorted([(-B - sqrt_disc) / (2 * A), (-B + sqrt_disc) / (2 * A)])
    nonneg_roots = [root for root in roots if root >= 0]
    if not nonneg_roots:
        return (None, True)  # only reachable at sigma < 0
    return (nonneg_roots[0], False)


def calibrate_sigma_eps(
    demand: pd.Series,
    horizon_steps: int,
    lag_fraction: float,
    mu: float,
    phi: float,
    target_rmse: float,
    seed: int,
):
    """Solve for sigma_eps (§A2 steps 1-3). Returns (sigma_eps or None, infeasible)."""
    lam = lag_steps(lag_fraction, horizon_steps)
    shift_amount = horizon_steps - lam
    anchor = demand.shift(-shift_amount) if shift_amount else demand.copy()
    target = demand.shift(-horizon_steps)
    a = (target - anchor).to_numpy() + mu  # lag_term + mu

    burn_in = burn_in_steps(phi)
    rng = np.random.default_rng(seed)
    w = _unit_ar1_process(len(demand), phi, burn_in, rng)

    valid = ~np.isnan(a)
    if not valid.any():
        return (None, True)
    return _solve_sigma(a[valid], w[valid], target_rmse)


def generate_synthetic_forecast(
    demand: pd.Series,
    horizon_steps: int,
    target_rmse: float,
    bias_fraction: float,
    phi: float,
    lag_fraction: float,
    seed: int,
) -> dict:
    """Calibrated centered-AR(1) synthetic forecast for one (m, c_b, phi, c_lambda, H).

    If the target RMSE is unreachable, infeasible=True and forecast_point is all
    NaN -- never a silently wrong RMSE (§A2 feasibility rule). achieved_rmse is
    verified against target_rmse (§A2 step 4) and raises if they disagree, since a
    mismatch would mean a bug in the calibration math, not a legitimate result.
    """
    mu = bias_fraction * target_rmse
    lam = lag_steps(lag_fraction, horizon_steps)
    sigma_eps, infeasible = calibrate_sigma_eps(
        demand, horizon_steps, lag_fraction, mu, phi, target_rmse, seed
    )

    if infeasible:
        return {
            "forecast_point": pd.Series(np.nan, index=demand.index),
            "lambda_steps": lam,
            "mu": mu,
            "sigma_eps": sigma_eps,
            "achieved_rmse": None,
            "infeasible": True,
        }

    forecast_point = raw_synthetic_forecast(
        demand, horizon_steps, lag_fraction, mu, phi, sigma_eps, seed
    )

    target = demand.shift(-horizon_steps)
    e_total = (target - forecast_point).to_numpy()
    achieved_rmse = float(np.sqrt(np.nanmean(e_total**2)))
    if not math.isclose(achieved_rmse, target_rmse, rel_tol=1e-6, abs_tol=1e-9):
        raise AssertionError(
            f"achieved RMSE {achieved_rmse} != target {target_rmse}: sigma_eps calibration bug"
        )

    return {
        "forecast_point": forecast_point,
        "lambda_steps": lam,
        "mu": mu,
        "sigma_eps": sigma_eps,
        "achieved_rmse": achieved_rmse,
        "infeasible": False,
    }


def generate_synthetic_error_forecasts(
    demand_df,
    horizons_steps,
    folds,
    target_rmse: float,
    bias_fraction: float,
    phi: float,
    lag_fraction: float,
    seed: int,
    synthetic_config_id: str,
    split: str = "test",
) -> pd.DataFrame:
    """Synthetic AR(1) forecasts, exported like every other Track A model, in the
    §A6 output contract (locked columns + synthetic_config_id + infeasible).

    Calibration uses the whole demand trace, including future demand relative to
    each origin t -- the one Track A model explicitly permitted to do this (§P):
    "The synthetic generator may use future demand offline to create controlled
    forecast_df, but once forecast_df is produced, it enters the same simulator
    interface as all other forecast sources. The simulator itself receives no
    privileged future information." No special simulator code path exists for it.
    """
    if split not in ("test", "val"):
        raise ValueError(f"split must be 'test' or 'val', got {split!r}")

    validate_demand_df(demand_df)
    demand_df = demand_df.sort_values("timestamp").reset_index(drop=True)
    day_index = assign_day_index(demand_df["timestamp"])
    demand = demand_df["demand"]

    rows = []
    for horizon in horizons_steps:
        result = generate_synthetic_forecast(
            demand, horizon, target_rmse, bias_fraction, phi, lag_fraction, seed
        )
        forecast_point = result["forecast_point"].shift(horizon)  # origin-t -> target row t+H
        for fold in folds:
            target_day = fold.test_day if split == "test" else fold.val_day
            mask = day_index == target_day
            if not mask.any():
                continue
            rows.append(
                pd.DataFrame(
                    {
                        "timestamp": demand_df.loc[mask, "timestamp"],
                        "actual_demand": demand_df.loc[mask, "demand"],
                        "forecast_point": forecast_point.loc[mask],
                        "horizon_steps": horizon,
                        "fold_id": fold.fold_id,
                        "model_id": MODEL_ID,
                        "synthetic_config_id": synthetic_config_id,
                        "infeasible": result["infeasible"],
                    }
                )
            )

    extra_cols = ["synthetic_config_id", "infeasible"]
    if rows:
        forecast_df = pd.concat(rows, ignore_index=True)
    else:
        forecast_df = pd.DataFrame(columns=[*FORECAST_COLUMNS, *extra_cols])

    # Infeasible configs are kept, not silently dropped, so callers can see/filter
    # them (§A2: leave the cell blank/hatched downstream, don't just vanish it).
    # Only boundary-NaN rows from a *feasible* config are dropped.
    keep = forecast_df["forecast_point"].notna() | forecast_df["infeasible"]
    forecast_df = forecast_df[keep].reset_index(drop=True)
    return forecast_df[[*FORECAST_COLUMNS, *extra_cols]]


def structural_grid(
    target_rmse_values, bias_fractions=BIAS_FRACTIONS, phis=PHI_GRID, lag_fractions=LAG_FRACTIONS
):
    """Full factorial of (m, c_b, phi, c_lambda) configs (§A5).

    docs/OPEN_DECISIONS_CLOSED.md §Q: the full factorial may support T3 but is not
    the main F4 visualization -- selecting which panels/subsets to plot is a
    downstream figure-generation concern, not this generator's job.
    """
    for m in target_rmse_values:
        for c_b in bias_fractions:
            for phi in phis:
                for c_lambda in lag_fractions:
                    yield {
                        "target_rmse": m,
                        "bias_fraction": c_b,
                        "phi": phi,
                        "lag_fraction": c_lambda,
                        "synthetic_config_id": f"m{m:g}_cb{c_b:+.2f}_phi{phi:.2f}_clam{c_lambda:.2f}",
                    }
