import numpy as np
import pandas as pd

from src.data.folds import assign_day_index
from src.data.schema import FORECAST_COLUMNS, validate_demand_df
from src.metrics.forecast_metrics import rmse

MODEL_ID = "ewma"

# Not locked in docs/OPEN_DECISIONS_CLOSED.md; overridable via configs/forecasting.yaml ewma.alpha_grid.
DEFAULT_ALPHA_GRID = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)


def ewma_level(demand: pd.Series, alpha: float) -> pd.Series:
    """Causal exponentially-weighted level: S_t = alpha*D_t + (1-alpha)*S_{t-1}."""
    return demand.ewm(alpha=alpha, adjust=False).mean()


def ewma_forecast(demand: pd.Series, horizon_steps: int, alpha: float) -> pd.Series:
    """forecast(t+H) = S_t (flat extrapolation), aligned to the target timestep t+H."""
    return ewma_level(demand, alpha).shift(horizon_steps)


def _select_alpha(demand: pd.Series, horizon: int, val_mask: pd.Series, alpha_grid):
    """Pick the alpha minimizing RMSE on the validation day only."""
    val_actual = demand.loc[val_mask]
    best_alpha, best_rmse = None, np.inf
    for alpha in alpha_grid:
        val_forecast = ewma_forecast(demand, horizon, alpha).loc[val_mask]
        valid = val_forecast.notna()
        if not valid.any():
            continue
        score = rmse(val_actual[valid], val_forecast[valid])
        if score < best_rmse:
            best_alpha, best_rmse = alpha, score
    if best_alpha is None:
        raise ValueError(f"no valid alpha for horizon={horizon}: validation day too short")
    return best_alpha, best_rmse


def generate_ewma_forecasts(
    demand_df, horizons_steps, folds, alpha_grid=DEFAULT_ALPHA_GRID
) -> pd.DataFrame:
    """EWMA forecasts per test fold; alpha is tuned per (fold, horizon) on validation only.

    The causal level S_t at a test-day timestamp is computed from the whole demand
    history up to t (which naturally spans earlier train/val days) -- this is not
    leakage, since no information at or after the target timestamp t+H is used.
    """
    validate_demand_df(demand_df)
    demand_df = demand_df.sort_values("timestamp").reset_index(drop=True)
    day_index = assign_day_index(demand_df["timestamp"])
    demand = demand_df["demand"]

    rows = []
    for horizon in horizons_steps:
        for fold in folds:
            val_mask = day_index == fold.val_day
            test_mask = day_index == fold.test_day
            if not val_mask.any() or not test_mask.any():
                continue

            alpha, _ = _select_alpha(demand, horizon, val_mask, alpha_grid)
            forecast_point = ewma_forecast(demand, horizon, alpha)
            rows.append(
                pd.DataFrame(
                    {
                        "timestamp": demand_df.loc[test_mask, "timestamp"],
                        "actual_demand": demand_df.loc[test_mask, "demand"],
                        "forecast_point": forecast_point.loc[test_mask],
                        "horizon_steps": horizon,
                        "fold_id": fold.fold_id,
                        "model_id": MODEL_ID,
                        "alpha": alpha,
                    }
                )
            )

    if rows:
        forecast_df = pd.concat(rows, ignore_index=True)
    else:
        forecast_df = pd.DataFrame(columns=[*FORECAST_COLUMNS, "alpha"])

    forecast_df = forecast_df.dropna(subset=["forecast_point"]).reset_index(drop=True)
    return forecast_df[[*FORECAST_COLUMNS, "alpha"]]
