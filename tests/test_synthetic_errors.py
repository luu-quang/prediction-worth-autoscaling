import pandas as pd
import pytest

from src.data.folds import generate_folds
from src.data.schema import validate_forecast_df
from src.forecasting.persistence import persistence_forecast
from src.forecasting.synthetic_errors import (
    BIAS_FRACTIONS,
    LAG_FRACTIONS,
    PHI_GRID,
    burn_in_steps,
    calibrate_sigma_eps,
    generate_synthetic_error_forecasts,
    generate_synthetic_forecast,
    lag_steps,
    raw_synthetic_forecast,
    round_half_up,
    structural_grid,
)


def test_round_half_up():
    assert round_half_up(2.5) == 3
    assert round_half_up(2.4) == 2
    assert round_half_up(0.5) == 1
    assert round_half_up(0.0) == 0


def test_lag_steps_endpoints_match_locked_interpretation():
    horizon = 6
    assert lag_steps(0.0, horizon) == 0  # Oracle-like
    assert lag_steps(1.0, horizon) == horizon  # Persistence-like
    assert lag_steps(0.5, horizon) == 3
    assert lag_steps(0.25, horizon) == round_half_up(0.25 * horizon)


def test_burn_in_steps_matches_formula():
    assert burn_in_steps(0.0) == 50  # max(50, 10/1)
    assert burn_in_steps(0.8) == 50  # max(50, 10/0.2=50)
    assert burn_in_steps(0.9) == 100  # max(50, 10/0.1=100)
    with pytest.raises(ValueError):
        burn_in_steps(1.0)


def test_t9_synthetic_persistence_equivalence(synthetic_demand_df):
    # Step 11 / docs/OPEN_DECISIONS_CLOSED.md §M T9: c_lambda=1, c_b=0, phi=0,
    # sigma_eps->0 must reproduce Persistence exactly (identical forecast arrays,
    # not just approximately).
    demand = synthetic_demand_df["demand"]
    horizon = 6

    synthetic = raw_synthetic_forecast(
        demand, horizon, lag_fraction=1.0, mu=0.0, phi=0.0, sigma_eps=0.0, seed=123
    ).shift(horizon)
    persistence = persistence_forecast(demand, horizon)

    pd.testing.assert_series_equal(synthetic, persistence, check_names=False)


def test_t9_holds_for_any_phi_when_sigma_is_exactly_zero(synthetic_demand_df):
    # sigma_eps=0 makes r_t == mu == 0 regardless of phi (0*w == 0 exactly), so
    # T9 equivalence must hold for every phi in the locked grid, not just phi=0.
    demand = synthetic_demand_df["demand"]
    horizon = 3
    persistence = persistence_forecast(demand, horizon)
    for phi in PHI_GRID:
        synthetic = raw_synthetic_forecast(
            demand, horizon, lag_fraction=1.0, mu=0.0, phi=phi, sigma_eps=0.0, seed=1
        ).shift(horizon)
        pd.testing.assert_series_equal(synthetic, persistence, check_names=False)


def test_oracle_like_lag_uses_exact_future_demand(synthetic_demand_df):
    # c_lambda=0 -> lambda=0 -> anchor = D_{t+H} (the exact future value), so with
    # mu=0, sigma_eps=0 the synthetic forecast must equal the actual future demand.
    demand = synthetic_demand_df["demand"]
    horizon = 3
    forecast = raw_synthetic_forecast(
        demand, horizon, lag_fraction=0.0, mu=0.0, phi=0.0, sigma_eps=0.0, seed=1
    ).shift(horizon)
    actual = demand.shift(0)  # forecast row t+H should hold D_{t+H} itself
    valid = forecast.notna()
    assert (forecast[valid] == actual[valid]).all()


def test_generate_synthetic_forecast_achieves_target_rmse(synthetic_demand_df):
    # target_rmse=10.0 is comfortably above this config's sigma_eps=0 floor
    # (~4.46, from the lag_term + bias contribution alone), so it's feasible.
    demand = synthetic_demand_df["demand"]
    result = generate_synthetic_forecast(
        demand,
        horizon_steps=3,
        target_rmse=10.0,
        bias_fraction=0.5,
        phi=0.5,
        lag_fraction=0.5,
        seed=7,
    )
    assert not result["infeasible"]
    assert result["achieved_rmse"] == pytest.approx(10.0, rel=1e-6)
    assert result["sigma_eps"] >= 0


def test_generate_synthetic_forecast_bias_sign_matches_convention(synthetic_demand_df):
    # mu > 0 must mean systematic under-forecasting: mean(actual - forecast) > 0.
    demand = synthetic_demand_df["demand"]
    result = generate_synthetic_forecast(
        demand,
        horizon_steps=1,
        target_rmse=10.0,
        bias_fraction=1.0,
        phi=0.0,
        lag_fraction=0.0,
        seed=3,
    )
    forecast = result["forecast_point"].shift(1)
    actual = demand
    residual = (actual - forecast).dropna()
    assert residual.mean() > 0

    result_over = generate_synthetic_forecast(
        demand,
        horizon_steps=1,
        target_rmse=10.0,
        bias_fraction=-1.0,
        phi=0.0,
        lag_fraction=0.0,
        seed=3,
    )
    forecast_over = result_over["forecast_point"].shift(1)
    residual_over = (actual - forecast_over).dropna()
    assert residual_over.mean() < 0


def test_generate_synthetic_forecast_infeasible_when_target_too_tight(synthetic_demand_df):
    # lag_fraction=1.0 (Persistence-like) makes lag_term = D_{t+H}-D_t nonzero for a
    # real varying series; an unreasonably tiny target_rmse can't be reached even
    # at sigma_eps=0, so this must be flagged infeasible, not silently wrong.
    demand = synthetic_demand_df["demand"]
    result = generate_synthetic_forecast(
        demand,
        horizon_steps=6,
        target_rmse=1e-6,
        bias_fraction=0.0,
        phi=0.0,
        lag_fraction=1.0,
        seed=1,
    )
    assert result["infeasible"]
    assert result["forecast_point"].isna().all()
    assert result["achieved_rmse"] is None


def test_calibrate_sigma_eps_matches_generate_synthetic_forecast(synthetic_demand_df):
    # target_rmse=10.0 is comfortably above this config's sigma_eps=0 floor (~4.46).
    demand = synthetic_demand_df["demand"]
    sigma_eps, infeasible = calibrate_sigma_eps(
        demand, horizon_steps=3, lag_fraction=0.5, mu=1.0, phi=0.5, target_rmse=10.0, seed=9
    )
    assert not infeasible
    result = generate_synthetic_forecast(
        demand,
        horizon_steps=3,
        target_rmse=10.0,
        bias_fraction=0.1,
        phi=0.5,
        lag_fraction=0.5,
        seed=9,
    )
    # bias_fraction=0.1 at target_rmse=10.0 gives mu=1.0, matching the direct call above
    assert result["sigma_eps"] == pytest.approx(sigma_eps)


def test_generate_synthetic_error_forecasts_schema_and_folds(synthetic_demand_df):
    folds = generate_folds()
    forecast_df = generate_synthetic_error_forecasts(
        synthetic_demand_df,
        horizons_steps=[1, 3],
        folds=folds,
        target_rmse=5.0,
        bias_fraction=0.0,
        phi=0.5,
        lag_fraction=0.5,
        seed=42,
        synthetic_config_id="test_config",
    )
    validate_forecast_df(forecast_df)
    assert set(forecast_df["model_id"]) == {"synthetic"}
    assert set(forecast_df["fold_id"]) == {"fold_4", "fold_5", "fold_6", "fold_7"}
    assert set(forecast_df["synthetic_config_id"]) == {"test_config"}
    assert not forecast_df["infeasible"].any()


def test_generate_synthetic_error_forecasts_keeps_infeasible_rows_flagged(synthetic_demand_df):
    folds = generate_folds()
    forecast_df = generate_synthetic_error_forecasts(
        synthetic_demand_df,
        horizons_steps=[6],
        folds=folds,
        target_rmse=1e-6,
        bias_fraction=0.0,
        phi=0.0,
        lag_fraction=1.0,
        seed=1,
        synthetic_config_id="infeasible_config",
    )
    assert not forecast_df.empty
    assert forecast_df["infeasible"].all()
    assert forecast_df["forecast_point"].isna().all()


def test_structural_grid_is_full_factorial_and_locked():
    configs = list(structural_grid(target_rmse_values=[5.0, 10.0]))
    assert len(configs) == 2 * len(BIAS_FRACTIONS) * len(PHI_GRID) * len(LAG_FRACTIONS)
    ids = {c["synthetic_config_id"] for c in configs}
    assert len(ids) == len(configs)  # all config ids unique
