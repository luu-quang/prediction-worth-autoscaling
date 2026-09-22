import numpy as np
import pandas as pd
import pytest

from src.data.folds import generate_folds
from src.data.schema import validate_forecast_df
from src.forecasting.ewma import ewma_forecast, ewma_level, generate_ewma_forecasts


def test_ewma_level_matches_recursion():
    demand = pd.Series([10.0, 20.0, 10.0, 20.0])
    alpha = 0.5
    level = ewma_level(demand, alpha)
    expected = [10.0, 15.0, 12.5, 16.25]
    assert level.tolist() == pytest.approx(expected)


def test_ewma_forecast_is_flat_and_shifted_level():
    demand = pd.Series([10.0, 20.0, 10.0, 20.0, 10.0])
    alpha = 0.5
    horizon = 2
    forecast = ewma_forecast(demand, horizon, alpha)
    level = ewma_level(demand, alpha)
    assert forecast.iloc[2] == level.iloc[0]
    assert forecast.iloc[4] == level.iloc[2]
    assert pd.isna(forecast.iloc[0])
    assert pd.isna(forecast.iloc[1])


def test_ewma_constant_series_is_exact_regardless_of_alpha():
    demand = pd.Series([42.0] * 20)
    for alpha in (0.05, 0.5, 0.9):
        forecast = ewma_forecast(demand, horizon_steps=3, alpha=alpha)
        assert (forecast.dropna() == 42.0).all()


def test_generate_ewma_forecasts_schema_and_folds(synthetic_demand_df):
    folds = generate_folds()
    forecast_df = generate_ewma_forecasts(
        synthetic_demand_df, horizons_steps=[1, 3, 6], folds=folds
    )
    validate_forecast_df(forecast_df)

    assert set(forecast_df["model_id"]) == {"ewma"}
    assert set(forecast_df["horizon_steps"]) == {1, 3, 6}
    assert set(forecast_df["fold_id"]) == {"fold_4", "fold_5", "fold_6", "fold_7"}
    assert "alpha" in forecast_df.columns
    assert forecast_df["alpha"].between(0, 1).all()

    # exactly one alpha selected per (fold, horizon)
    n_alphas = forecast_df.groupby(["fold_id", "horizon_steps"])["alpha"].nunique()
    assert (n_alphas == 1).all()


def test_generate_ewma_forecasts_tunes_alpha_on_validation_only(synthetic_demand_df):
    # A demand series with a day-3 (validation for fold_4) anomaly that only a very
    # small alpha can track well; the anomaly is far from the test day so it must
    # not affect any other fold's tuning.
    demand_df = synthetic_demand_df.copy()
    from src.data.folds import assign_day_index

    day_index = assign_day_index(demand_df["timestamp"])
    val_day_3_mask = day_index == 3
    rng = np.random.default_rng(0)
    demand_df.loc[val_day_3_mask, "demand"] += rng.normal(0, 50, size=val_day_3_mask.sum())
    demand_df["demand"] = demand_df["demand"].clip(lower=0)

    folds = generate_folds()
    forecast_df = generate_ewma_forecasts(demand_df, horizons_steps=[1], folds=folds)

    alpha_fold4 = forecast_df.loc[forecast_df["fold_id"] == "fold_4", "alpha"].iloc[0]
    alpha_fold7 = forecast_df.loc[forecast_df["fold_id"] == "fold_7", "alpha"].iloc[0]
    # fold_4's validation day (day 3) is noisy: a high alpha would feed that noise
    # straight into the level, so RMSE-minimization on that validation day should
    # prefer a smaller (smoother) alpha than fold_7, whose validation day (day 6)
    # is clean synthetic data.
    assert alpha_fold4 < alpha_fold7
