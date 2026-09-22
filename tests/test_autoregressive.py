import numpy as np
import pandas as pd
import pytest

from src.data.folds import assign_day_index, generate_folds
from src.data.schema import validate_forecast_df
from src.forecasting.autoregressive import (
    _fit_model,
    build_lag_features,
    generate_autoregressive_forecasts,
)


def test_build_lag_features_alignment():
    demand = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    features = build_lag_features(demand, lags=(1, 2, 3))
    # lag_1 = D_t (no shift)
    assert features["lag_1"].tolist() == demand.tolist()
    # lag_2 = D_{t-1}
    assert features["lag_2"].iloc[2] == demand.iloc[1]
    # lag_3 = D_{t-2}
    assert features["lag_3"].iloc[2] == demand.iloc[0]
    assert pd.isna(features["lag_2"].iloc[0])


def test_generate_autoregressive_forecasts_schema_and_folds(synthetic_demand_df):
    folds = generate_folds()
    forecast_df = generate_autoregressive_forecasts(
        synthetic_demand_df, horizons_steps=[1, 3, 6], folds=folds
    )
    validate_forecast_df(forecast_df)

    assert set(forecast_df["model_id"]) == {"autoregressive"}
    assert set(forecast_df["horizon_steps"]) == {1, 3, 6}
    assert set(forecast_df["fold_id"]) == {"fold_4", "fold_5", "fold_6", "fold_7"}


def test_generate_autoregressive_forecasts_exact_on_linear_trend():
    # D_t = a + b*t is exactly linear in lag_1 (D_{t+H} = D_t + b*H), so a fitted
    # linear regression should reproduce it almost exactly out of sample.
    n_days = 7
    steps_per_day = 288
    n_steps = n_days * steps_per_day
    timestamps = pd.date_range("2024-01-01", periods=n_steps, freq="300s")
    demand = 100.0 + 0.01 * np.arange(n_steps)
    demand_df = pd.DataFrame({"timestamp": timestamps, "demand": demand})

    folds = generate_folds()
    forecast_df = generate_autoregressive_forecasts(demand_df, horizons_steps=[3], folds=folds)

    assert forecast_df["forecast_point"].to_numpy() == pytest.approx(
        forecast_df["actual_demand"].to_numpy(), abs=1e-6
    )


def test_fit_model_ignores_rows_outside_train_mask():
    # Corrupting demand on rows the train_mask excludes must not change the fitted
    # model at all -- the model is fit only on rows where train_mask is True
    # (docs/OPEN_DECISIONS_CLOSED.md §E2/§P: no test/val information may fit the model).
    # Day 3 is used (not day 2) so the corruption doesn't touch day 1's last few
    # rows via label spillover (target = demand.shift(-horizon)) or lag lookback.
    n_steps = 3 * 288
    timestamps = pd.Series(pd.date_range("2024-01-01", periods=n_steps, freq="300s"))
    rng = np.random.default_rng(1)
    demand = pd.Series(100.0 + 0.01 * np.arange(n_steps) + rng.normal(0, 0.01, n_steps))
    day_index = assign_day_index(timestamps)

    features = build_lag_features(demand)
    horizon = 1
    target = demand.shift(-horizon)
    train_mask = day_index == 1

    baseline_model = _fit_model(features, target, train_mask)

    corrupted_demand = demand.copy()
    corrupted_demand[day_index == 3] += 1e6  # corrupt only rows outside train_mask
    corrupted_features = build_lag_features(corrupted_demand)
    corrupted_target = corrupted_demand.shift(-horizon)
    corrupted_model = _fit_model(corrupted_features, corrupted_target, train_mask)

    assert baseline_model.coef_ == pytest.approx(corrupted_model.coef_, abs=1e-8)
    assert baseline_model.intercept_ == pytest.approx(corrupted_model.intercept_, abs=1e-8)
