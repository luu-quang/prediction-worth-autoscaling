import numpy as np
import pandas as pd
import pytest

from src.data.folds import assign_day_index, generate_folds
from src.data.schema import validate_forecast_df
from src.forecasting.features import build_features
from src.forecasting.lightgbm_model import _fit_model, generate_lightgbm_forecasts

SMALL_PARAMS = {"n_estimators": 30, "num_leaves": 7, "min_child_samples": 5, "verbosity": -1}


def test_generate_lightgbm_forecasts_schema_and_folds(synthetic_demand_df):
    folds = generate_folds()
    forecast_df = generate_lightgbm_forecasts(
        synthetic_demand_df,
        horizons_steps=[1, 6],
        folds=folds,
        lgb_params=SMALL_PARAMS,
        early_stopping_rounds=5,
    )
    validate_forecast_df(forecast_df)

    assert set(forecast_df["model_id"]) == {"lightgbm"}
    assert set(forecast_df["horizon_steps"]) == {1, 6}
    assert set(forecast_df["fold_id"]) == {"fold_4", "fold_5", "fold_6", "fold_7"}
    # LightGBM output should be in a plausible range for this demand series, not garbage
    assert forecast_df["forecast_point"].between(0, 200).all()


def test_generate_lightgbm_forecasts_deterministic_given_seed(synthetic_demand_df):
    folds = generate_folds(test_days=(4,))
    kwargs = {
        "horizons_steps": [1],
        "folds": folds,
        "seed": 7,
        "lgb_params": SMALL_PARAMS,
        "early_stopping_rounds": 5,
    }
    run_1 = generate_lightgbm_forecasts(synthetic_demand_df, **kwargs)
    run_2 = generate_lightgbm_forecasts(synthetic_demand_df, **kwargs)
    assert run_1["forecast_point"].to_numpy() == pytest.approx(run_2["forecast_point"].to_numpy())


def test_fit_model_ignores_rows_outside_train_and_val_mask():
    # Same invariant as Track A Step 7's AR test: corrupting demand on rows outside
    # both train_mask and val_mask must not change the fitted model's predictions
    # on a fixed held-out feature row (docs/OPEN_DECISIONS_CLOSED.md §E2/§P).
    n_steps = 4 * 288  # day 1 = train, day 2 = val, day 3 untouched buffer, day 4 = corrupted
    timestamps = pd.Series(pd.date_range("2024-01-01", periods=n_steps, freq="300s"))
    rng = np.random.default_rng(2)
    demand = pd.Series(100.0 + 0.01 * np.arange(n_steps) + rng.normal(0, 0.01, n_steps))
    day_index = assign_day_index(timestamps)

    features = build_features(demand, rolling_windows=(6,))
    horizon = 1
    target = demand.shift(-horizon)
    train_mask = day_index == 1
    val_mask = day_index == 2

    fit_kwargs = {
        "train_mask": train_mask,
        "val_mask": val_mask,
        "seed": 7,
        "lgb_params": SMALL_PARAMS,
        "early_stopping_rounds": 5,
    }
    baseline_model = _fit_model(features, target, **fit_kwargs)

    corrupted_demand = demand.copy()
    corrupted_demand[day_index == 4] += 1e6  # corrupt only rows outside train/val
    corrupted_features = build_features(corrupted_demand, rolling_windows=(6,))
    corrupted_target = corrupted_demand.shift(-horizon)
    corrupted_model = _fit_model(corrupted_features, corrupted_target, **fit_kwargs)

    # Predict on a fixed, uncorrupted held-out row (from day 3, untouched by both
    # the corruption and the train/val masks).
    probe_row = features.loc[day_index == 3].dropna().iloc[[0]]
    baseline_pred = baseline_model.predict(probe_row)
    corrupted_pred = corrupted_model.predict(probe_row)
    assert baseline_pred == pytest.approx(corrupted_pred)
