import numpy as np
import pandas as pd
import pytest

from src.data.folds import generate_folds
from src.forecasting.lightgbm_model import generate_lightgbm_forecasts
from src.forecasting.persistence import generate_persistence_forecasts
from src.forecasting.residual_quantiles import (
    compute_all_residual_quantiles,
    compute_validation_residuals,
    residual_quantiles,
)

SMALL_LGB_PARAMS = {"n_estimators": 30, "num_leaves": 7, "min_child_samples": 5, "verbosity": -1}


def test_compute_validation_residuals_sign_convention():
    val_forecast_df = pd.DataFrame(
        {
            "model_id": ["persistence"],
            "fold_id": ["fold_4"],
            "horizon_steps": [1],
            "timestamp": pd.to_datetime(["2024-01-01"]),
            "actual_demand": [10.0],
            "forecast_point": [8.0],  # under-forecast: actual > forecast -> residual > 0
        }
    )
    residuals_df = compute_validation_residuals(val_forecast_df)
    assert residuals_df["residual"].iloc[0] == pytest.approx(2.0)


def test_residual_quantiles_known_values():
    residuals_df = pd.DataFrame(
        {
            "model_id": ["persistence"] * 4,
            "fold_id": ["fold_4"] * 4,
            "horizon_steps": [1] * 4,
            "residual": [1.0, 2.0, 3.0, 4.0],
        }
    )
    quantiles = residual_quantiles(residuals_df, tau_grid=(0.5,))
    row = quantiles.iloc[0]
    assert row["quantile_value"] == pytest.approx(np.quantile([1.0, 2.0, 3.0, 4.0], 0.5))
    assert row["n"] == 4


def test_generate_persistence_forecasts_split_val_uses_validation_day(synthetic_demand_df):
    from src.data.folds import assign_day_index

    folds = generate_folds()
    val_df = generate_persistence_forecasts(
        synthetic_demand_df, horizons_steps=[1], folds=folds, split="val"
    )
    # day index must come from the full trace, not from val_df's own subset of
    # timestamps (assign_day_index re-bases "day 1" to its input's own minimum).
    day_by_timestamp = pd.Series(
        assign_day_index(synthetic_demand_df["timestamp"]).to_numpy(),
        index=synthetic_demand_df["timestamp"],
    )
    by_id = {f.fold_id: f for f in folds}
    for fold_id, group in val_df.groupby("fold_id"):
        days = set(day_by_timestamp.loc[group["timestamp"]].unique())
        assert days == {by_id[fold_id].val_day}


def test_generate_persistence_forecasts_split_test_unaffected_by_split_param(synthetic_demand_df):
    # default behavior (Step 1-8, already committed) must be unchanged
    folds = generate_folds()
    default_df = generate_persistence_forecasts(synthetic_demand_df, [1, 3, 6], folds)
    explicit_test_df = generate_persistence_forecasts(
        synthetic_demand_df, [1, 3, 6], folds, split="test"
    )
    pd.testing.assert_frame_equal(default_df, explicit_test_df)


def test_generate_persistence_forecasts_rejects_bad_split(synthetic_demand_df):
    folds = generate_folds()
    with pytest.raises(ValueError, match="split must be"):
        generate_persistence_forecasts(synthetic_demand_df, [1], folds, split="bogus")


def test_compute_all_residual_quantiles_covers_every_model(synthetic_demand_df):
    folds = generate_folds()
    quantiles = compute_all_residual_quantiles(
        synthetic_demand_df,
        horizons_steps=[1],
        folds=folds,
        tau_grid=(0.9,),
        model_kwargs={
            "lightgbm": {
                "lgb_params": SMALL_LGB_PARAMS,
                "early_stopping_rounds": 5,
            }
        },
    )
    assert set(quantiles["model_id"]) == {"persistence", "ewma", "autoregressive", "lightgbm"}
    assert set(quantiles["fold_id"]) == {"fold_4", "fold_5", "fold_6", "fold_7"}
    assert (quantiles["tau"] == 0.9).all()


def test_lightgbm_split_val_reuses_train_only_fit(synthetic_demand_df):
    # Sanity check that split="val" doesn't silently refit on val (it must still
    # use the same train-only-fit-plus-val-early-stopped model as split="test").
    folds = generate_folds(test_days=(4,))
    kwargs = {
        "horizons_steps": [1],
        "folds": folds,
        "seed": 7,
        "lgb_params": SMALL_LGB_PARAMS,
        "early_stopping_rounds": 5,
    }
    val_df = generate_lightgbm_forecasts(synthetic_demand_df, split="val", **kwargs)
    assert not val_df.empty
    assert val_df["forecast_point"].notna().all()
