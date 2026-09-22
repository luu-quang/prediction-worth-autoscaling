import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data.folds import assign_day_index
from src.data.schema import FORECAST_COLUMNS, validate_demand_df
from src.forecasting.features import DEFAULT_ROLLING_WINDOWS, build_features

MODEL_ID = "lightgbm"

# Not locked in docs/OPEN_DECISIONS_CLOSED.md; overridable via configs/forecasting.yaml lightgbm.*.
DEFAULT_LGB_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 20,
    "verbosity": -1,
}
DEFAULT_EARLY_STOPPING_ROUNDS = 20


def _fit_model(
    features: pd.DataFrame,
    target: pd.Series,
    train_mask: pd.Series,
    val_mask: pd.Series,
    seed: int,
    lgb_params: dict,
    early_stopping_rounds: int,
) -> lgb.LGBMRegressor:
    """Fit on train rows only; early stopping uses the validation rows only
    (docs/OPEN_DECISIONS_CLOSED.md §E2: model selection / early stopping -> validation only).
    """
    train_X, train_y = features.loc[train_mask], target.loc[train_mask]
    valid_train = train_X.notna().all(axis=1) & train_y.notna()
    val_X, val_y = features.loc[val_mask], target.loc[val_mask]
    valid_val = val_X.notna().all(axis=1) & val_y.notna()

    if valid_train.sum() < 2 or valid_val.sum() < 1:
        raise ValueError(
            f"not enough rows to fit: {int(valid_train.sum())} train, {int(valid_val.sum())} val"
        )

    model = lgb.LGBMRegressor(random_state=seed, **lgb_params)
    model.fit(
        train_X.loc[valid_train],
        train_y.loc[valid_train],
        eval_X=val_X.loc[valid_val],
        eval_y=val_y.loc[valid_val],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
    )
    return model


def generate_lightgbm_forecasts(
    demand_df,
    horizons_steps,
    folds,
    seed: int = 42,
    lags=None,
    rolling_windows=DEFAULT_ROLLING_WINDOWS,
    lgb_params: dict | None = None,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
) -> pd.DataFrame:
    """LightGBM forecasts per test fold: fit on train days, early-stopped on the
    validation day, predicted on the test day. No test-day information is ever
    used to fit or select the model (§P).
    """
    validate_demand_df(demand_df)
    demand_df = demand_df.sort_values("timestamp").reset_index(drop=True)
    day_index = assign_day_index(demand_df["timestamp"])
    demand = demand_df["demand"]
    feature_kwargs = {"rolling_windows": rolling_windows}
    if lags is not None:
        feature_kwargs["lags"] = lags
    features = build_features(demand, **feature_kwargs)
    params = {**DEFAULT_LGB_PARAMS, **(lgb_params or {})}

    rows = []
    for horizon in horizons_steps:
        target = demand.shift(-horizon)  # D_{t+H} aligned at origin row t
        for fold in folds:
            train_mask = day_index.isin(fold.train_days)
            val_mask = day_index == fold.val_day
            test_mask = day_index == fold.test_day
            if not train_mask.any() or not val_mask.any() or not test_mask.any():
                continue

            try:
                model = _fit_model(
                    features, target, train_mask, val_mask, seed, params, early_stopping_rounds
                )
            except ValueError as e:
                raise ValueError(f"fold={fold.fold_id} horizon={horizon}: {e}") from e

            valid_features = features.notna().all(axis=1)
            predictions_at_origin = pd.Series(np.nan, index=demand.index)
            predictions_at_origin.loc[valid_features] = model.predict(features.loc[valid_features])
            forecast_point = predictions_at_origin.shift(horizon)

            rows.append(
                pd.DataFrame(
                    {
                        "timestamp": demand_df.loc[test_mask, "timestamp"],
                        "actual_demand": demand_df.loc[test_mask, "demand"],
                        "forecast_point": forecast_point.loc[test_mask],
                        "horizon_steps": horizon,
                        "fold_id": fold.fold_id,
                        "model_id": MODEL_ID,
                    }
                )
            )

    if rows:
        forecast_df = pd.concat(rows, ignore_index=True)
    else:
        forecast_df = pd.DataFrame(columns=FORECAST_COLUMNS)

    forecast_df = forecast_df.dropna(subset=["forecast_point"]).reset_index(drop=True)
    return forecast_df[FORECAST_COLUMNS]
