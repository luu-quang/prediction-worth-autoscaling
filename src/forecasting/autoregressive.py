import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data.folds import assign_day_index
from src.data.schema import FORECAST_COLUMNS, validate_demand_df

MODEL_ID = "autoregressive"

# lag_k = demand shifted (k-1) steps, so lag_1 = D_t (the forecast origin itself),
# lag_2 = D_{t-1}, ..., lag_12 = D_{t-11}. Not stated explicitly in
# team-a-forecasting/README.md Step 7; chosen so lag_1 matches the same origin
# point Persistence and EWMA use.
LAGS = (1, 2, 3, 6, 12)


def build_lag_features(demand: pd.Series, lags=LAGS) -> pd.DataFrame:
    return pd.DataFrame({f"lag_{k}": demand.shift(k - 1) for k in lags})


def _fit_model(
    features: pd.DataFrame, target: pd.Series, train_mask: pd.Series
) -> LinearRegression:
    """Fit on rows where train_mask is True only; never sees val/test rows."""
    train_X = features.loc[train_mask]
    train_y = target.loc[train_mask]
    valid_train = train_X.notna().all(axis=1) & train_y.notna()
    if valid_train.sum() < len(features.columns) + 1:
        raise ValueError(f"not enough train rows: {int(valid_train.sum())} valid rows")

    model = LinearRegression()
    model.fit(train_X.loc[valid_train], train_y.loc[valid_train])
    return model


def generate_autoregressive_forecasts(demand_df, horizons_steps, folds, lags=LAGS) -> pd.DataFrame:
    """Linear regression on lag features, fit per (fold, horizon) on train days only.

    Fitting uses only rows whose forecast origin t falls in the fold's train days
    (docs/OPEN_DECISIONS_CLOSED.md §E2: model fitting -> train only). No test-day
    demand is ever used to fit or select the model (§P). The last train day's final
    few rows can have their label (target = demand.shift(-horizon)) land on the
    following (validation) day; this is a normal walk-forward boundary effect, not
    test-day leakage, since the validation day is never the test day.
    """
    validate_demand_df(demand_df)
    demand_df = demand_df.sort_values("timestamp").reset_index(drop=True)
    day_index = assign_day_index(demand_df["timestamp"])
    demand = demand_df["demand"]
    features = build_lag_features(demand, lags)

    rows = []
    for horizon in horizons_steps:
        target = demand.shift(-horizon)  # D_{t+H} aligned at origin row t
        for fold in folds:
            train_mask = day_index.isin(fold.train_days)
            test_mask = day_index == fold.test_day
            if not train_mask.any() or not test_mask.any():
                continue

            try:
                model = _fit_model(features, target, train_mask)
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
