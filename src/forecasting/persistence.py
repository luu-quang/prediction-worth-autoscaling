import pandas as pd

from src.data.folds import assign_day_index
from src.data.schema import FORECAST_COLUMNS, validate_demand_df

MODEL_ID = "persistence"


def persistence_forecast(demand: pd.Series, horizon_steps: int) -> pd.Series:
    """forecast(t+H) = demand(t), returned aligned to the target timestep t+H."""
    return demand.shift(horizon_steps)


def generate_persistence_forecasts(demand_df, horizons_steps, folds) -> pd.DataFrame:
    """Persistence forecasts for the primary test days, in the locked forecast schema.

    Persistence has no parameters to fit, so only test-day rows are exported:
    train/val days aren't needed to produce its forecasts, and the matched-WDR
    comparison (T2) only consumes test-day forecasts.
    """
    validate_demand_df(demand_df)
    demand_df = demand_df.sort_values("timestamp").reset_index(drop=True)
    day_index = assign_day_index(demand_df["timestamp"])

    rows = []
    for horizon in horizons_steps:
        forecast_point = persistence_forecast(demand_df["demand"], horizon)
        for fold in folds:
            mask = day_index == fold.test_day
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
                    }
                )
            )

    if rows:
        forecast_df = pd.concat(rows, ignore_index=True)
    else:
        forecast_df = pd.DataFrame(columns=FORECAST_COLUMNS)

    forecast_df = forecast_df.dropna(subset=["forecast_point"]).reset_index(drop=True)
    return forecast_df[FORECAST_COLUMNS]
