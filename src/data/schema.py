import pandas as pd

DEMAND_COLUMNS = ["timestamp", "demand"]

FORECAST_COLUMNS = [
    "timestamp",
    "actual_demand",
    "forecast_point",
    "horizon_steps",
    "fold_id",
    "model_id",
]


class SchemaError(ValueError):
    """Raised when a dataframe does not conform to the locked schema."""


def validate_demand_df(df: pd.DataFrame) -> None:
    missing = [c for c in DEMAND_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"demand_df missing required columns: {missing}")
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise SchemaError("demand_df['timestamp'] must be datetime64")


def validate_forecast_df(df: pd.DataFrame) -> None:
    missing = [c for c in FORECAST_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"forecast_df missing required columns: {missing}")
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise SchemaError("forecast_df['timestamp'] must be datetime64")
