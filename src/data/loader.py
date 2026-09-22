from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schema import SchemaError, validate_demand_df


class DataQualityError(ValueError):
    """Raised when raw demand data fails a quality check (Track A Step 1)."""


def _read_raw(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise SchemaError(f"unsupported raw data format: {path.suffix}")


def load_demand(
    path,
    timestamp_col: str = "timestamp",
    demand_col: str = "demand",
    dt_seconds: int = 300,
) -> pd.DataFrame:
    """Load, validate, and standardize a raw Plan-B/Plan-A demand trace.

    Raises DataQualityError rather than silently dropping or fixing rows,
    so data-quality issues in a real trace surface immediately instead of
    being papered over.
    """
    path = Path(path)
    raw = _read_raw(path)

    missing = [c for c in (timestamp_col, demand_col) if c not in raw.columns]
    if missing:
        raise SchemaError(f"raw file missing expected columns: {missing}")

    df = raw[[timestamp_col, demand_col]].rename(
        columns={timestamp_col: "timestamp", demand_col: "demand"}
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if df["timestamp"].duplicated().any():
        n_dup = int(df["timestamp"].duplicated().sum())
        raise DataQualityError(f"{n_dup} duplicate timestamp(s) found")

    if df[["timestamp", "demand"]].isna().any().any():
        n_nan = int(df["demand"].isna().sum())
        raise DataQualityError(f"{n_nan} NaN demand value(s) found")

    if (df["demand"] < 0).any():
        n_neg = int((df["demand"] < 0).sum())
        raise DataQualityError(f"{n_neg} negative demand value(s) found")

    deltas = df["timestamp"].diff().dropna().dt.total_seconds()
    if not deltas.empty:
        expected = pd.Series([dt_seconds] * len(deltas), index=deltas.index)
        if not np.isclose(deltas.values, expected.values).all():
            bad = deltas[~np.isclose(deltas.values, expected.values)]
            raise DataQualityError(
                f"{len(bad)} timestep(s) do not match dt_seconds={dt_seconds}; "
                f"e.g. observed gaps: {sorted(bad.unique())[:5]}"
            )

    validate_demand_df(df)
    return df
