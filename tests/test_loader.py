import pandas as pd
import pytest

from src.data.loader import DataQualityError, load_demand


def _write_csv(tmp_path, df, name="demand.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def test_load_demand_happy_path(tmp_path, synthetic_demand_df):
    path = _write_csv(tmp_path, synthetic_demand_df)
    loaded = load_demand(path, dt_seconds=300)
    assert list(loaded.columns) == ["timestamp", "demand"]
    assert pd.api.types.is_datetime64_any_dtype(loaded["timestamp"])
    assert len(loaded) == len(synthetic_demand_df)


def test_load_demand_rejects_duplicate_timestamps(tmp_path, synthetic_demand_df):
    bad = pd.concat([synthetic_demand_df, synthetic_demand_df.iloc[[0]]], ignore_index=True)
    path = _write_csv(tmp_path, bad)
    with pytest.raises(DataQualityError, match="duplicate timestamp"):
        load_demand(path, dt_seconds=300)


def test_load_demand_rejects_nan(tmp_path, synthetic_demand_df):
    bad = synthetic_demand_df.copy()
    bad.loc[10, "demand"] = None
    path = _write_csv(tmp_path, bad)
    with pytest.raises(DataQualityError, match="NaN"):
        load_demand(path, dt_seconds=300)


def test_load_demand_rejects_negative_demand(tmp_path, synthetic_demand_df):
    bad = synthetic_demand_df.copy()
    bad.loc[10, "demand"] = -5.0
    path = _write_csv(tmp_path, bad)
    with pytest.raises(DataQualityError, match="negative"):
        load_demand(path, dt_seconds=300)


def test_load_demand_rejects_bad_timestep(tmp_path, synthetic_demand_df):
    bad = synthetic_demand_df.copy()
    bad = bad.drop(index=5).reset_index(drop=True)  # creates a gap
    path = _write_csv(tmp_path, bad)
    with pytest.raises(DataQualityError, match="dt_seconds"):
        load_demand(path, dt_seconds=300)
