import pandas as pd
import pytest

from src.forecasting.features import (
    build_features,
    recent_slope,
    rolling_max,
    rolling_mean,
    rolling_std,
)


def test_rolling_stats_are_trailing_not_centered():
    demand = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    window = 3
    mean = rolling_mean(demand, window)
    # value at t=2 (0-indexed) uses only demand[0:3] = [1,2,3], never demand[3] or [4]
    assert mean.iloc[2] == pytest.approx(2.0)
    assert pd.isna(mean.iloc[0])
    assert pd.isna(mean.iloc[1])

    std = rolling_std(demand, window)
    assert std.iloc[2] == pytest.approx(demand.iloc[0:3].std())

    rmax = rolling_max(demand, window)
    assert rmax.iloc[2] == 3.0
    assert rmax.iloc[4] == 5.0


def test_recent_slope_causal():
    demand = pd.Series([0.0, 2.0, 4.0, 6.0, 8.0])
    slope = recent_slope(demand, window=3)
    # over rows [0,1,2]=[0,2,4]: (4-0)/2 = 2.0
    assert slope.iloc[2] == pytest.approx(2.0)
    assert pd.isna(slope.iloc[1])


def test_build_features_includes_lags_and_rolling_columns():
    demand = pd.Series(range(30), dtype=float)
    features = build_features(demand, lags=(1, 2), rolling_windows=(6,))
    assert set(features.columns) == {
        "lag_1",
        "lag_2",
        "rolling_mean_6",
        "rolling_std_6",
        "rolling_max_6",
        "slope_6",
    }
    assert len(features) == len(demand)
