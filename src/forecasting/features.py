import pandas as pd

from src.forecasting.autoregressive import LAGS, build_lag_features

# Not locked in docs/OPEN_DECISIONS_CLOSED.md; overridable via
# configs/forecasting.yaml lightgbm.rolling_windows. 6/12 steps = 30min/1h at dt=300s.
DEFAULT_ROLLING_WINDOWS = (6, 12)


def rolling_mean(demand: pd.Series, window: int) -> pd.Series:
    """Trailing window (never centered): uses only demand up to and including t."""
    return demand.rolling(window=window, min_periods=window).mean()


def rolling_std(demand: pd.Series, window: int) -> pd.Series:
    return demand.rolling(window=window, min_periods=window).std()


def rolling_max(demand: pd.Series, window: int) -> pd.Series:
    return demand.rolling(window=window, min_periods=window).max()


def recent_slope(demand: pd.Series, window: int) -> pd.Series:
    """Causal linear trend over the trailing window: (D_t - D_{t-window+1}) / (window-1)."""
    return (demand - demand.shift(window - 1)) / (window - 1)


def build_features(
    demand: pd.Series, lags=LAGS, rolling_windows=DEFAULT_ROLLING_WINDOWS
) -> pd.DataFrame:
    """Lag + rolling-window features, all trailing/causal (no centered windows)."""
    features = build_lag_features(demand, lags)
    for window in rolling_windows:
        features[f"rolling_mean_{window}"] = rolling_mean(demand, window)
        features[f"rolling_std_{window}"] = rolling_std(demand, window)
        features[f"rolling_max_{window}"] = rolling_max(demand, window)
        features[f"slope_{window}"] = recent_slope(demand, window)
    return features
