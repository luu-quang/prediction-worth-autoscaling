import numpy as np
import pandas as pd


# Locked residual convention: residual > 0 means under-forecasting.
def residual(actual_demand, forecast_point):
    return np.asarray(actual_demand, dtype=float) - np.asarray(forecast_point, dtype=float)


def mae(actual_demand, forecast_point) -> float:
    return float(np.mean(np.abs(residual(actual_demand, forecast_point))))


def rmse(actual_demand, forecast_point) -> float:
    return float(np.sqrt(np.mean(residual(actual_demand, forecast_point) ** 2)))


def summarize_forecast_errors(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """MAE/RMSE per model_id, fold_id, horizon_steps."""
    rows = []
    group_cols = ["model_id", "fold_id", "horizon_steps"]
    for keys, group in forecast_df.groupby(group_cols):
        model_id, fold_id, horizon_steps = keys
        rows.append(
            {
                "model_id": model_id,
                "fold_id": fold_id,
                "horizon_steps": horizon_steps,
                "mae": mae(group["actual_demand"], group["forecast_point"]),
                "rmse": rmse(group["actual_demand"], group["forecast_point"]),
                "n": len(group),
            }
        )
    return pd.DataFrame(rows, columns=group_cols + ["mae", "rmse", "n"])
