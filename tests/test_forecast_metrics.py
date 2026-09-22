import math

import pandas as pd
import pytest

from src.metrics.forecast_metrics import mae, residual, rmse, summarize_forecast_errors


def test_residual_sign_convention():
    # under-forecast: actual > forecast -> residual > 0
    assert residual([10.0], [8.0])[0] > 0
    # over-forecast: actual < forecast -> residual < 0
    assert residual([8.0], [10.0])[0] < 0


def test_mae_and_rmse_known_values():
    actual = [1.0, 2.0, 3.0]
    forecast = [1.0, 1.0, 1.0]
    # residuals = [0, 1, 2]
    assert mae(actual, forecast) == pytest.approx(1.0)
    assert rmse(actual, forecast) == pytest.approx(math.sqrt(5 / 3))


def test_summarize_forecast_errors_groups_correctly():
    forecast_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"] * 4),
            "actual_demand": [10.0, 20.0, 10.0, 20.0],
            "forecast_point": [10.0, 18.0, 5.0, 20.0],
            "horizon_steps": [1, 1, 3, 3],
            "fold_id": ["fold_4", "fold_4", "fold_4", "fold_4"],
            "model_id": ["persistence", "persistence", "persistence", "persistence"],
        }
    )
    summary = summarize_forecast_errors(forecast_df)
    assert len(summary) == 2  # one row per horizon within the single fold/model

    h1 = summary[summary["horizon_steps"] == 1].iloc[0]
    assert h1["mae"] == pytest.approx(1.0)  # |0| and |2| averaged
    assert h1["n"] == 2
