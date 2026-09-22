import pandas as pd

from src.data.folds import generate_folds
from src.data.schema import validate_forecast_df
from src.forecasting.persistence import generate_persistence_forecasts, persistence_forecast


def test_persistence_forecast_shifts_by_horizon():
    demand = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    forecast = persistence_forecast(demand, horizon_steps=2)
    # forecast(t+H) = demand(t): row index 2 (t+H) should carry demand at t=0
    assert forecast.iloc[2] == demand.iloc[0]
    assert pd.isna(forecast.iloc[0])
    assert pd.isna(forecast.iloc[1])


def test_generate_persistence_forecasts_schema_and_values(synthetic_demand_df):
    folds = generate_folds()
    forecast_df = generate_persistence_forecasts(
        synthetic_demand_df, horizons_steps=[1, 3, 6], folds=folds
    )
    validate_forecast_df(forecast_df)

    assert set(forecast_df["model_id"]) == {"persistence"}
    assert set(forecast_df["horizon_steps"]) == {1, 3, 6}
    assert set(forecast_df["fold_id"]) == {"fold_4", "fold_5", "fold_6", "fold_7"}

    # forecast_point at horizon H must equal demand exactly H steps earlier
    demand_by_ts = synthetic_demand_df.set_index("timestamp")["demand"]
    for horizon in [1, 3, 6]:
        rows = forecast_df[forecast_df["horizon_steps"] == horizon]
        origin_ts = rows["timestamp"] - pd.Timedelta(seconds=300 * horizon)
        expected = demand_by_ts.reindex(origin_ts).to_numpy()
        assert (rows["forecast_point"].to_numpy() == expected).all()


def test_generate_persistence_forecasts_excludes_day_3(synthetic_demand_df):
    folds = generate_folds()
    forecast_df = generate_persistence_forecasts(
        synthetic_demand_df, horizons_steps=[1], folds=folds
    )
    # day 3 only ever appears as a validation day in the primary folds, never test
    assert "fold_3" not in set(forecast_df["fold_id"])
