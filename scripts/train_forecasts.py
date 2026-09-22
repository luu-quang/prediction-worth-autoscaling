"""Track A Steps 2-5: walk-forward folds, Persistence forecasts, MAE/RMSE.

Usage:
    python scripts/train_forecasts.py

Reads data/processed/demand/demand.parquet (produced by prepare_data.py),
exports data/processed/forecasts/persistence.parquet, and prints the
MAE/RMSE summary per model/fold/horizon.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.folds import generate_folds
from src.forecasting.persistence import generate_persistence_forecasts
from src.metrics.forecast_metrics import summarize_forecast_errors
from src.utils.config import horizons_to_steps, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    base_cfg = load_config("base.yaml")
    forecasting_cfg = load_config("forecasting.yaml")
    dt_seconds = base_cfg["data"]["dt_seconds"]

    demand_path = REPO_ROOT / "data" / "processed" / "demand" / "demand.parquet"
    if not demand_path.exists():
        print(
            f"{demand_path} not found. Run scripts/prepare_data.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    demand_df = pd.read_parquet(demand_path)
    folds = generate_folds(test_days=tuple(base_cfg["evaluation"]["primary_test_days"]))
    horizons_steps = horizons_to_steps(
        forecasting_cfg["forecasting"]["horizons_minutes"], dt_seconds
    )

    forecast_df = generate_persistence_forecasts(demand_df, horizons_steps, folds)

    out_dir = REPO_ROOT / "data" / "processed" / "forecasts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "persistence.parquet"
    forecast_df.to_parquet(out_path, index=False)
    print(f"Wrote {len(forecast_df)} rows to {out_path}")

    summary = summarize_forecast_errors(forecast_df)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
