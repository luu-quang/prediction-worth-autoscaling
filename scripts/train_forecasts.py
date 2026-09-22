"""Track A Steps 2-7: walk-forward folds, Persistence + EWMA + AR forecasts, MAE/RMSE.

Usage:
    python scripts/train_forecasts.py

Reads data/processed/demand/demand.parquet (produced by prepare_data.py),
exports data/processed/forecasts/{persistence,ewma,autoregressive}.parquet,
and prints the MAE/RMSE summary per model/fold/horizon.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.folds import generate_folds
from src.data.schema import FORECAST_COLUMNS
from src.forecasting.autoregressive import generate_autoregressive_forecasts
from src.forecasting.ewma import generate_ewma_forecasts
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

    out_dir = REPO_ROOT / "data" / "processed" / "forecasts"
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha_grid = forecasting_cfg.get("ewma", {}).get("alpha_grid")
    ewma_kwargs = {"alpha_grid": alpha_grid} if alpha_grid else {}

    model_generators = {
        "persistence": lambda: generate_persistence_forecasts(demand_df, horizons_steps, folds),
        "ewma": lambda: generate_ewma_forecasts(demand_df, horizons_steps, folds, **ewma_kwargs),
        "autoregressive": lambda: generate_autoregressive_forecasts(
            demand_df, horizons_steps, folds
        ),
    }

    forecast_dfs = []
    for name, generate in model_generators.items():
        forecast_df = generate()
        out_path = out_dir / f"{name}.parquet"
        forecast_df.to_parquet(out_path, index=False)
        print(f"Wrote {len(forecast_df)} rows to {out_path}")
        forecast_dfs.append(forecast_df[FORECAST_COLUMNS])

    summary = summarize_forecast_errors(pd.concat(forecast_dfs, ignore_index=True))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
