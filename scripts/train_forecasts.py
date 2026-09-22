"""Track A Steps 2-9: folds, Persistence + EWMA + AR + LightGBM, MAE/RMSE, residual quantiles.

Usage:
    python scripts/train_forecasts.py

Reads data/processed/demand/demand.parquet (produced by prepare_data.py),
exports data/processed/forecasts/{persistence,ewma,autoregressive,lightgbm}.parquet
and data/processed/forecasts/residual_quantiles.parquet, and prints the MAE/RMSE
summary per model/fold/horizon.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.folds import generate_folds
from src.data.schema import FORECAST_COLUMNS
from src.forecasting.autoregressive import generate_autoregressive_forecasts
from src.forecasting.ewma import generate_ewma_forecasts
from src.forecasting.lightgbm_model import generate_lightgbm_forecasts
from src.forecasting.persistence import generate_persistence_forecasts
from src.forecasting.residual_quantiles import compute_all_residual_quantiles
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

    seed = base_cfg["project"]["seed"]

    alpha_grid = forecasting_cfg.get("ewma", {}).get("alpha_grid")
    ewma_kwargs = {"alpha_grid": alpha_grid} if alpha_grid else {}

    lgb_cfg = forecasting_cfg.get("lightgbm", {})
    lgb_kwargs = {}
    if lgb_cfg.get("rolling_windows"):
        lgb_kwargs["rolling_windows"] = tuple(lgb_cfg["rolling_windows"])
    if lgb_cfg.get("early_stopping_rounds"):
        lgb_kwargs["early_stopping_rounds"] = lgb_cfg["early_stopping_rounds"]
    if lgb_cfg.get("params"):
        lgb_kwargs["lgb_params"] = lgb_cfg["params"]

    model_generators = {
        "persistence": lambda: generate_persistence_forecasts(demand_df, horizons_steps, folds),
        "ewma": lambda: generate_ewma_forecasts(demand_df, horizons_steps, folds, **ewma_kwargs),
        "autoregressive": lambda: generate_autoregressive_forecasts(
            demand_df, horizons_steps, folds
        ),
        "lightgbm": lambda: generate_lightgbm_forecasts(
            demand_df, horizons_steps, folds, seed=seed, **lgb_kwargs
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

    tau_grid = forecasting_cfg.get("residual_quantiles", {}).get("tau_grid")
    rq_kwargs = {"tau_grid": tuple(tau_grid)} if tau_grid else {}
    quantiles_df = compute_all_residual_quantiles(
        demand_df,
        horizons_steps,
        folds,
        model_kwargs={"ewma": ewma_kwargs, "lightgbm": {**lgb_kwargs, "seed": seed}},
        **rq_kwargs,
    )
    quantiles_path = out_dir / "residual_quantiles.parquet"
    quantiles_df.to_parquet(quantiles_path, index=False)
    print(f"Wrote {len(quantiles_df)} rows to {quantiles_path}")


if __name__ == "__main__":
    main()
