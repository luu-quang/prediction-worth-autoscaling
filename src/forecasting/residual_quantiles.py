import numpy as np
import pandas as pd

from src.forecasting.autoregressive import generate_autoregressive_forecasts
from src.forecasting.ewma import generate_ewma_forecasts
from src.forecasting.lightgbm_model import generate_lightgbm_forecasts
from src.forecasting.persistence import generate_persistence_forecasts
from src.metrics.forecast_metrics import residual

# Not locked in docs/OPEN_DECISIONS_CLOSED.md. Candidate tau grid for the
# forecast-aware policy's residual-quantile safety offset (§0, §I):
#   safe_forecast = point_forecast + quantile(validation_residuals, tau)
DEFAULT_TAU_GRID = (0.5, 0.75, 0.9, 0.95, 0.99)

# Each generator must accept (demand_df, horizons_steps, folds, split="val", **kwargs)
# and return a forecast_df restricted to that split (see each module's split param).
MODEL_GENERATORS = {
    "persistence": generate_persistence_forecasts,
    "ewma": generate_ewma_forecasts,
    "autoregressive": generate_autoregressive_forecasts,
    "lightgbm": generate_lightgbm_forecasts,
}


def compute_validation_residuals(val_forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Residuals (actual - forecast, locked sign convention) on validation-split rows.

    val_forecast_df must come from a model's generate_*_forecasts(..., split="val")
    call, so it is already restricted to the validation day of each fold. Never
    pass a split="test" (or mixed) forecast_df here (§P: residual-quantile
    calibration must come from validation, never test).
    """
    out = val_forecast_df[["model_id", "fold_id", "horizon_steps"]].copy()
    out["residual"] = residual(val_forecast_df["actual_demand"], val_forecast_df["forecast_point"])
    return out


def residual_quantiles(residuals_df: pd.DataFrame, tau_grid=DEFAULT_TAU_GRID) -> pd.DataFrame:
    """Q_tau(validation residuals) per (model_id, fold_id, horizon_steps, tau).

    Used later as: safe_forecast = point_forecast + Q_tau(r), r estimated
    separately for each horizon and each validation fold (§0).
    """
    rows = []
    group_cols = ["model_id", "fold_id", "horizon_steps"]
    for keys, group in residuals_df.groupby(group_cols):
        model_id, fold_id, horizon_steps = keys
        for tau in tau_grid:
            rows.append(
                {
                    "model_id": model_id,
                    "fold_id": fold_id,
                    "horizon_steps": horizon_steps,
                    "tau": tau,
                    "quantile_value": float(np.quantile(group["residual"], tau)),
                    "n": len(group),
                }
            )
    return pd.DataFrame(rows, columns=[*group_cols, "tau", "quantile_value", "n"])


def compute_all_residual_quantiles(
    demand_df,
    horizons_steps,
    folds,
    tau_grid=DEFAULT_TAU_GRID,
    model_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Validation-day residual quantiles for every model in MODEL_GENERATORS.

    model_kwargs, if given, maps model_id -> extra kwargs for that model's
    generator (e.g. {"lightgbm": {"seed": 42}}).
    """
    model_kwargs = model_kwargs or {}
    residuals = []
    for model_id, generate in MODEL_GENERATORS.items():
        val_forecast_df = generate(
            demand_df, horizons_steps, folds, split="val", **model_kwargs.get(model_id, {})
        )
        residuals.append(compute_validation_residuals(val_forecast_df))

    residuals_df = pd.concat(residuals, ignore_index=True)
    return residual_quantiles(residuals_df, tau_grid)
