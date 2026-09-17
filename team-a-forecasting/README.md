# Team A — Forecasting

**Owner:** Viet-Duc Hoang  
**Branch:** `track-a-forecasting`

## Goal
Build the forecasting pipeline and export standardized forecasts for Team B.

## Step 1 — Prepare data
Work mainly in:
- `src/data/`
- `data/processed/demand/`

Tasks:
- load Plan-B workload data
- sort timestamps
- check duplicates / NaN / negative demand
- verify timestep
- save `data/processed/demand/demand.parquet`

## Step 2 — Walk-forward folds
Implement `src/data/folds.py`.

Primary folds:
- test day 4: train days 1–2, val day 3
- test day 5: train days 1–3, val day 4
- test day 6: train days 1–4, val day 5
- test day 7: train days 1–5, val day 6

Never use random train/test split.

## Step 3 — Persistence
Implement `src/forecasting/persistence.py`.

For Plan B:
- H=1 step = 5 min
- H=3 steps = 15 min
- H=6 steps = 30 min

Definition:
`forecast(t+H) = demand(t)`

## Step 4 — Common output schema
Every forecast file must contain:
- `timestamp`
- `actual_demand`
- `forecast_point`
- `horizon_steps`
- `fold_id`
- `model_id`

Save to `data/processed/forecasts/`.

## Step 5 — Metrics
Implement MAE and RMSE.

Residual convention:
`residual = actual_future - forecast`

So:
- residual > 0 = under-forecast
- residual < 0 = over-forecast

## Step 6 — EWMA
Implement `src/forecasting/ewma.py`.

Tune only on validation data.

## Step 7 — Linear / AR
Implement `src/forecasting/autoregressive.py`.

Start with:
- lag_1
- lag_2
- lag_3
- lag_6
- lag_12

No future information.

## Step 8 — LightGBM
Implement:
- `src/forecasting/features.py`
- `src/forecasting/lightgbm_model.py`

Start with:
- lag features
- rolling mean
- rolling std
- rolling max
- recent slope

No centered rolling windows.

## Step 9 — Residual quantiles
For each model/fold/horizon, compute validation residuals.

Later:
`safe_forecast = point_forecast + quantile(validation_residuals, tau)`

Never tune with test residuals.

## Step 10 — Synthetic generator
Only after the real forecasting models work.

Implement `src/forecasting/synthetic_errors.py`.

Use the locked residual/sign convention and export the same forecast schema.

## Step 11 — T9
Synthetic setting corresponding to Persistence must reproduce Persistence exactly.

## Deliverables
- `data/processed/demand/demand.parquet`
- `data/processed/forecasts/persistence.parquet`
- `data/processed/forecasts/ewma.parquet`
- `data/processed/forecasts/autoregressive.parquet`
- `data/processed/forecasts/lightgbm.parquet`

Also provide:
`model_id, fold_id, horizon_steps, MAE, RMSE`

## Definition of Done
- [ ] demand cleaned
- [ ] 4 folds built
- [ ] Persistence done
- [ ] EWMA done
- [ ] Linear/AR done
- [ ] LightGBM done
- [ ] MAE/RMSE done
- [ ] no future leakage
- [ ] standardized forecast files exported

## Git
```bash
git checkout track-a-forecasting
git pull origin track-a-forecasting
git add .
git commit -m "Track A: <what changed>"
git push origin track-a-forecasting
```

Do not push unfinished work directly to `main`.
