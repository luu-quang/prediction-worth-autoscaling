# CLAUDE.md

## Project
Implementation for a Student Forum 2026 paper: *When Is Prediction Worth It? A Trace-Driven Study of
Forecast-Aware Autoscaling Under Provisioning Delay, Forecast Uncertainty, and Server Energy
Proportionality* (USTH). Compares Static / Reactive / Forecast-aware / Oracle at matched service risk (WDR).
- Reports **estimated server-side IT energy**, never measured data-center electricity.
- Out of scope: proposing a new autoscaler; cooling/PUE; LightGBM native quantile learning, volatility
  regimes and Net Zero extensions until GO/NO-GO 1 passes (§V).
- The output is a paper: every number in it must be reproducible from this repo.

**Source of truth:** `docs/OPEN_DECISIONS_CLOSED.md` (frozen) defines all methodology. This file restates
it with section refs. If they disagree, the doc wins and CLAUDE.md is the file to fix.

## Data provenance
- **Plan B (default, `plan: plan_b`):** Zenodo DataCenter-Traces-Datasets (DOI 10.5281/zenodo.14564935),
  `machine_usage_days_1_to_8_grouped_300_seconds.csv`, derived from Alibaba cluster-trace-v2018.
  dt = 300 s. Mapping (§O): `D_t = M * C * util_t / 100`.
- **Inherited, not done by this repo:** a third party already averaged per-machine records across the
  fleet and grouped them into 300 s buckets.
- **Plan A (only if raw trace arrives):** `machine_usage.csv` joined with `machine_meta.csv`
  (`cpu_num`), cluster-trace-v2018. `D_t = sum_m cpu_num_m * cpu_util_percent_{m,t} / 100`.
  dt = 60 s only if source granularity genuinely supports it (§H); never upsample.
- **Preprocessing applied to real data by this repo: none.** Only synthetic dev data
  (`tests/conftest.py`) has passed through the pipeline; `data/processed/*` is synthetic.
- **Files on disk:** none. `data/raw/plan_a/` and `plan_b/` hold only `.gitkeep` (checked 2026-09-21).
  Raw trace files are never committed (`data/raw/*` and `data/processed/*` are gitignored).
- TODO: verify the Plan B column schema on first load (§O step 1), fill it in here, then set
  `--timestamp-col/--demand-col`; `load_demand` defaults (`timestamp`, `demand`) are placeholders.
- TODO: verify 7 complete days on load. Day 8 of v2018 is documented as incomplete; §E1 assumes 7.
- TODO: log how often `D_t ~= M*C` (fleet saturation/censoring, §O). `M` and `C` values are not yet set.
- **SPECpower (TODO, not yet used):** realism anchors for the hardware-sensitivity experiment only
  (T4b, T3). Planned: `data/raw/specpower/<spec_result_id>.csv` (one row per load level) plus a sibling
  `.md` with full result URL, SPEC result number, system name, date accessed. Neither exists.
  TODO: "Which published SPECpower_ssj2008 result is the realism anchor, and what are its
  (target load, average power) points?"

## Definitions (exact; §-refs to OPEN_DECISIONS_CLOSED.md)
- **Residual (§0):** `r = actual_future - forecast`; `r > 0` = under-forecast. Everywhere.
  Quantile-protected forecast: `f + Q_tau(r)`, residuals estimated per horizon H and per validation fold.
- **Policies and their single risk knob (§I):**
  - Static: fixed active count `N` (knob `N`).
  - Target-tracking Reactive (§J): `N_t = ceil(D_t / (C * theta))` (knob `theta`).
  - Forecast-aware: forecast + validation-residual quantile offset (knob `tau`).
  - Oracle (§K): `f = alpha * D_{t+H}`, `0 < alpha <= 1` (knob `alpha`). Non-deployable.
  - All other parameters stay fixed during one sweep. Reactive gets the same validation-tuning budget.
- **Deficit / WDR (§N):** `deficit_t = max(0, D_t - Capacity_t)`; `WDR = sum(deficit) / sum(D_t)`.
  DTR = share of timesteps with deficit > 0; PDR = `max_t deficit_t/D_t` (0 when `D_t = 0`).
  DTR and PDR are diagnostics, never matching variables. Code: `src/metrics/service_risk.py`.
- **Matched WDR (§D, §I):** targets fixed before test evaluation: `eps in {0.001, 0.005, 0.01, 0.02}`.
  Per policy and per test fold: sweep the knob, then sort by WDR, dedupe identical WDR keeping lowest
  energy, take the cumulative-min energy envelope, keep frontier breakpoints, interpolate **only on the
  envelope**. Never extrapolate; unsupported (policy, fold, eps) is excluded and `n_eps` is reported.
  Interpolate inside each fold first, then summarize; never pool folds first. Report median, IQR,
  min/max and `n_eps` (max 4 folds).
- **Folds (§E):** test days 4-7; train = days 1..k-2, validation = day k-1. Day 3 is never a primary
  test fold. Fit on train only; residual quantiles, reactive knobs, model selection on validation
  only; metrics on test only. No test-day information may set any parameter.
- **Timing (§F, §H):** fixed 2 h warm-up for every policy/delay/model/fold, excluded from all metrics;
  `N_0 = ceil(D_0 / C)`. Scale-up has provisioning delay; scale-down is immediate after cooldown and
  min-on-time. Plan B: delay in {1,3,6} steps = {5,15,30} min; `H` = provisioning delay.
- **Leakage (§P):** only Oracle and the offline synthetic error generator may use future demand. The
  simulator receives no privileged future information.
- TODO: exact rule mapping the forecast-aware `q_safe` to `N_desired` (is it `ceil(q_safe / C)`, or
  does it use a `theta`?). §I does not state it.
- TODO: confirm `Capacity_t = N_active,t * C` (booting servers excluded), and values for `M`, `C`,
  cooldown and min-on-time (not in `configs/`).

## Power model and energy (§C, §L, §M)
- Primary model is the **normalized synthetic family** `P(u) = P_peak * [k + (1 - k) * u]`,
  `k = P_idle / P_peak in {0.1, 0.3, 0.5, 0.7}` (`configs/sweeps/power_sweep.yaml`). `k` is the swept
  parameter and `C` stays fixed across the sweep. This is not a SPECpower curve.
- `E_IT = sum_t [N_t * P_server(u_t) + (M - N_t) * P_off] * dt + completed_boots * E_boot`. A boot is
  charged when startup completes. MVP may set `E_boot = 0` and/or `P_off = 0`; state it when so.
- **T4a** (exact linear-power accounting, rtol = atol = 1e-9) must pass before any result is trusted.
  Also T4b (nonlinear/SPECpower magnitude check) and T9 (synthetic == Persistence). Not written yet.

## Reproducibility
- Seed: `project.seed: 42` in `configs/base.yaml`. **No code consumes it yet.**
- The simulator contains **no RNG**: given `(demand, forecast, policy_cfg, sim_cfg)` its output must be
  bit-identical across runs. Seeds belong only to the synthetic error generator and model training.
- Every result row stores `seed` and `experiment_id`/config hash; required row schema is §T (do not retype).
- One config (YAML) per experiment family; preprocessing cached as Parquet.
- **Single regenerate-everything command:** `python scripts/run_all.py --config configs/base.yaml`
  - **NOT IMPLEMENTED as of 2026-09-21.** `scripts/make_figures.py`, `run_baselines.py`,
    `run_matched_wdr.py`, `run_synthetic_sweep.py` are placeholders and should converge on this entry point.
  - **No number, figure or table enters the paper unless it came from this command.** No Makefile/CI yet.

## Commands
```bash
python3.12 -m venv .venv && source .venv/bin/activate    # Python 3.12 (.python-version = 3.12.3)
pip install -r requirements.txt
python scripts/prepare_data.py [--raw-path P]   # raw -> data/processed/demand/demand.parquet
python scripts/train_forecasts.py               # Persistence forecasts + MAE/RMSE (Track A steps 1-5)
pytest -q
ruff check . && ruff format .                   # ruff for lint + format; no mypy yet
```

## Code conventions (as used)
- Python 3.12, ruff, line length 100 (`pyproject.toml`). Imports as `from src.<pkg>.<mod> import ...`;
  scripts prepend the repo root to `sys.path`.
- Layout: `src/{data,forecasting,simulator,policies,energy,metrics,evaluation,utils}`, `scripts/`,
  `configs/` (+ `sweeps/`), `tests/`, `docs/`, `figures/F1..F6`, `results/tables`, `paper/`.
- Plain functions and frozen dataclasses (`Fold`); pandas DataFrames with locked schemas in
  `src/data/schema.py` (`demand_df`, `forecast_df`); custom errors (`DataQualityError`, `SchemaError`)
  that fail loudly instead of silently cleaning data. Tests: pytest, shared fixtures in `tests/conftest.py`.
- Track A/B interface: `simulate(demand_df, forecast_df, policy_cfg, sim_cfg) -> trace_df`
  (`src/simulator/simulator.py`, currently `NotImplementedError`). Branches `track-a-forecasting` /
  `track-b-simulator`; do not push unfinished work to `main`.

## Rules for Claude
- Do not change experiment parameters (`configs/`, WDR targets, folds, grids) or metric definitions
  without asking first.
- Never hand-edit files in `results/`, `figures/`, `paper/tables/` or `data/processed/`; regenerate them.
- Never invent numbers for the paper. If a value is missing, say so and leave a TODO.
- Do not commit raw trace data. Do not commit or push unless asked.
