# Prediction Worth Autoscaling

Trace-driven evaluation of when workload prediction improves energy-efficient cloud autoscaling under provisioning delay and forecast uncertainty.

## Paper title

**When Is Prediction Worth It? A Trace-Driven Study of Forecast-Aware Autoscaling Under Provisioning Delay, Forecast Uncertainty, and Server Energy Proportionality**

## Core question

When does workload prediction actually provide enough operational value to justify predictive autoscaling?

This repository does **not** aim to propose another autoscaling algorithm. Instead, it compares Static, target-tracking Reactive, forecast-aware, and Oracle policies under controlled provisioning delay, forecast error, workload volatility, and server power characteristics.

## Research questions

- **RQ1:** At the same service-risk level, does forecast-aware autoscaling reduce estimated server-side IT energy compared with reactive autoscaling?
- **RQ2:** How do provisioning delay, forecast error, workload volatility, and energy proportionality change the benefit of prediction?
- **RQ3:** Does better forecast accuracy (MAE/RMSE) reliably translate into better downstream energy-service performance?

## Pipeline

```text
Workload trace
    ↓
Forecasting
    ↓
Provisioning policy
    ↓
Provisioning delay
    ↓
Active servers
    ↓
Served workload / deficit
    ↓
WDR / DTR / PDR
    ↓
Server-hours
    ↓
Estimated server-side IT energy
```

## Team split

### Track A — Forecasting
- preprocessing and walk-forward folds
- Persistence
- EWMA
- Linear / autoregressive baseline
- LightGBM
- MAE / RMSE
- residual analysis
- synthetic error generator

Track A exports:

```text
timestamp
actual_demand
forecast_point
horizon_steps
fold_id
model_id
```

### Track B — Simulator / server / energy
- server state transitions
- provisioning delay
- Static / Reactive / Oracle
- forecast-aware policy integration
- WDR / DTR / PDR
- server-hours
- power curves
- energy accounting
- matched-WDR evaluation

## Walk-forward evaluation

For seven complete days, primary paper folds are:

```text
Test day 4: train days 1–2, validation day 3
Test day 5: train days 1–3, validation day 4
Test day 6: train days 1–4, validation day 5
Test day 7: train days 1–5, validation day 6
```

The day-3 test fold is excluded from primary analysis.

## Residual convention

```text
residual = actual_future - forecast
```

So:
- residual > 0 → under-forecast
- residual < 0 → over-forecast

## Service-risk metrics

```text
WDR = sum(deficit) / sum(demand)
DTR = timesteps_with_deficit / total_timesteps
PDR = max(deficit_t / demand_t)
```

For zero-demand timesteps, the instantaneous PDR ratio is 0.

Fixed WDR operating points:

```text
0.1%, 0.5%, 1.0%, 2.0%
```

No extrapolation outside a fold's Energy-WDR support.

## Energy accounting

The study reports **estimated server-side IT energy**, not measured total data-center electricity.

```text
E_IT =
sum_t [
  N_active(t) * P_server(utilization_t)
  + (M - N_active(t)) * P_off
] * delta_t
+ completed_boots * E_boot
```

Server-hours are reported as an intermediate metric.

## Policies

### Static
Fixed active server count.

### Target-tracking Reactive
```text
N_desired = ceil(D_t / (C * theta))
```

### Forecast-aware
Uses a forecast plus a validation-residual quantile safety offset.

### Oracle
Uses exact future demand as a non-deployable upper bound.

## Planned figures

- F1 — System pipeline
- F2 — Workload regimes
- F3 — Energy-WDR curves
- F4 — Prediction-benefit boundary
- F5 — Real forecasting models on synthetic map
- F6 — Forecast accuracy vs downstream value

## Planned tables

- T1 — Dataset and simulator assumptions
- T2 — Main matched-WDR results
- T3 — Robustness / sensitivity

## Setup

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install:

```bash
pip install -r requirements.txt
```

## Data policy

Do **not** commit raw Alibaba trace files to GitHub.

Use:

```text
data/raw/plan_a/
data/raw/plan_b/
```

Processed data should be reproducible from scripts and YAML configs.

## First milestone

Before adding more models:

```text
real demand
→ Static / Reactive / Oracle
→ delayed server activation
→ capacity
→ deficit
→ WDR / DTR / PDR
→ server-hours
→ energy
```

The exact linear-power accounting test (T4a) must pass before research experiments are trusted.

## Authors

**Minh-Quang Luu**  
**Viet-Duc Hoang**

University of Science and Technology of Hanoi (USTH), Hanoi, Vietnam.
