# OPEN_DECISIONS_FINAL.md — SF2026 Project

**Project:** *When Is Prediction Worth It? A Trace-Driven Study of Forecast-Aware Autoscaling Under Provisioning Delay, Forecast Uncertainty, and Server Energy Proportionality*  
**Team:** Minh-Quang Luu, Viet-Duc Hoang — USTH  
**Status:** **CLOSED / FROZEN FOR IMPLEMENTATION.** No V4/V5. The next evidence must be T1–T9 + GO/NO-GO 1 output.

---

# 0. Global conventions — locked

## Residual sign convention
Use **one convention everywhere**:

\[
r_{t,H}=D_{t+H}-f_{t,H}
\]

where:
- \(D_{t+H}\) = actual future demand,
- \(f_{t,H}\) = forecast made at time \(t\) for horizon \(H\),
- \(r_{t,H}>0\) means **under-forecasting**,
- \(r_{t,H}<0\) means **over-forecasting**.

For residual-quantile protection:

\[
q^{safe}_{t,H}(\tau)=f_{t,H}+Q_\tau(r_{\cdot,H})
\]

Residual distributions are estimated **separately for each horizon \(H\) and each validation fold**.

F4/F5 captions must explicitly state:

> Positive bias means under-forecasting because residuals are defined as actual minus forecast.

---

# A. Synthetic forecast-error generator

**Deadline:** before E4 / prediction-benefit boundary.

## A1. Generator — centered AR(1)

Use:

\[
f_{t,H}=D_{t+H-\lambda}-r_t
\]

with:

\[
r_t=\mu+\phi(r_{t-1}-\mu)+\varepsilon_t
\]

where:

- \(m\) = target **total** forecast RMSE,
- \(\mu\) = stationary mean residual / bias,
- \(\phi\) = residual autocorrelation,
- \(\lambda\) = ramp-lag in timesteps,
- \(\varepsilon_t\) = zero-mean innovation noise.

Define:

\[
\mu=c_b\,m
\]

where \(c_b\) is the bias fraction.

Interpretation:
- \(\mu>0\) means systematic **under-forecasting**,
- \(\mu<0\) means systematic **over-forecasting**,
- \(\phi>0\) produces persistent errors,
- \(\lambda>0\) makes the forecast react late to ramps.

This centered AR(1) parameterization guarantees:

\[
E[r_t]=\mu
\]

for every \(\phi\), so changing autocorrelation does not silently change the intended bias.

### AR burn-in
Before measuring RMSE or exporting the synthetic forecast, simulate and discard an AR burn-in period.

Default:
- `burn_in_steps = max(50, 10 / (1 - phi))`, rounded up.

The burn-in is not part of the evaluation trace.

---

## A2. Total-error magnitude normalization

Total forecast error is:

\[
e^{total}_{t,H}
=
D_{t+H}-f_{t,H}
=
\left[D_{t+H}-D_{t+H-\lambda}\right]+r_t
\]

For each structural configuration \((c_b,\phi,\lambda)\):

1. Generate the raw forecast.
2. Measure **total** forecast RMSE.
3. Adjust the innovation scale \(\sigma_\varepsilon\) so that:
   \[
   RMSE(e^{total}_{t,H})=m
   \]
4. Recompute the forecast and verify the achieved RMSE.

Do **not** normalize only the random residual component.

### Feasibility rule
Some combinations cannot achieve a requested target \(m\).

Examples:
- lag component alone already exceeds \(m\),
- \(|\mu|\) alone exceeds \(m\),
- lag + bias leaves no non-negative innovation variance compatible with target RMSE.

If the target cannot be reached:

- set `infeasible = true`,
- do not silently return the wrong RMSE,
- leave the corresponding F4/F4-panel cell blank or hatched.

---

## A3. Bias grid

Bias is expressed as a fraction of the target error budget:

\[
c_b\in\{-1,-0.5,0,0.5,1\}
\]

with:

\[
\mu=c_bm
\]

This keeps bias interpretable and compatible with the fixed-magnitude design.

---

## A4. Lag parameterization — relative to horizon

Do **not** use a fixed absolute lag grid across all horizons.

Define the integer lag with **round-half-up**, not Python/NumPy `round()`:

\[
\lambda=
\left\lfloor c_\lambda H + 0.5 \right\rfloor
\]

Equivalent implementation:

```python
lambda_steps = int(math.floor(c_lambda * H + 0.5))
```

with:

\[
c_\lambda\in\{0,0.25,0.5,1.0\}
\]

Interpretation:
- \(c_\lambda=0\): forecast is anchored at \(D_{t+H}\) before residual noise → Oracle-like lag structure,
- \(c_\lambda=1\): forecast is anchored at \(D_t\) → Persistence-like lag structure,
- intermediate values lie between Oracle and Persistence.

This makes the lag axis comparable across different provisioning delays.

### Lag-resolution note
Under Plan B:
- \(H=1\) can only realize \(\lambda\in\{0,1\}\), so several \(c_\lambda\) settings collapse to the same integer lag.
- \(H=3\) and \(H=6\) provide more useful multi-level lag resolution.
- If Plan A arrives in time, prefer Plan A for the dedicated F4d lag-sensitivity panel because \(H\in\{5,15,30\}\) steps gives a much cleaner lag axis.

Collapsed \(c_\lambda\) settings at a given \(H\) must be deduplicated before plotting and must not be interpreted as evidence that lag has no effect.

---

## A5. Structure grid

Initial structure values:

- `m`: low / medium / high, calibrated from empirical validation RMSE levels,
- `bias_fraction c_b`: `{-1, -0.5, 0, 0.5, 1}`,
- `phi`: `{0.0, 0.5, 0.8}`,
- `lag_fraction c_lambda`: `{0, 0.25, 0.5, 1.0}`.

Do not force the full factorial into one figure.

---

## A6. Synthetic forecast output contract

Synthetic forecasts use the same Track-A schema:

- `timestamp`
- `actual_demand`
- `forecast_point`
- `horizon_steps`
- `fold_id`
- optional quantile columns
- `synthetic_config_id`
- `infeasible`

No special simulator code path is allowed for synthetic forecasts.

Synthetic forecasts are generated **offline** and then passed through the same `forecast_df` interface as deployable models.

---

# B. Mapping real models onto the synthetic map

For each real model and each \((fold,H)\), define:

\[
r_{t,H}=D_{t+H}-f_{t,H}
\]

Estimate:

### Magnitude
\[
m=RMSE(r_{t,H})
\]

### Bias
\[
\mu=mean(r_{t,H})
\]

and:

\[
c_b=\frac{\mu}{m}
\]

when \(m>0\).

### Residual autocorrelation
\[
\phi=Corr(r_{t,H},r_{t-1,H})
\]

### Relative ramp lag
First estimate the best non-negative lag:

\[
\lambda^*
=
\arg\max_{\ell\in\{0,\ldots,H\}}
Corr(\Delta f_{t,H},\Delta D_{t+H-\ell})
\]

Then report:

\[
c_\lambda=\frac{\lambda^*}{H}
\]

Estimate these coordinates on validation folds for characterization/tuning.

**F5 must show one point per model per fold**, not one single collapsed point per model. This directly shows cross-fold variation.

---

# C. Power-curve family

**Deadline:** before hardware-sensitivity experiment.

Keep server capacity \(C\) fixed across the energy-proportionality sweep.

Use:

\[
P(u)=P_{peak}[k+(1-k)u]
\]

where:

\[
k=\frac{P_{idle}}{P_{peak}}
\]

Initial grid:

\[
k\in\{0.1,0.3,0.5,0.7\}
\]

Interpretation:
- low \(k\): highly energy-proportional,
- high \(k\): high idle-power fraction.

For empirical SPECpower profiles:
- use published utilization/power points,
- interpolate piecewise linearly,
- use them as realism anchors / sensitivity profiles,
- do not change \(C\) when the scientific axis is energy proportionality.

---

# D. Matched-WDR statistical protocol

## D1. Pre-registered WDR targets

Fix the WDR operating points **before looking at test results**:

\[
\epsilon\in\{0.1\%,0.5\%,1.0\%,2.0\%\}
\]

Equivalent decimal values:

```text
[0.001, 0.005, 0.010, 0.020]
```

Do not change this set after test evaluation begins.

Use all four in T2 / robustness reporting rather than selecting whichever produces the nicest result.

F3 shows the full Energy–WDR curves and marks these operating points where supported.

---

## D2. Per-fold interpolation first

Correct order:

1. Sweep one policy's risk knob inside one test fold.
2. Build that fold's Energy–WDR curve.
3. For each \(\epsilon\), interpolate energy at that WDR **inside the fold**.
4. Compute energy saving / policy difference for that fold.
5. Repeat for all folds.
6. Summarize fold-level results.

Never pool all folds first and interpolate once.

---

## D3. Unsupported epsilon rule — no extrapolation

If a target \(\epsilon\) lies outside the WDR support of a policy curve in a fold:

- do **not** extrapolate,
- mark that policy/fold/\(\epsilon\) as unsupported,
- exclude that fold from the comparison at that \(\epsilon\).

Report the effective sample size separately for every \(\epsilon\):

\[
n_\epsilon = \text{number of valid test folds supporting that target WDR}
\]

T2 must include or footnote \(n_\epsilon\).

---

## D4. Cross-day reporting

Primary uncertainty reporting:
- median,
- IQR,
- min/max,
- \(n_\epsilon\).

Blocked bootstrap may be supplementary.

Do not present a narrow-looking CI without also showing the number of independent test days.

---

# E. Walk-forward fold definition

## E1. Main-analysis folds — pre-registered

Assuming **7 complete days**, the main paper uses test days:

\[
k\in\{4,5,6,7\}
\]

For test day \(k\):

- **Train:** days \(1,\ldots,k-2\)
- **Validation:** day \(k-1\)
- **Test:** day \(k\)

Therefore the maximum number of independent main-analysis test folds is:

\[
n_{\max}=4
\]

before any fold is removed because a target WDR \(\epsilon\) is unsupported.

### Why test day 3 is excluded
The day-3 fold would have only:
- train = day 1,
- validation = day 2,
- test = day 3.

At Plan-B resolution this gives only about 288 five-minute training points, which is too weak for a fair main comparison involving LightGBM.

**Pre-registered decision:** exclude the day-3 fold from all primary paper analyses so every compared method uses the same fold set.

Day 3 may be used only for debugging/pilot checks and is not included in T2/F3/F5/F6 or reported main statistics.

## E2. Leakage-safe tuning

Thus all tuning for fold \(k\) uses only information strictly before test day \(k\).

For each fold:
- model fitting → train only,
- residual quantile / safety offset → validation only,
- reactive knob tuning → validation only,
- model selection / early stopping → validation only,
- final system metrics → test only.

No test-day information may be used to set policy/model hyperparameters.

---

# F. Initialization and warm-up

For every policy:

\[
N_0=
\left\lceil
\frac{D_0}{C}
\right\rceil
\]

Use one **fixed 2-hour warm-up** for:
- every policy,
- every delay scenario,
- every model,
- every fold.

Exclude warm-up from:
- energy,
- WDR,
- DTR,
- PDR,
- server-hours,
- reported scaling-operation statistics.

---

# G. Boot-event accounting

A boot is counted when startup **completes** and the server becomes usable.

If \(B_t\) servers complete boot at timestep \(t\):

\[
E^{boot}_t=B_tE_{boot}
\]

Charge \(E_{boot}\) at that timestep.

MVP may use \(E_{boot}=0\), but the event definition is fixed.

---

# H. Simulator timing

## Plan B
- \(\Delta t=300\) s,
- delay = `{1, 3, 6}` timesteps,
- equivalent to `{5, 15, 30}` minutes,
- \(H=\) provisioning delay.

## Plan A
If raw source supports true 60-second aggregation:
- \(\Delta t=60\) s,
- delay = `{5, 15, 30}` timesteps,
- equivalent to `{5, 15, 30}` minutes,
- \(H=\) provisioning delay.

Do not upsample beyond source granularity.

## Scale-up
Has provisioning delay.

## Scale-down
Immediate once cooldown and min-on-time conditions are satisfied.

---

# I. Risk knobs — exactly one per policy

| Policy | Risk knob |
|---|---|
| Static | active server count \(N\) |
| Target-tracking reactive | target utilization \(\theta\) |
| Forecast-aware | residual quantile \(\tau\) |
| Oracle | under-provision factor \(\alpha\) |

All other parameters stay fixed during one matched-WDR sweep.

Expect staircase behavior because server count is integer-valued.

Before interpolation:
- sort the sweep deterministically,
- deduplicate identical WDR values by keeping the **lowest-energy** point,
- remove Pareto-dominated points,
- construct the **monotone Energy–WDR envelope**,
- interpolate only on that envelope, never on raw sweep points,
- record how many raw points were removed,
- log any unexpected monotonicity violation.

### Envelope definition
After sorting by WDR from low to high, the feasible best-energy function is:

\[
E^*(\epsilon)=
\min_{j:\,WDR_j\le\epsilon} E_j
\]

Hence energy on the envelope must be non-increasing as allowed WDR increases.

Operationally:
1. sort unique points by WDR ascending,
2. keep the cumulative minimum energy,
3. retain only frontier breakpoints where the cumulative minimum improves,
4. interpolate between those frontier points for supported \(\epsilon\).

This handles the staircase behavior caused by `ceil()` without allowing a dominated raw point to distort matched-WDR interpolation.

---

# J. Target-tracking reactive baseline

Use:

\[
N_t=
\left\lceil
\frac{D_t}{C\theta}
\right\rceil
\]

where \(\theta\) is target utilization.

This is a target-tracking utilization controller.

Verify and cite the exact relationship to Kubernetes HPA before the paper claims practical equivalence.

Give the reactive policy the same validation-tuning budget as forecast-aware policies.

---

# K. Oracle definition

Oracle sees exact demand at horizon \(H\).

Use:

\[
f^{oracle}_{t,H}=\alpha D_{t+H}
\]

with:

\[
0<\alpha\le1
\]

Sweep \(\alpha\) to create an Oracle Energy–WDR curve.

---

# L. Complete energy equation

\[
E_{IT}
=
\sum_t
\left[
N_tP_{server}(u_t)
+
(M-N_t)P_{off}
\right]\Delta t
+
\sum_tB_tE_{boot}
\]

where:
- \(M\) = physical fleet size,
- \(N_t\) = active usable servers,
- \(B_t\) = boot completions at timestep \(t\).

If MVP uses \(P_{off}=0\), state that explicitly.

---

# M. Exact energy sanity tests

## T4a — exact linear-power accounting test

For:

\[
P(u)=P_{idle}+(P_{peak}-P_{idle})u
\]

with:
- fixed demand,
- no clipping,
- deterministic active-server schedules,

the equality is exact:

\[
\Delta E=
(P_{idle}-P_{off})
\Delta(\text{server-hours})
\]

Require floating-point-level agreement.

Suggested numerical tolerance:
- `rtol = 1e-9`
- `atol = 1e-9` after unit normalization.

If T4a fails, stop and debug accounting.

---

## T4b — empirical power-curve magnitude check

Repeat with a nonlinear / piecewise SPECpower profile.

No exact equality is expected.

Use only as a qualitative or few-percent magnitude sanity check.

---

## T9 — Persistence equivalence test

Construct synthetic forecasts with:

- \(c_\lambda=1\),
- \(c_b=0\),
- \(\phi=0\),
- innovation scale \(\sigma_\varepsilon\rightarrow0\).

Then:

\[
\lambda=H
\]

and:

\[
f_{t,H}\approx D_t
\]

Therefore this synthetic generator configuration must match the ordinary **Persistence forecast** path through the simulator.

Require:
- identical forecast arrays within numerical tolerance,
- identical active-server traces,
- identical WDR,
- identical energy.

If T9 fails, either the synthetic generator or Persistence implementation is inconsistent.

---

# N. Service-risk metrics

## WDR — primary matched-risk metric

\[
WDR=
\frac{\sum_t deficit_t}{\sum_tD_t}
\]

where:

\[
deficit_t=\max(0,D_t-Capacity_t)
\]

## DTR — Deficit Timestep Ratio

\[
DTR=
\frac{\#\{t:deficit_t>0\}}
{\#\{t\}}
\]

## PDR — Peak Deficit Ratio

\[
PDR=
\max_t
\left(
\frac{deficit_t}{D_t}
\right)
\]

For \(D_t=0\), define instantaneous deficit ratio as 0.

DTR and PDR are diagnostics, not matching variables.

---

# O. Plan-B demand units

If Plan B provides whole-data-center average utilization:

\[
D_t=
MC\frac{util_t}{100}
\]

On Day 1:
1. verify source utilization definition,
2. compute \(D_t\),
3. inspect how often \(D_t\approx MC\),
4. log saturation/censoring.

If fleet cap is frequently reached, low-WDR curve support may be truncated.

---

# P. Leakage policy — T5 must encode this explicitly

Deployable methods must never access future demand.

### Allowed future access
Only:
1. **Oracle policy** — explicitly nondeployable upper bound.
2. **Offline synthetic forecast generator** — research instrument used to construct controlled `forecast_df`.

### Not allowed
Persistence, EWMA, Linear/AR, LightGBM, target-tracking Reactive, Static, residual-quantile calibration and deployable policy logic must not access test-future demand.

Important implementation rule:

> The synthetic generator may use future demand offline to create controlled forecasts, but once `forecast_df` is produced, it enters the same simulator interface as all other forecast sources. The simulator itself receives no privileged future information.

State the same distinction in the paper so synthetic forecasting is not mistaken for data leakage.

---

# Q. Locked F4 design

## Main F4
- X-axis = error magnitude \(m\),
- Y-axis = provisioning delay \(H\),
- color = energy saving over reactive at matched WDR.

## Structure panels
Vary one factor around a fixed reference configuration:

- F4a: baseline structure,
- F4b: bias fraction \(c_b\),
- F4c: autocorrelation \(\phi\),
- F4d: relative lag \(c_\lambda\).

The full factorial may support T3 but is not the main visualization.

---

# R. F5 design — per-fold model points

F5 overlays real forecasting models on the synthetic map.

For each model and fold, plot one point characterized by:

\[
(m,c_b,\phi,c_\lambda)
\]

Do not collapse all folds into one point.

This shows day-to-day uncertainty directly without pretending the coordinate estimates are exact.

---

# S. Related-work task — Day 0–3

Search in parallel with coding:

1. predict-then-optimize / decision-focused learning,
2. energy proportionality,
3. autoscaling survey / predictive autoscaling,
4. newsvendor / quantile decision logic,
5. Kubernetes HPA / utilization-target autoscaling.

Goal: detect novelty collision by Day 2–3.

---

# T. Reproducibility

Before large sweeps:

- fixed random seed in YAML,
- preprocessing cached as Parquet,
- one YAML config per experiment family,
- config hash / experiment ID in all result rows.

Each result row stores:
- `experiment_id`
- `fold_id`
- `data_subset_id`
- `model_id`
- `policy_id`
- `risk_knob_name`
- `risk_knob_value`
- `epsilon_wdr`
- `seed`
- `dt_seconds`
- `delay_steps`
- `horizon_steps`
- `power_curve_id`
- `warmup_steps`
- `support_valid`
- `infeasible`

---

# U. Locked figures and tables

## Figures

### F1 — System pipeline
Trace → forecast → delayed provisioning → power model → energy/risk evaluation.

### F2 — Workload regimes
Smooth vs bursty segments / aggregation scales.

### F3 — Energy–WDR curves
Static / Target-tracking Reactive / Persistence / LightGBM / Oracle.

Show the fixed \(\epsilon\) operating points where supported.

### F4 — Prediction-benefit boundary
Main: magnitude × delay.  
Panels: bias / autocorrelation / relative lag.

Caption explicitly states positive bias = under-forecasting.

### F5 — Real models on synthetic map
One point per model **per fold**.

### F6 — Forecast accuracy vs downstream value
MAE/RMSE/pinball loss versus matched-WDR energy or energy saving.

---

## Tables

### T1 — Dataset and simulator assumptions
Source, granularity, \(\Delta t\), folds, warm-up, capacity, delays, cooldown, power model.

For the 7-complete-day design, state explicitly:

- primary test days = `{4, 5, 6, 7}`,
- maximum main-analysis fold count = \(n_{max}=4\),
- actual \(n_\epsilon\le4\) is reported separately for each target WDR after support filtering.

### T2 — Main matched-WDR results
For each \(\epsilon\):
- policy,
- horizon,
- energy,
- WDR,
- server-hours,
- DTR,
- PDR,
- \(n_\epsilon\),
- cross-day spread.

### T3 — Robustness / sensitivity
Across:
- days,
- workload regimes,
- power proportionality,
- structured forecast-error conditions.

---

# V. Immediate execution plan

## Track B — start now
Implement:
1. server state transitions,
2. Static,
3. Target-tracking Reactive,
4. Oracle,
5. startup queue,
6. cooldown / min-on-time,
7. WDR / DTR / PDR,
8. server-hours,
9. full energy accounting,
10. T1–T8,
11. T4a exact test.

Track B is **not blocked** by the synthetic generator fixes.

## Track A — start now
Implement:
1. Plan-B loader,
2. verified \(D_t\) units,
3. missingness / saturation checks,
4. walk-forward fold generator,
5. Persistence,
6. common `forecast_df` schema.

Before Day 7:
- implement centered-AR synthetic generator,
- implement total-RMSE calibration,
- implement relative lag \(c_\lambda\),
- implement T9,
- freeze fixed WDR targets,
- freeze unsupported-\(\epsilon\) handling.

When Plan-A raw data arrives:
- inspect `cpu_num` distribution immediately,
- only aggregate to 60 s if source granularity genuinely supports it.

---

# GO/NO-GO 1 — within 2–3 days

The following chain must run on one real trace:

\[
demand
\rightarrow
policy
\rightarrow
delayed\ active\ servers
\rightarrow
served\ demand
\rightarrow
deficit
\rightarrow
server\!-\!hours
\rightarrow
energy
\]

for:
- Static,
- Target-tracking Reactive,
- Oracle.

And:

**T4a must pass at exact numerical tolerance.**

If it fails, debug code/spec implementation.

Do not create another architecture document.
Do not add LightGBM, native quantile learning, volatility regimes or Net Zero extensions until GO/NO-GO 1 passes.
