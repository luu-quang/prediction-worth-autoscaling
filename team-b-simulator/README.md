# Team B — Simulator / Server / Energy

**Owner:** Minh-Quang Luu  
**Branch:** `track-b-simulator`

## Goal
Build the autoscaling simulator, server-state logic, service-risk metrics, energy accounting, and matched-WDR evaluation.

## Step 1 — Simulator state
Work mainly in:
- `src/simulator/`
- `src/policies/`
- `src/energy/`
- `src/metrics/`
- `src/evaluation/`
- `tests/`

Core states:
- OFF
- BOOTING
- ACTIVE

Locked behavior:
- scale-up has provisioning delay
- boot counted when startup finishes
- E_boot charged when startup finishes
- scale-down is immediate after cooldown/min-on constraints
- fixed 2-hour warm-up

## Step 2 — Stable simulator interface
Use:
```python
simulate(demand_df, forecast_df, policy_cfg, sim_cfg) -> trace_df
```

Per-step output:
- timestamp
- demand
- desired_servers
- active_servers
- booting_servers
- capacity
- served_demand
- deficit
- utilization
- power_w
- energy_wh
- boots

## Step 3 — Static policy
Implement `src/policies/static.py`.

Risk knob:
`N = fixed server count`

## Step 4 — Reactive policy
Implement `src/policies/reactive.py`.

Formula:
`N_desired = ceil(D_t / (C * theta))`

Risk knob:
`theta`

## Step 5 — Oracle
Implement `src/policies/oracle.py`.

Use:
`forecast_oracle = alpha * D_(t+H)`

Risk knob:
`alpha`

## Step 6 — Service-risk metrics
Implement:
- WDR
- DTR
- PDR

Definitions:
`WDR = sum(deficit)/sum(demand)`

`DTR = deficit timesteps / total timesteps`

`PDR = max(deficit_t / demand_t)`

## Step 7 — Server-hours
`server_hours = sum(active_servers_t * delta_t_hours)`

## Step 8 — Energy accounting
Implement `src/energy/accounting.py`.

Use:
`E_IT = sum_t[N_t*P_server(u_t) + (M-N_t)*P_off]*dt + boots*E_boot`

Paper wording:
**estimated server-side IT energy**

## Step 9 — T4a exact sanity test
With a linear power curve and no clipping:

`Delta E = (P_idle - P_off) * Delta(server-hours)`

Require approximately:
- `rtol=1e-9`
- `atol=1e-9`

If T4a fails, stop and debug.

## Step 10 — First Energy–WDR curves
Run only:
- Static
- Reactive
- Oracle

Before adding forecast-aware policies.

## Step 11 — Monotone envelope
Before interpolation:
1. sort by WDR
2. deduplicate identical WDR by lowest energy
3. remove dominated points
4. cumulative-min energy frontier
5. keep frontier breakpoints
6. interpolate only on the envelope

## Step 12 — Fixed WDR targets
Use:
- 0.1%
- 0.5%
- 1.0%
- 2.0%

No extrapolation.

## Step 13 — Timing
Plan B:
- dt=300s
- delays=1,3,6 steps
- H=delay

Plan A, if source supports 60s:
- dt=60s
- delays=5,15,30 steps

## GO/NO-GO 1
Within 2–3 days, real trace must run:

`demand -> policy -> delayed activation -> active servers -> capacity -> deficit -> WDR/DTR/PDR -> server-hours -> energy`

for:
- Static
- Reactive
- Oracle

And:
- [ ] T4a PASS

## First-curve checks
1. Oracle should beat Reactive at supported WDR.
2. Static should usually be worse.
3. Envelope energy must decrease as allowed WDR increases.
4. Server-hours should broadly explain energy ordering.
5. Check whether 0.1% WDR is reachable.

## Definition of Done
- [ ] simulator works
- [ ] provisioning delay works
- [ ] Static works
- [ ] Reactive works
- [ ] Oracle works
- [ ] WDR/DTR/PDR work
- [ ] server-hours work
- [ ] energy accounting works
- [ ] T4a passes
- [ ] first Energy–WDR curves exist

## Git
```bash
git checkout track-b-simulator
git pull origin track-b-simulator
git add .
git commit -m "Track B: <what changed>"
git push origin track-b-simulator
```

Do not push unfinished work directly to `main`.
