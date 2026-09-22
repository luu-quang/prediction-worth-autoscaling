import numpy as np
import pandas as pd
import pytest

DT_SECONDS = 300
STEPS_PER_DAY = 86400 // DT_SECONDS
N_DAYS = 7


def make_synthetic_demand_df(
    n_days: int = N_DAYS,
    dt_seconds: int = DT_SECONDS,
    seed: int = 42,
    start: str = "2024-01-01 00:00:00",
) -> pd.DataFrame:
    """Dev/test-only synthetic demand trace: diurnal pattern + noise, non-negative.

    Not a stand-in for the real Alibaba trace — used only so the pipeline can
    be built and tested before real data (data/raw/plan_b/) is available.
    """
    rng = np.random.default_rng(seed)
    steps_per_day = 86400 // dt_seconds
    n_steps = n_days * steps_per_day

    timestamps = pd.date_range(start=start, periods=n_steps, freq=f"{dt_seconds}s")
    t = np.arange(n_steps)
    diurnal = 50 + 30 * np.sin(2 * np.pi * t / steps_per_day - np.pi / 2)
    noise = rng.normal(0, 3, size=n_steps)
    demand = np.clip(diurnal + noise, 0, None)

    return pd.DataFrame({"timestamp": timestamps, "demand": demand})


@pytest.fixture
def synthetic_demand_df():
    return make_synthetic_demand_df()
