import numpy as np


def workload_deficit_ratio(demand, deficit):
    demand = np.asarray(demand, dtype=float)
    deficit = np.asarray(deficit, dtype=float)
    denom = demand.sum()
    return 0.0 if denom == 0 else float(deficit.sum() / denom)


def deficit_timestep_ratio(deficit):
    deficit = np.asarray(deficit, dtype=float)
    return float(np.mean(deficit > 0))


def peak_deficit_ratio(demand, deficit):
    demand = np.asarray(demand, dtype=float)
    deficit = np.asarray(deficit, dtype=float)
    ratios = np.divide(deficit, demand, out=np.zeros_like(deficit), where=demand > 0)
    return float(ratios.max(initial=0.0))
