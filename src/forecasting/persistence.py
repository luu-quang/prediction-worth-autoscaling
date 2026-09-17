def persistence_forecast(demand_series):
    """Persistence baseline: forecast(t + H) = demand(t)."""
    return demand_series.copy()
