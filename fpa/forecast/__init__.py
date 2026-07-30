"""Forecast layer — bottom-up hierarchical forecasting with an honest backtest.

Not driver-based, and the distinction is worth keeping straight. No exogenous variable
enters any model here: each cost line is extrapolated from its own history and parents
are formed by aggregation. What is driver-*shaped* is structural — forecast the leaves
and let the parent fall out, never forecast a margin directly — which the backtest
supports rather than assumes, since operating income is consistently the hardest series.
"""

from fpa.forecast.models import (
    MODELS,
    drift_seasonal,
    ets,
    forecast_hierarchy,
    seasonal_naive,
)
from fpa.forecast.backtest import (
    BacktestResult,
    mase,
    rolling_origin_splits,
    run_backtest,
    wmape,
)

__all__ = [
    "MODELS",
    "BacktestResult",
    "drift_seasonal",
    "ets",
    "forecast_hierarchy",
    "seasonal_naive",
    "mase",
    "rolling_origin_splits",
    "run_backtest",
    "wmape",
]
