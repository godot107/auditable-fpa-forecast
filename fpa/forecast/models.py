"""Forecast models, including the naive benchmark everything is measured against.

The benchmark is not decoration. A forecast that cannot beat "same month last
year" is not worth deploying, and reporting accuracy without one is how a model
gets adopted on the strength of a number nobody contextualized. Every model here
is scored against :func:`seasonal_naive` in ``backtest.py``.

Forecasts are produced **bottom-up**: each leaf cost center is forecast
independently and the parents are formed by aggregation. That makes the hierarchy
coherent by construction — the cost-center forecasts always sum to the total the
CFO sees — at the cost of noisier leaf-level series than a top-down split would
give. Coherence is the property an FP&A team actually needs.
"""

from __future__ import annotations

import logging
import warnings
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def seasonal_naive(train: pd.Series, horizon: int, *, period: int = 12) -> np.ndarray:
    """Repeat the value from ``period`` months ago. The benchmark.

    Falls back to the last observation when the history is shorter than one
    seasonal cycle, which is the degenerate but correct behavior.
    """
    values = train.to_numpy(dtype=float)
    if len(values) < period:
        return np.full(horizon, values[-1])
    season = values[-period:]
    return np.array([season[i % period] for i in range(horizon)])


def drift_seasonal(train: pd.Series, horizon: int, *, period: int = 12) -> np.ndarray:
    """Seasonal naive plus the average year-over-year drift.

    A deliberately simple driver-flavored model: it keeps the seasonal shape and
    adds the trend the business is actually on. Cheap, transparent, and hard to
    beat on short monthly financial series — which is the point of including it.
    """
    values = train.to_numpy(dtype=float)
    base = seasonal_naive(train, horizon, period=period)
    if len(values) < 2 * period:
        return base

    recent = values[-period:].mean()
    prior = values[-2 * period : -period].mean()
    if prior == 0:
        return base
    growth = recent / prior - 1.0
    # Ramp the annual growth across the horizon rather than applying it flat.
    ramp = np.arange(1, horizon + 1) / period
    return base * (1.0 + growth * ramp)


# ``ets`` degrades to ``drift_seasonal`` on four conditions rather than raising, so one
# short leaf series cannot take down a whole run. That is the right behaviour and it used to
# be **silent**, which is not: a reported "ets" score was partly a drift_seasonal score with
# no way to know how much. Measured on the monthly backtest, 9 of 54 calls — **17%** — fell
# back because fold 1 trains on 24 months and ETS needs 2*period+1 = 25.
#
# This codebase's signature defect is a missing input producing a plausible number instead of
# an error. A silent degradation is the same shape, so it gets counted.
_FALLBACKS: Counter = Counter()


def reset_fallbacks() -> None:
    """Clear the fallback counters. Call before a run you want to attribute."""
    _FALLBACKS.clear()


def fallbacks() -> dict[str, int]:
    """How many times ``ets`` degraded, by reason, since the last reset."""
    return dict(_FALLBACKS)


def _fell_back(reason: str, train: pd.Series, horizon: int, period: int) -> np.ndarray:
    _FALLBACKS[reason] += 1
    logger.debug("ets fell back to drift_seasonal (%s), n=%d", reason, len(train))
    return drift_seasonal(train, horizon, period=period)


def ets(train: pd.Series, horizon: int, *, period: int = 12) -> np.ndarray:
    """Holt-Winters exponential smoothing (additive trend, multiplicative season).

    Needs two full seasonal cycles; below that it degrades to ``drift_seasonal`` rather than
    raising, so a short leaf series cannot take down the whole run. **Every degradation is
    counted** — see :func:`fallbacks` — because a score labelled ``ets`` that is partly
    ``drift_seasonal`` is a number that misreports what produced it.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    values = train.to_numpy(dtype=float)
    if len(values) < 2 * period + 1:
        return _fell_back("too_short", train, horizon, period)
    if np.any(values <= 0):
        # Multiplicative seasonality is undefined at or below zero.
        return _fell_back("non_positive", train, horizon, period)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                values,
                trend="add",
                seasonal="mul",
                seasonal_periods=period,
                initialization_method="estimated",
            ).fit()
        forecast = np.asarray(model.forecast(horizon), dtype=float)
        if not np.all(np.isfinite(forecast)):
            return _fell_back("non_finite", train, horizon, period)
        return forecast
    except Exception:
        # A model that fails to converge falls back rather than propagating.
        return _fell_back("exception", train, horizon, period)


MODELS = {
    "seasonal_naive": seasonal_naive,
    "drift_seasonal": drift_seasonal,
    "ets": ets,
}


def leaf_series(ledger: pd.DataFrame) -> pd.DataFrame:
    """Pivot the ledger to monthly series, one column per leaf cost center."""
    frame = ledger.pivot_table(
        index="period", columns=["function", "sub_center"], values="amount", aggfunc="sum"
    )
    return frame.sort_index()


def forecast_hierarchy(
    ledger: pd.DataFrame,
    horizon: int,
    *,
    model: str = "ets",
    period: int = 12,
) -> pd.DataFrame:
    """Forecast every leaf cost center and aggregate upward.

    Returns a tidy frame with ``period``, ``function``, ``sub_center``,
    ``forecast``. Parent totals are never forecast directly — they are the sum of
    their children, so the hierarchy reconciles by construction.
    """
    fn = MODELS[model]
    leaves = leaf_series(ledger)

    last = leaves.index.max()
    future = pd.date_range(last + pd.offsets.MonthEnd(1), periods=horizon, freq="ME")

    rows: list[dict] = []
    for (function, sub_center), series in leaves.items():
        values = fn(series.dropna(), horizon, period=period)
        for month, value in zip(future, values):
            rows.append(
                {
                    "period": month,
                    "function": function,
                    "sub_center": sub_center,
                    "forecast": float(value),
                    "model": model,
                }
            )
    return pd.DataFrame(rows)
