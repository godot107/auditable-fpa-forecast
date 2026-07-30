"""``ets`` degrades to ``drift_seasonal``, and the degradation must be counted.

Falling back rather than raising is correct: one short leaf series should not take down a
whole run. Doing it **silently** is not, and it did for the whole build. A score labelled
``ets`` that is partly ``drift_seasonal`` is a number that misreports what produced it —
the same shape as this codebase's other defects, where a missing input yields a plausible
value instead of an error.

Measured on the monthly backtest: **11 of 66 calls, 16.7%**, all ``too_short``, because the
first rolling-origin fold trains on 24 months and ETS needs ``2 x period + 1 = 25``. The
filed-quarterly backtest — the one the report leads with — degrades **0 times**, which is why
`0.936` is a real ETS score.

``test_the_two_passes_are_gone`` guards the fix that made the count trustworthy at all:
``run_backtest`` used to run every model twice, so every degradation was counted twice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpa.forecast import models as M
from fpa.forecast.backtest import run_backtest


def _series(n: int, *, freq: str = "ME", start: str = "2020-01-31", scale: float = 100.0):
    index = pd.date_range(start, periods=n, freq=freq)
    trend = np.linspace(scale, scale * 1.5, n)
    season = 1 + 0.1 * np.sin(np.arange(n) * 2 * np.pi / 12)
    return pd.Series(trend * season, index=index)


# ---------------------------------------------------------------------------
# Each fallback reason is reachable and attributed
# ---------------------------------------------------------------------------
def test_too_short_history_is_counted_with_its_reason():
    M.reset_fallbacks()
    M.ets(_series(24), 12, period=12)  # needs 25
    assert M.fallbacks() == {"too_short": 1}


def test_a_long_enough_series_does_not_fall_back():
    M.reset_fallbacks()
    M.ets(_series(40), 12, period=12)
    assert M.fallbacks() == {}, "a healthy fit must not be recorded as a degradation"


def test_non_positive_values_are_counted_separately():
    """Multiplicative seasonality is undefined at or below zero."""
    M.reset_fallbacks()
    series = _series(40)
    series.iloc[5] = 0.0
    M.ets(series, 12, period=12)
    assert M.fallbacks() == {"non_positive": 1}


def test_the_fallback_returns_the_drift_seasonal_path_exactly():
    """Degrading has to mean *this other model*, not an approximation of it."""
    series = _series(24)
    np.testing.assert_array_equal(
        M.ets(series, 12, period=12), M.drift_seasonal(series, 12, period=12)
    )


def test_reset_clears_the_counter():
    M.reset_fallbacks()
    M.ets(_series(24), 12, period=12)
    assert M.fallbacks()
    M.reset_fallbacks()
    assert M.fallbacks() == {}


# ---------------------------------------------------------------------------
# The backtest reports the rate
# ---------------------------------------------------------------------------
@pytest.fixture
def frame():
    return pd.DataFrame({f"series_{i}": _series(78, scale=100.0 * (i + 1)) for i in range(3)})


def test_the_backtest_carries_the_rate(frame):
    result = run_backtest(frame, horizon=12, folds=6)

    assert result.ets_calls == 3 * 6, "one ets call per series per fold"
    assert sum(result.fallbacks.values()) > 0, "fold 1 trains on 24 months; ETS needs 25"
    assert 0 < result.fallback_rate < 1


def test_the_two_passes_are_gone(frame):
    """The fix that made the count meaningful.

    ``run_backtest`` used to compute every forecast twice — once for the detail rows and once
    for the summary — so the counter double-reported and the two passes had to agree. Calls
    must now equal series x folds exactly, with no factor of two.
    """
    M.reset_fallbacks()
    result = run_backtest(frame, horizon=12, folds=6)
    observed = sum(M.fallbacks().values())

    assert observed == sum(result.fallbacks.values())
    assert observed == len(frame.columns), "one degradation per series on the one short fold"


def test_a_clean_backtest_reports_a_zero_rate():
    """The filed-quarterly path, which is why 0.936 is trustworthy."""
    quarterly = pd.DataFrame({"a": _series(26, freq="QE"), "b": _series(26, freq="QE", scale=50.0)})
    result = run_backtest(quarterly, horizon=4, folds=4, period=4)

    assert result.fallbacks == {}
    assert result.fallback_rate == 0.0


def test_the_validation_report_publishes_the_rate(frame):
    """A rate held only in the object protects nobody."""
    from fpa.forecast.backtest import honest_validation_report

    monthly = run_backtest(frame, horizon=12, folds=6)
    quarterly = pd.DataFrame({"a": _series(26, freq="QE")})
    filed = run_backtest(quarterly, horizon=4, folds=4, period=4)

    markdown = honest_validation_report(
        monthly, filed, horizon_months=12, horizon_quarters=4
    )
    assert "does not always run" in markdown
    assert "too_short" in markdown
    assert "ets calls" in markdown


# ---------------------------------------------------------------------------
# The ninth defect: the headline frame must not be silently partial
# ---------------------------------------------------------------------------
def test_the_filed_backtest_frame_requires_every_series():
    """Found by the standing sweep, not by a failure.

    ``backtest_frames`` filtered the filed-series list with
    ``[c for c in wanted if c in frame.columns]``, so a renamed tag or a broken derivation
    would have scored five series instead of six — changing the headline MASE with no signal
    that it had changed. Nothing was missing on the day, which is how this class survives.
    """
    from fpa.config import get_settings
    from fpa.pipeline import backtest_frames, build_ledger

    context = build_ledger(get_settings())
    context.quarterly = context.quarterly.drop(columns=["marketing"])

    with pytest.raises(ValueError, match="missing"):
        backtest_frames(context)
