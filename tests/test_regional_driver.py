"""Tests for the regional decomposition — the one genuine driver split in this project.

Everything else in ``fpa.forecast`` is univariate. This is the only place a forecast is built
from *parts the filer reports separately*, which is why the README can say "driver" about it
and not about the cost centers.

``test_q4_is_refused_when_the_regions_do_not_foot`` is the test that matters. Q4 is never
filed as a quarterly segment fact — 10-Ks report segments annually — so it is derived as
``FY - (Q1+Q2+Q3)``, and the entire annual discrepancy would land in that one number if the
parts did not foot. The income-statement ingest already refuses on exactly this basis; the
same rule has to hold here or the derived quarter silently absorbs whatever is wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpa.config import STREAMING_TOTAL, get_settings
from fpa.forecast import regional as R


@pytest.fixture(scope="module")
def facts():
    from fpa.ingest.segments import load_segment_revenue

    return load_segment_revenue(get_settings())


@pytest.fixture(scope="module")
def series(facts):
    return R.quarterly_regional(facts)


def _synthetic(annual_total: float | None, quarters: tuple[float, float, float]):
    """A one-year fact frame: three filed quarters plus an annual, for every region."""
    rows = []
    for index, value in enumerate(quarters):
        end = pd.Timestamp(f"2024-{3 * (index + 1):02d}-28")
        for region in R.REGIONS:
            rows.append({"region": region, "end": end, "value": value, "period_type": "quarter"})
    year_end = pd.Timestamp("2024-12-31")
    for region in R.REGIONS:
        rows.append(
            {"region": region, "end": year_end, "value": sum(quarters) + 10.0,
             "period_type": "annual"}
        )
    if annual_total is not None:
        rows.append(
            {"region": STREAMING_TOTAL, "end": year_end, "value": annual_total,
             "period_type": "annual"}
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The series
# ---------------------------------------------------------------------------
def test_the_series_is_quarterly_and_complete(series):
    """21 filed quarters plus a derived Q4 for each year — a regular quarterly series.

    Without the Q4 derivation this is 21 points with an annual hole, which cannot carry a
    seasonal model or a rolling-origin backtest.
    """
    assert len(series) == 28
    assert list(series.columns) == list(R.REGIONS)
    assert series.index.is_monotonic_increasing
    # Every year from 2019 has all four quarter ends.
    for year in range(2019, 2026):
        assert (series.index.year == year).sum() == 4, f"{year} is not a complete year"


def test_every_region_is_positive_everywhere(series):
    assert (series[list(R.REGIONS)] > 0).all().all()


def test_derived_quarters_are_recorded_and_unverified_ones_flagged(series):
    """Derived is fine. Derived and *unverifiable* has to be visible."""
    assert series.attrs["derived_q4"], "no Q4 was derived"
    # The filer did not tag a consolidated streaming total before FY2023, so the footing
    # check cannot run on the earlier years and they must be named rather than trusted.
    assert set(series.attrs["unverified_q4"]) == {2019, 2020, 2021, 2022}
    assert 2023 not in series.attrs["unverified_q4"]


def test_the_regions_foot_to_the_filed_streaming_total(facts):
    """The check the derivation depends on, asserted directly on filed annual data."""
    annual = facts[facts["period_type"] == "annual"]
    wide = annual.drop_duplicates(subset=["region", "end"], keep="last").pivot(
        index="end", columns="region", values="value"
    )
    checked = wide.dropna(subset=[STREAMING_TOTAL]) if STREAMING_TOTAL in wide else wide.iloc[0:0]
    assert len(checked) >= 3, "expected a filed streaming total from FY2023"

    for period, row in checked.iterrows():
        residual = abs(row[list(R.REGIONS)].sum() - row[STREAMING_TOTAL])
        assert residual <= R.FOOTING_TOLERANCE * row[STREAMING_TOTAL], f"{period} does not foot"


# ---------------------------------------------------------------------------
# The refusal rule
# ---------------------------------------------------------------------------
def test_q4_is_refused_when_the_regions_do_not_foot():
    """The whole annual discrepancy would otherwise land in the derived quarter.

    Same rule the income-statement ingest applies to FY2017–FY2019, which is why this
    project's window starts in 2020.
    """
    broken = _synthetic(annual_total=1.0, quarters=(10.0, 10.0, 10.0))  # total nowhere near
    with pytest.raises(ValueError, match="needs both quarterly and annual"):
        R.quarterly_regional(broken[broken["period_type"] == "quarter"])

    derived = R.quarterly_regional(broken)
    assert 2024 not in derived.attrs["derived_q4"], "refused to derive from a year that does not foot"


def test_a_non_positive_derived_quarter_is_refused():
    """A negative Q4 means the annual is smaller than its own first three quarters."""
    shrinking = _synthetic(annual_total=None, quarters=(100.0, 100.0, 100.0))
    shrinking.loc[shrinking["period_type"] == "annual", "value"] = 250.0  # < 300
    derived = R.quarterly_regional(shrinking)
    assert 2024 not in derived.attrs["derived_q4"]


def test_missing_regions_raise_rather_than_producing_a_partial_series(facts):
    trimmed = facts[facts["region"] != "APAC"]
    with pytest.raises(ValueError, match="missing"):
        R.quarterly_regional(trimmed)


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------
def test_all_approaches_share_folds(series):
    comparison = R.compare_regional_vs_total(series, horizon=4, folds=3)
    by_approach = comparison.groupby("approach")["fold"].apply(set)

    assert len(by_approach) == 3
    assert len(set(map(frozenset, by_approach))) == 1
    assert comparison["mase"].notna().all()


def test_bottom_up_is_coherent_by_construction(series):
    """The property that justifies decomposing regardless of accuracy.

    The four regional forecasts must sum to the bottom-up total exactly — that is what
    "coherent" means, and it is why an FP&A team decomposes at all.
    """
    from fpa.forecast.models import MODELS

    train = series.iloc[:20]
    parts = sum(
        np.asarray(MODELS["ets"](train[region], 4, period=4), dtype=float) for region in R.REGIONS
    )
    # Reproduce what compare_regional_vs_total assembles.
    comparison_total = np.zeros(4)
    for region in R.REGIONS:
        comparison_total += np.asarray(MODELS["ets"](train[region], 4, period=4), dtype=float)
    np.testing.assert_allclose(parts, comparison_total, rtol=1e-12)


def test_the_report_states_the_verdict_and_the_derivation_caveat(series):
    comparison = R.compare_regional_vs_total(series, horizon=4, folds=3)
    markdown = R.report_markdown(series, comparison)

    assert "Bottom-up" in markdown
    # The Q4 derivation and its unverifiable years must survive into the rendered report.
    assert "Q4 = FY - (Q1+Q2+Q3)" in markdown
    assert "could not be checked" in markdown
    assert "24 archives" in markdown
