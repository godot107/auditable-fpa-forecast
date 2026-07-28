"""Tests for the regional-segment ingest.

Offline: they run against the pinned Parquet vintage and the pure parsing
functions, never against SEC. The archives are ~85 MB each and a test suite has no
business downloading them.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpa.config import REGION_LABELS, STREAMING_TOTAL
from fpa.controls import LedgerContext, run_controls
from fpa.ingest.segments import _parse_segments


def _outcome(report, name: str):
    return next(r for r in report.results if r.name == name)


def _context(result, **overrides) -> LedgerContext:
    base = dict(
        quarterly=result.quarterly,
        ledger=result.ledger,
        revenue=result.revenue,
        drivers=result.drivers,
        budget=result.budget,
        facts=result.facts,
        balance_sheet=result.balance_sheet,
        cash_flow=result.cash_flow,
        segments=result.segments,
    )
    return LedgerContext(**{**base, **overrides})


def test_segment_string_parses_into_axes():
    assert _parse_segments("Geographical=EMEA;ProductOrService=Streaming;") == {
        "Geographical": "EMEA",
        "ProductOrService": "Streaming",
    }
    assert _parse_segments("") == {}


def test_only_the_four_operating_regions_are_ingested(result):
    """``Geographical=US`` overlaps UCAN and must never reach the frame.

    It is worth $18.5B in FY2025. Including it inflates revenue by roughly 40% — a
    figure obvious in a total and invisible in a per-region chart.
    """
    if result.segments is None:
        pytest.skip("no segment vintage in this checkout")

    columns = set(result.segments.columns) - {"total", STREAMING_TOTAL}
    assert columns == set(REGION_LABELS.values())
    assert "US" not in result.segments.columns


def test_regions_sum_to_the_filed_streaming_line(result):
    """Two SEC products must agree: the Data Sets' regions and the filed streaming total."""
    if result.segments is None or STREAMING_TOTAL not in result.segments.columns:
        pytest.skip("no streaming total in this vintage")

    both = result.segments[["total", STREAMING_TOTAL]].dropna()
    assert len(both) >= 3
    assert (both["total"] - both[STREAMING_TOTAL]).abs().max() <= 1.0


def test_legacy_dvd_revenue_declines_to_zero(result):
    """The residual against consolidated revenue is a real business, not an error.

    Netflix's DVD-by-mail service ran until September 2023. The gap between
    consolidated and streaming revenue is that segment, and it should shrink to
    exactly nil afterwards — which is a far stronger statement about the ingest than
    any single year's tie.
    """
    if result.segments is None:
        pytest.skip("no segment vintage in this checkout")

    revenue = result.quarterly["revenue"]
    annual = revenue.groupby(revenue.index.year).sum()
    complete = annual[revenue.groupby(revenue.index.year).size() == 4]

    segments = result.segments.copy()
    segments.index = segments.index.year
    shared = segments.index.intersection(complete.index)
    legacy = complete.reindex(shared) - segments.loc[shared, "total"]

    # Regions can never exceed the consolidated whole.
    assert legacy.min() >= -1.0
    # And the tail is gone by the year after the shutdown.
    for year in (2024, 2025):
        if year in legacy.index:
            assert abs(legacy.loc[year]) <= 1.0


def test_every_segment_fact_carries_its_accession_number(result):
    """``adsh`` is the accession number; without it the traceability claim is unbacked."""
    if result.segments is None:
        pytest.skip("no segment vintage in this checkout")

    accessions = result.segments.attrs.get("accessions", [])
    assert accessions
    assert all(a.startswith("0001065280-") for a in accessions)


def test_control_catches_a_double_counted_region(result):
    """Adding the overlapping US disclosure back in must fail the control."""
    if result.segments is None:
        pytest.skip("no segment vintage in this checkout")

    broken = result.segments.copy()
    broken["total"] = broken["total"] * 1.4  # what including Geographical=US does

    report = run_controls(_context(result, segments=broken))
    assert not _outcome(report, "segment_revenue_foots_to_filed").passed


def test_control_skips_cleanly_without_segment_data(result):
    """The archives are a separate ~340 MB download; the pipeline must run without them."""
    report = run_controls(_context(result, segments=None))
    outcome = _outcome(report, "segment_revenue_foots_to_filed")
    assert outcome.passed
    assert outcome.detail.get("skipped") is True
