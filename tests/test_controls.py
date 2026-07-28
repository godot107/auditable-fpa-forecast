"""Tests for the control layer — including that the controls actually fail.

Same principle as the groundedness tests: a control suite that always passes is a
liability, because it certifies nothing while looking like assurance. Each blocking
control is fed deliberately corrupted data and must catch it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpa.controls import LedgerContext, Severity, run_controls


def _context(result) -> LedgerContext:
    return LedgerContext(
        quarterly=result.quarterly,
        ledger=result.ledger,
        revenue=result.revenue,
        drivers=result.drivers,
        budget=result.budget,
        facts=result.facts,
    )


def _outcome(report, name: str):
    return next(r for r in report.results if r.name == name)


def test_clean_run_opens_the_gate(result):
    """If this fails, no downstream claim in the project holds."""
    assert result.controls.passed
    assert not result.controls.blocking_failures


def test_corrupting_the_ledger_trips_the_footing_control(result):
    """The claim 'modeled but not invented' rests entirely on this control.

    Move a single dollar of allocated cost and the ledger no longer foots to the
    filed figure it was derived from.
    """
    ctx = _context(result)
    broken = ctx.ledger.copy()
    broken.loc[broken.index[0], "amount"] += 1.0
    ctx.ledger = broken

    report = run_controls(ctx)
    assert not report.passed
    assert not _outcome(report, "ledger_foots_to_filed").passed


def test_a_missing_month_trips_period_completeness(result):
    """A gap silently biases every rolling statistic downstream."""
    ctx = _context(result)
    dropped = sorted(ctx.ledger["period"].unique())[len(ctx.ledger["period"].unique()) // 2]
    ctx.ledger = ctx.ledger[ctx.ledger["period"] != dropped]

    report = run_controls(ctx)
    assert not _outcome(report, "period_completeness").passed


def test_an_unknown_cost_center_is_caught(result):
    """Orphan spend would never roll up — the classic way a total stops tying."""
    ctx = _context(result)
    broken = ctx.ledger.copy()
    broken.loc[broken.index[0], "sub_center"] = "Ministry of Silly Walks"
    ctx.ledger = broken

    assert not _outcome(run_controls(ctx), "cost_centers_known").passed


def test_a_negative_allocation_is_caught(result):
    ctx = _context(result)
    broken = ctx.ledger.copy()
    broken.loc[broken.index[0], "amount"] = -1.0
    ctx.ledger = broken

    assert not _outcome(run_controls(ctx), "no_negative_expense").passed


def test_breaking_the_driver_identity_is_caught(result):
    """members x ARPU must equal revenue, or the rate/volume split is meaningless."""
    ctx = _context(result)
    broken = ctx.drivers.copy()
    broken.loc[broken.index[0], "members"] *= 1.5
    ctx.drivers = broken

    assert not _outcome(run_controls(ctx), "driver_identity").passed


def test_a_control_that_raises_counts_as_a_failure(result):
    """A crashed control must never read as a pass."""
    ctx = _context(result)
    ctx.ledger = pd.DataFrame({"nonsense": [1]})

    report = run_controls(ctx)
    assert not report.passed


def test_blocking_and_warn_are_distinguished(result):
    """WARN must not halt the pipeline; only BLOCKING may."""
    severities = {r.severity for r in result.controls.results}
    assert Severity.BLOCKING in severities
    assert Severity.WARN in severities
    assert result.controls.warnings  # budget coverage + restatements are expected
    assert result.controls.passed  # ...and neither blocks


def test_pass_rate_is_a_real_fraction(result):
    assert 0.0 <= result.controls.pass_rate <= 1.0
