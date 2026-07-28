"""Tests for the three-statement layer.

Following this project's convention: the tests validate the validators. It is not
enough that the balance sheet balances on today's pinned vintage — the controls
that assert it have to actually fail when it does not, or they are decoration.

Each test names what claim in the README would be void if it failed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpa.config import (
    ABSENT_MEANS_ZERO,
    DERIVED_BALANCE_SHEET,
    EDGAR_TAGS,
    NON_ADDITIVE_ACCOUNTS,
    balance_sheet_lines,
)
from fpa.controls import LedgerContext, run_controls
from fpa.ingest.statements import apply_derivations, balance_sheet, cash_flow, income_statement


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
    )
    return LedgerContext(**{**base, **overrides})


# ---------------------------------------------------------------------------
# Articulation
# ---------------------------------------------------------------------------
def test_balance_sheet_balances_on_filed_totals(result):
    """Assets = Liabilities + Equity, to the dollar.

    If this breaks, the ERP loader's central claim goes with it: the balance-sheet
    journal entry only balances without a clearing account *because* the filed
    statement does.
    """
    bs = result.balance_sheet
    residual = (bs["assets"] - bs["liabilities"] - bs["equity"]).abs()
    assert residual.max() <= 1.0
    assert len(bs) >= 20, "expected the full quarterly window"


def test_posted_lines_partition_the_filed_totals_exactly(result):
    """The detail lines must sum to the filed totals with no gap and no overlap.

    Void this and the ERP holds a balance sheet that is not the filed one, while
    every individual line still looks right.
    """
    bs = result.balance_sheet
    lines = [(l, s) for l, s in balance_sheet_lines() if l.account in bs.columns]
    debits = sum(bs[line.account] for line, sign in lines if sign == 1)
    credits = sum(bs[line.account] for line, sign in lines if sign == -1)

    assert (debits - bs["assets"]).abs().max() <= 1.0
    assert (credits - bs["liabilities"] - bs["equity"]).abs().max() <= 1.0
    # The property the journal entry depends on.
    assert (debits - credits).abs().max() <= 1.0


def test_cash_flow_roll_forward_explains_the_movement_in_cash(result):
    """CFO + CFI + CFF + FX must equal the change in cash including restricted cash."""
    residual = result.cash_flow["roll_forward_residual"].dropna().abs()
    assert not residual.empty
    assert residual.max() <= 1.0


def test_net_income_bridge_holds_and_the_derivation_reproduces_the_filed_tag(result):
    """Pre-tax − tax = net income, and ``pretax = NI + tax`` matches where both exist.

    The second half is what makes the first half meaningful: eight quarters of
    pre-tax income are derived, and the derivation is only trustworthy because it
    is exact across the eighteen quarters where the tag was actually filed.
    """
    q = result.quarterly
    rows = q[["pretax_income", "income_tax", "net_income"]].dropna()
    assert (rows["pretax_income"] - rows["income_tax"] - rows["net_income"]).abs().max() <= 1.0

    overlap = q[["pretax_filed", "pretax_derived"]].dropna()
    assert len(overlap) >= 18
    assert (overlap["pretax_filed"] - overlap["pretax_derived"]).abs().max() <= 1.0


# ---------------------------------------------------------------------------
# The controls must bite
# ---------------------------------------------------------------------------
def test_balance_sheet_control_catches_an_unbalanced_statement(result):
    """A balance sheet that does not balance must block the pipeline."""
    broken = result.balance_sheet.copy()
    broken.loc[broken.index[-1], "assets"] += 1_000_000.0

    report = run_controls(_context(result, balance_sheet=broken))
    outcome = _outcome(report, "balance_sheet_balances")
    assert not outcome.passed
    assert not report.passed, "an unbalanced balance sheet must be blocking"


def test_partition_control_catches_a_double_counted_line(result):
    """Posting a disclosure tag as a face line must be caught.

    This is not a hypothetical corruption — it is the bug the lease tags actually
    caused, and the reason the partition excludes them.
    """
    broken = result.balance_sheet.copy()
    broken["ppe_net"] = broken["ppe_net"] * 2

    report = run_controls(_context(result, balance_sheet=broken))
    assert not _outcome(report, "balance_sheet_partition").passed


def test_sign_control_catches_a_wrong_signed_residual(result):
    """A negative content-asset balance is not a balance sheet line."""
    broken = result.balance_sheet.copy()
    broken["content_assets"] = -broken["content_assets"]

    report = run_controls(_context(result, balance_sheet=broken))
    outcome = _outcome(report, "derived_balance_sheet_lines")
    assert not outcome.passed
    assert "content_assets" in outcome.detail["offenders"]


def test_cash_flow_control_catches_a_broken_roll_forward(result):
    """A cash-flow statement that does not explain the movement in cash must block."""
    broken = result.cash_flow.copy()
    broken.loc[broken.index[-1], "roll_forward_residual"] = 5_000_000.0

    report = run_controls(_context(result, cash_flow=broken))
    assert not _outcome(report, "cash_flow_articulates").passed


def test_eps_control_catches_a_unit_regression(result):
    """Reading share counts as dollars must be visible.

    ``eps_consistency`` exists to catch exactly one regression: the ingest going
    back to a hardcoded ``USD`` unit. That failure is silent — the per-share tags
    would not error, they would vanish — so the control has to be the thing that
    notices.
    """
    broken_quarterly = result.quarterly.copy()
    broken_quarterly["shares_basic"] = broken_quarterly["shares_basic"] * 1000

    report = run_controls(_context(result, quarterly=broken_quarterly))
    assert not _outcome(report, "eps_consistency").passed


# ---------------------------------------------------------------------------
# Ingest correctness
# ---------------------------------------------------------------------------
def test_every_configured_tag_declares_a_unit(result):
    """Three XBRL units are in play; none may be left to a default."""
    units = {spec.unit for spec in EDGAR_TAGS.values()}
    assert units == {"USD", "shares", "USD/shares"}
    assert result.facts["unit"].notna().all()


def test_non_additive_accounts_are_never_derived_by_differencing(result):
    """Q4 share counts and EPS may be *filed*, but must never be *differenced*.

    ``FY − (Q1+Q2+Q3)`` is meaningless for a weighted-average share count and for a
    ratio. Leaving the cell empty is the correct answer; filling it produces a
    number that looks real and is not.

    Note the distinction the assertion turns on. Netflix does tag Q4 2020 diluted
    EPS as a three-month fact ($1.19), and that value is kept — it is reported, not
    computed. So the claim under test is about *provenance*, not absence: no
    non-additive account may appear in any year's derived list.
    """
    for year, info in result.metadata["q4_provenance"].items():
        derived = set(info.get("derived", []))
        offenders = derived & NON_ADDITIVE_ACCOUNTS
        assert not offenders, f"FY{year} differenced {sorted(offenders)} into Q4"


def test_q4_derivation_refuses_an_incomplete_year(result):
    """Q4 = FY − (Q1+Q2+Q3) requires all three quarters, not however many exist.

    Pandas' ``sum`` skips NaN, so a partially-filed account yields ``FY − Q3`` and
    buries two missing quarters in the derived one. That produced a $1.83B error in
    2020-Q4 pre-tax income before the guard existed.
    """
    provenance = result.metadata["q4_provenance"]
    incomplete = {
        year: info["incomplete_quarters"]
        for year, info in provenance.items()
        if info.get("incomplete_quarters")
    }
    # 2020 is the known case: pre-tax income is only tagged as a quarter from Q3.
    assert 2020 in incomplete
    assert "pretax_income" in incomplete[2020]


def test_derivations_propagate_nan_for_a_missing_required_part(result):
    """A required part that is not filed must produce NaN, not a silent zero."""
    quarterly = result.quarterly.copy()
    quarterly.loc[quarterly.index[-1], "ppe_net"] = pd.NA

    enriched = apply_derivations(quarterly)
    assert pd.isna(enriched.loc[enriched.index[-1], "content_assets"])


def test_absent_means_zero_is_declared_rather_than_assumed(result):
    """The absence-means-zero reading must be counted and reported."""
    assumed = result.balance_sheet.attrs["assumed_zero"]
    assert set(assumed) <= set(ABSENT_MEANS_ZERO)
    assert assumed["short_term_investments"] > 0

    report = run_controls(_context(result))
    outcome = _outcome(report, "absent_tags_assumed_zero")
    assert outcome.passed
    assert "short_term_investments" in outcome.message


def test_every_derived_line_holds_its_expected_sign(result):
    """Content assets cannot be negative; treasury stock cannot be positive."""
    for name, spec in DERIVED_BALANCE_SHEET.items():
        series = result.balance_sheet[name].dropna()
        assert (series * spec.sign).min() >= 0, f"{name} broke its expected sign"


def test_income_statement_computes_gross_profit_rather_than_reading_it(result):
    """``GrossProfit`` stops being tagged after 2020; the definition does not."""
    statement = result.income_statement
    assert "gross_profit" in statement.columns
    computed = statement["revenue"] - statement["cost_of_revenue"]
    assert (statement["gross_profit"] - computed).abs().max() <= 1.0
