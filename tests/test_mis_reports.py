"""Tests for the MIS Builder report definitions.

These run without Odoo. The definitions are plain data, and the failures worth
catching — a KPI referring to an account that is not in the chart, a subtotal that
silently omits a line, a ratio accumulated by summing — are all visible in the data
itself. Catching them here is cheaper than catching them in a rendered report, where
a wrong number still looks like a number.
"""

from __future__ import annotations

import re

import pytest

from fpa.config import (
    BALANCE_SHEET_ASSETS,
    BALANCE_SHEET_EQUITY,
    BALANCE_SHEET_LIABILITIES,
    balance_sheet_lines,
)
from fpa.ledger.mis_reports import BALANCE_SHEET, PROFIT_AND_LOSS, REPORTS
from fpa.ledger.odoo_load import ACCOUNT_MAP, CLEARING_ACCOUNT, REVENUE_ACCOUNT

# ``balp[500000]``, ``bale[1%]``, ``-balp[400000]`` …
_ACCOUNT_REF = re.compile(r"bal[pie]\[([^\]]+)\]")


def _known_codes() -> set[str]:
    codes = {code for code, _n, _t in ACCOUNT_MAP.values()}
    codes.add(REVENUE_ACCOUNT[0])
    codes.add(CLEARING_ACCOUNT[0])
    codes.update(line.code for line, _sign in balance_sheet_lines())
    return codes


@pytest.mark.parametrize("spec", REPORTS, ids=lambda s: s.name)
def test_every_account_reference_exists_in_the_chart(spec):
    """A KPI referencing an account that was never created renders as zero.

    Silently, and looking exactly like a real zero — which is why this is a test
    rather than something to notice in the UI.
    """
    known = _known_codes()
    for kpi in spec.kpis:
        for reference in _ACCOUNT_REF.findall(kpi.expression):
            assert reference in known, f"{kpi.name} references unknown account {reference}"


@pytest.mark.parametrize("spec", REPORTS, ids=lambda s: s.name)
def test_kpi_names_are_unique_and_subtotals_resolve(spec):
    """A subtotal may only reference KPIs defined before it.

    `mis_builder` evaluates in sequence, so a forward reference is a runtime error
    in the report rather than a load-time one here.
    """
    seen: set[str] = set()
    for kpi in spec.kpis:
        assert kpi.name not in seen, f"duplicate KPI name {kpi.name}"
        # Identifiers in the expression that are not account references.
        stripped = _ACCOUNT_REF.sub("", kpi.expression)
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", stripped):
            assert token in seen, f"{kpi.name} references {token} before it is defined"
        seen.add(kpi.name)


@pytest.mark.parametrize("spec", REPORTS, ids=lambda s: s.name)
def test_selection_values_match_the_18_0_schema(spec):
    """Field vocabularies verified against mis_builder 18.0, not assumed."""
    for kpi in spec.kpis:
        assert kpi.kpi_type in {"num", "pct", "str"}
        assert kpi.compare_method in {"diff", "pct", "none"}
        assert kpi.accumulation in {"sum", "avg", "none"}


def test_ratios_are_never_accumulated_by_summing():
    """Summing a margin across twelve months produces a 600% margin.

    The number is nonsense but perfectly plausible-looking in a year-to-date column,
    which is exactly the kind of error that survives review.
    """
    for spec in REPORTS:
        for kpi in spec.kpis:
            if kpi.kpi_type == "pct":
                assert kpi.accumulation == "avg", f"{kpi.name} sums a ratio"


def test_balance_sheet_kpis_never_sum_across_periods():
    """A balance is a position at a date, not a total of twelve positions."""
    for kpi in BALANCE_SHEET:
        assert kpi.accumulation == "none", f"{kpi.name} accumulates a balance"


def test_balance_sheet_report_covers_every_posted_line():
    """The report is generated from the same config the ERP loader posts from.

    If a chart-of-accounts line were added without appearing here, the ERP would
    hold a balance it does not report — and the `balance_check` KPI would be the
    only thing that noticed.
    """
    reported = {kpi.name for kpi in BALANCE_SHEET}
    posted = {line.account for line, _sign in balance_sheet_lines()}
    assert posted <= reported

    expected_subtotals = {"total_assets", "total_liabilities", "total_equity", "balance_check"}
    assert expected_subtotals <= reported


def test_balance_sheet_subtotals_include_every_line_of_their_section():
    """A subtotal that omits a line still foots — against the wrong total."""
    by_name = {kpi.name: kpi for kpi in BALANCE_SHEET}
    sections = {
        "total_assets": BALANCE_SHEET_ASSETS,
        "total_liabilities": BALANCE_SHEET_LIABILITIES,
        "total_equity": BALANCE_SHEET_EQUITY,
    }
    for subtotal, lines in sections.items():
        referenced = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", by_name[subtotal].expression))
        assert {line.account for line in lines} == referenced


def test_credit_balance_lines_are_negated_for_presentation():
    """Liabilities and equity are credits; presenting them raw inverts the statement."""
    by_name = {kpi.name: kpi for kpi in BALANCE_SHEET}
    for line, sign in balance_sheet_lines():
        expression = by_name[line.account].expression
        assert expression.startswith("-") is (sign < 0), (
            f"{line.account} has the wrong presentation sign"
        )


def test_revenue_is_negated_but_expenses_are_not():
    """Revenue is a credit and expenses are debits; both must present positive."""
    by_name = {kpi.name: kpi for kpi in PROFIT_AND_LOSS}
    assert by_name["revenue"].expression.startswith("-")
    for account in ("cost_of_revenue", "research_development", "marketing", "general_administrative"):
        assert not by_name[account].expression.startswith("-")
