"""The three financial statements, assembled from filed facts.

``fpa.ingest.edgar`` produces a tidy fact table. This module turns it into the
three statements a finance function actually works from, and — more to the point —
proves they articulate before anything reads them.

Articulation is the whole reason this layer exists. A pile of XBRL tags is not a
set of financial statements; what makes them statements is that they tie:

* the balance sheet balances,
* the income statement bridges from operating income to net income,
* the cash-flow statement explains the movement in cash,
* and the detail lines partition their filed subtotals.

Every one of those is measured here and re-asserted by a blocking control. Where a
line cannot be obtained as a tag it is derived as a residual — but a residual is
only honest if it is *named* as the thing it represents and its magnitude is
checked. ``content_assets`` comes out at $31.8B against Netflix's reported ~$32B,
which is the difference between a derivation and a plug.
"""

from __future__ import annotations

import logging

import pandas as pd

from fpa.config import (
    ABSENT_MEANS_ZERO,
    BALANCE_SHEET_ASSETS,
    BALANCE_SHEET_EQUITY,
    BALANCE_SHEET_LIABILITIES,
    CASH_FLOW_SECTIONS,
    DERIVED_BALANCE_SHEET,
    DISCLOSURE_ONLY,
    Settings,
)
from fpa.ingest.edgar import quarterly_actuals

logger = logging.getLogger(__name__)

# Filed statements are rounded to thousands, so exact articulation means "to the
# dollar", not "to float epsilon". A residual above a dollar on a $50B balance
# sheet is a real error; below it is presentation rounding.
ARTICULATION_ATOL = 1.0


def _require(frame: pd.DataFrame, columns: tuple[str, ...], what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"{what} needs {missing}, which the fact table does not carry")


def apply_derivations(quarterly: pd.DataFrame) -> pd.DataFrame:
    """Add every derived balance-sheet line as ``total − sum(parts)``.

    Each derivation is a subtraction against *filed* figures, so the resulting
    line is exact by construction: the detail sums back to the subtotal it was
    derived from with no residual at all. What is *modeled* is the claim that the
    residual corresponds to the caption we give it — and that claim is testable by
    magnitude, which ``fpa.controls.derived_balance_sheet_lines`` does.
    """
    out = quarterly.copy()
    assumed_zero: dict[str, int] = {}

    for name, spec in DERIVED_BALANCE_SHEET.items():
        needed = (spec.total, *spec.parts)
        if missing := [c for c in needed if c not in out.columns]:
            logger.warning("cannot derive %s: missing %s", name, missing)
            continue

        required = [p for p in spec.parts if p not in ABSENT_MEANS_ZERO]
        optional = [p for p in spec.parts if p in ABSENT_MEANS_ZERO]

        # skipna=False is the point: a required part that is not filed must
        # propagate NaN, not quietly contribute zero and produce a number that
        # looks derived when it is really assumed.
        parts_total = out[required].sum(axis=1, skipna=False)
        for part in optional:
            assumed_zero[part] = int(out[part].isna().sum())
            parts_total = parts_total + out[part].fillna(0.0)

        out[name] = out[spec.total] - parts_total

    # Surfaced rather than buried: ``fpa.controls.absent_tags_assumed_zero``
    # reports how many periods rest on the absence-means-zero reading.
    out.attrs["assumed_zero"] = assumed_zero
    return out


def balance_sheet(quarterly: pd.DataFrame) -> pd.DataFrame:
    """One row per quarter end, one column per balance-sheet line, in report order.

    Assets carry positive (debit) balances; liabilities and equity carry positive
    values too — the statement is presented, not signed. The ERP loader applies the
    debit/credit sign from :func:`fpa.config.balance_sheet_lines`.
    """
    enriched = apply_derivations(quarterly)
    lines = (*BALANCE_SHEET_ASSETS, *BALANCE_SHEET_LIABILITIES, *BALANCE_SHEET_EQUITY)

    columns: dict[str, pd.Series] = {}
    for line in lines:
        if line.account not in enriched.columns:
            logger.warning("balance-sheet line %s (%s) unavailable", line.code, line.account)
            continue
        series = enriched[line.account]
        if line.account in ABSENT_MEANS_ZERO:
            # Same reading as the derivations: an untagged line is a zero balance,
            # not an unknown one. Applied here too, or dropna below would discard
            # ten otherwise-complete quarters.
            series = series.fillna(0.0)
        columns[line.account] = series

    frame = pd.DataFrame(columns, index=enriched.index)
    # Only quarters where every posted line is present are a balance sheet at all.
    detail = [line.account for line in lines if line.account in frame.columns]
    frame = frame.dropna(subset=detail)

    # Totals ride along so a reader can foot the statement without re-deriving it.
    for total in ("assets", "liabilities", "equity", "assets_current", "liabilities_current"):
        if total in enriched.columns:
            frame[total] = enriched[total]

    # Filed disclosure tags that sit *inside* the captions above rather than
    # beside them. Reported, never posted — posting them would double-count, which
    # is exactly what the first version of this partition did.
    for extra in DISCLOSURE_ONLY:
        if extra in enriched.columns:
            frame[f"{extra}_disclosed"] = enriched[extra]

    # Set attrs last: pandas does not carry them through dropna/assignment.
    frame.attrs["posted_lines"] = detail
    frame.attrs["assumed_zero"] = enriched.attrs.get("assumed_zero", {})
    return frame


def income_statement(quarterly: pd.DataFrame) -> pd.DataFrame:
    """Revenue through to EPS, one row per quarter.

    ``gross_profit`` is computed rather than read: Netflix stopped tagging it after
    2020-12-31, and Revenue − CostOfRevenue is its definition.
    """
    rows = [
        "revenue",
        "cost_of_revenue",
        "research_development",
        "marketing",
        "general_administrative",
        "operating_income",
        "pretax_income",
        "income_tax",
        "net_income",
        "eps_basic",
        "eps_diluted",
        "shares_basic",
        "shares_diluted",
    ]
    frame = quarterly[[c for c in rows if c in quarterly.columns]].copy()
    if {"revenue", "cost_of_revenue"} <= set(frame.columns):
        frame.insert(2, "gross_profit", frame["revenue"] - frame["cost_of_revenue"])
    if {"pretax_income", "operating_income"} <= set(frame.columns):
        # Interest and other, net — the below-the-line residual. Netflix stopped
        # tagging InterestExpense after 2024-09-30, so this is the only route to a
        # complete bridge, and it is exact against operating and pre-tax income.
        frame["non_operating"] = frame["operating_income"] - frame["pretax_income"]
    return frame


def cash_flow(quarterly: pd.DataFrame) -> pd.DataFrame:
    """The cash-flow statement and the roll-forward it must explain.

    Deliberately built here rather than posted to the ERP: no general ledger
    journalizes a cash-flow statement. It is derived from the movement in balance
    sheet accounts, which is why a consolidation tool computes it and a
    transactional system does not.

    ``cash_including_restricted`` is used for the roll-forward, not the
    balance-sheet ``cash`` line. They are different concepts under ASU 2016-18, and
    substituting one for the other breaks the reconciliation by exactly the
    restricted balance — a discrepancy that looks like a bug in the arithmetic.
    """
    _require(quarterly, ("cash_including_restricted",), "cash-flow roll-forward")

    sections = [c for c in CASH_FLOW_SECTIONS if c in quarterly.columns]
    frame = quarterly[sections].copy()
    frame["net_change"] = frame[sections].sum(axis=1)

    for extra in ("depreciation_amortization", "stock_compensation", "capex", "buybacks"):
        if extra in quarterly.columns:
            frame[extra] = quarterly[extra]

    if {"cash_from_operations", "capex"} <= set(quarterly.columns):
        # Capex is filed as a positive outflow, so free cash flow subtracts it.
        frame["free_cash_flow"] = quarterly["cash_from_operations"] - quarterly["capex"]

    cash = quarterly["cash_including_restricted"]
    frame["cash_open"] = cash.shift(1)
    frame["cash_close"] = cash
    frame["roll_forward_residual"] = (
        frame["cash_close"] - frame["cash_open"] - frame["net_change"]
    )
    return frame


def statements(settings: Settings, *, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Build all three statements from the pinned fact vintage."""
    quarterly = quarterly_actuals(settings, refresh=refresh)
    return {
        "income_statement": income_statement(quarterly),
        "balance_sheet": balance_sheet(quarterly),
        "cash_flow": cash_flow(quarterly),
    }
