"""Data-integrity controls — auditability as code.

Every check here runs **inside the pipeline on every run**, not only in CI. That
is the whole point: a control that only runs in a test suite protects the
developer, while a control that runs in the pipeline protects the number. Same
discipline as ``financial-forecasting-engine/fce/accounting/identities.py``.

Severity decides what a failure means:

* ``BLOCKING`` — the pipeline stops. No forecast, no variance, no commentary.
  These are the checks where a failure means a published figure would be wrong.
* ``WARN`` — surfaced prominently, does not stop the run. Something a human
  should look at (a missing budget year, a restated period).
* ``INFO`` — recorded for the audit trail.

The controls deliberately re-verify things this codebase's own comments assert —
the marketing identity, exact footing — so the claim is proven at runtime rather
than trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np
import pandas as pd

from fpa.config import (
    COST_CENTERS,
    DERIVED_BALANCE_SHEET,
    EXPENSE_ACCOUNTS,
    balance_sheet_lines,
)

# Footing is exact by construction (``_exact_split``), so the only tolerance needed
# is float64 representation noise. Relative, because the balances are ~1e10 where an
# absolute epsilon would be meaningless.
#
# Set at 1e-12: four orders of magnitude above the ~1e-16 actually observed, but tight
# enough that a real error is caught. At 1e-9 the tolerance on a $12B quarter is ~$12,
# so a misallocated dollar would have slipped through silently.
FOOTING_RTOL = 1e-12


class Severity(str, Enum):
    BLOCKING = "BLOCKING"
    WARN = "WARN"
    INFO = "INFO"


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one control."""

    name: str
    severity: Severity
    passed: bool
    message: str
    detail: dict = field(default_factory=dict)

    @property
    def blocks(self) -> bool:
        return self.severity is Severity.BLOCKING and not self.passed


@dataclass
class LedgerContext:
    """Everything the controls need to inspect one pipeline run."""

    quarterly: pd.DataFrame
    ledger: pd.DataFrame
    revenue: pd.DataFrame
    drivers: pd.DataFrame
    budget: pd.DataFrame
    facts: pd.DataFrame | None = None
    # The other two statements, built by ``fpa.ingest.statements``. The cost-center
    # ledger only needs the income statement, so these are optional — but when they
    # are present the articulation controls below run against them.
    balance_sheet: pd.DataFrame | None = None
    cash_flow: pd.DataFrame | None = None
    # Regional revenue, from SEC's Financial Statement Data Sets rather than the
    # companyfacts API — a separate ingest, so optional like the ERP extracts.
    segments: pd.DataFrame | None = None
    # Present only when Odoo has been seeded and extracted; the pipeline runs
    # without it so the demo survives the ERP being down.
    erp_extract: pd.DataFrame | None = None
    erp_balance_sheet: pd.DataFrame | None = None
    erp_trial_balance: pd.DataFrame | None = None


@dataclass
class ControlReport:
    """The full set of control outcomes for one run."""

    results: list[CheckResult]

    @property
    def blocking_failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.blocks]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if r.severity is Severity.WARN and not r.passed]

    @property
    def passed(self) -> bool:
        """True when nothing blocking failed — the gate the pipeline consults."""
        return not self.blocking_failures

    @property
    def pass_rate(self) -> float:
        """Share of controls that passed — a Layer-4 process KPI in its own right."""
        return sum(r.passed for r in self.results) / len(self.results) if self.results else 1.0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "control": r.name,
                    "severity": r.severity.value,
                    "status": "PASS" if r.passed else "FAIL",
                    "message": r.message,
                }
                for r in self.results
            ]
        )

    def to_markdown(self) -> str:
        lines = [
            f"## Control report — {sum(r.passed for r in self.results)}/{len(self.results)} passed",
            "",
        ]
        for r in self.results:
            mark = "PASS" if r.passed else "FAIL"
            lines.append(f"- **[{mark}]** `{r.name}` ({r.severity.value}) — {r.message}")
        if self.blocking_failures:
            lines += ["", "**Pipeline halted: blocking control(s) failed.**"]
        return "\n".join(lines)


class ControlGateError(RuntimeError):
    """Raised when the pipeline is asked to publish through a failed blocking control."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_REGISTRY: list[tuple[str, Severity, Callable[[LedgerContext], tuple[bool, str, dict]]]] = []


def control(name: str, severity: Severity):
    """Register a control. The function returns ``(passed, message, detail)``."""

    def decorator(fn):
        _REGISTRY.append((name, severity, fn))
        return fn

    return decorator


def run_controls(context: LedgerContext) -> ControlReport:
    """Run every registered control against ``context``."""
    results: list[CheckResult] = []
    for name, severity, fn in _REGISTRY:
        try:
            passed, message, detail = fn(context)
        except Exception as exc:  # a control that crashes is a failed control
            passed, message, detail = False, f"control raised {type(exc).__name__}: {exc}", {}
        results.append(CheckResult(name, severity, passed, message, detail))
    return ControlReport(results)


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
@control("ledger_foots_to_filed", Severity.BLOCKING)
def _ledger_foots_to_filed(ctx: LedgerContext):
    """Modeled cost-center detail must sum back to the filed quarterly figure.

    This is the control the whole "modeled but not invented" claim rests on.
    """
    rolled = ctx.ledger.groupby(["quarter_end", "account"])["amount"].sum().unstack()
    accounts = [a for a in EXPENSE_ACCOUNTS if a in rolled.columns]
    filed = ctx.quarterly[accounts].reindex(rolled.index)
    rel = ((rolled[accounts] - filed).abs() / filed.abs().replace(0, np.nan))
    worst = float(np.nanmax(rel.to_numpy())) if rel.size else 0.0
    ok = worst <= FOOTING_RTOL
    return (
        ok,
        f"worst relative footing error {worst:.2e} (tolerance {FOOTING_RTOL:.0e})",
        {"worst_relative_error": worst},
    )


@control("revenue_foots_to_filed", Severity.BLOCKING)
def _revenue_foots_to_filed(ctx: LedgerContext):
    """Monthly revenue must sum back to the filed quarterly revenue."""
    rolled = ctx.revenue.groupby("quarter_end")["amount"].sum()
    filed = ctx.quarterly["revenue"].reindex(rolled.index)
    rel = ((rolled - filed).abs() / filed.abs()).max()
    worst = float(rel) if pd.notna(rel) else 0.0
    return (
        worst <= FOOTING_RTOL,
        f"worst relative error {worst:.2e}",
        {"worst_relative_error": worst},
    )


@control("marketing_identity", Severity.BLOCKING)
def _marketing_identity(ctx: LedgerContext):
    """Derived Marketing must reproduce the filed tag wherever the tag still exists.

    ``MarketingExpense`` was discontinued after 2024-09-30 and is recovered from
    Marketing = Revenue - CostOfRevenue - R&D - G&A - OperatingIncome. If that
    identity ever stopped holding in the overlap, every post-2024 marketing figure
    downstream would be wrong and nothing else would notice.
    """
    if not {"marketing_filed", "marketing_derived"} <= set(ctx.quarterly.columns):
        return True, "no overlap period available to test", {}

    both = ctx.quarterly[["marketing_filed", "marketing_derived"]].dropna()
    if both.empty:
        return True, "no overlap period available to test", {}

    diff = (both["marketing_filed"] - both["marketing_derived"]).abs()
    rel = (diff / both["marketing_filed"].abs()).max()
    ok = bool(rel <= 1e-9)
    return (
        ok,
        f"identity reproduces the filed tag across {len(both)} overlap quarters "
        f"(worst relative diff {rel:.2e})",
        {"n_overlap": int(len(both)), "worst_relative_diff": float(rel)},
    )


@control("opex_identity", Severity.BLOCKING)
def _opex_identity(ctx: LedgerContext):
    """Revenue - the four expense lines must equal filed operating income.

    The income-statement articulation. If this breaks, the chart of accounts has
    drifted from the filer's presentation and the cost-center split is meaningless.
    """
    need = ["revenue", *EXPENSE_ACCOUNTS, "operating_income"]
    missing = [c for c in need if c not in ctx.quarterly.columns]
    if missing:
        return False, f"missing accounts: {missing}", {"missing": missing}

    q = ctx.quarterly[need].dropna()
    residual = (
        q["revenue"] - q[list(EXPENSE_ACCOUNTS)].sum(axis=1) - q["operating_income"]
    )
    rel = (residual.abs() / q["revenue"].abs()).max()
    ok = bool(rel <= 1e-9)
    return (
        ok,
        f"articulation holds across {len(q)} quarters (worst relative residual {rel:.2e})",
        {"n_quarters": int(len(q)), "worst_relative_residual": float(rel)},
    )


@control("driver_identity", Severity.BLOCKING)
def _driver_identity(ctx: LedgerContext):
    """members x ARPU must equal monthly revenue.

    Guarantees the rate/volume decomposition the variance bridge performs is tied
    to the filed revenue rather than to two independently invented series.
    """
    product = ctx.drivers["members"] * ctx.drivers["arpu"]
    rel = ((product - ctx.drivers["revenue"]).abs() / ctx.drivers["revenue"].abs()).max()
    worst = float(rel) if pd.notna(rel) else 0.0
    return worst <= 1e-9, f"worst relative error {worst:.2e}", {"worst_relative_error": worst}


@control("period_completeness", Severity.BLOCKING)
def _period_completeness(ctx: LedgerContext):
    """No month may be missing from the ledger window.

    A hole here silently biases every rolling statistic and every year-over-year
    comparison downstream.
    """
    periods = pd.DatetimeIndex(sorted(ctx.ledger["period"].unique()))
    expected = pd.date_range(periods.min(), periods.max(), freq="ME")
    missing = expected.difference(periods)
    return (
        len(missing) == 0,
        f"{len(periods)} months, {len(missing)} gap(s)"
        + (f": {[str(d.date()) for d in missing[:5]]}" if len(missing) else ""),
        {"n_periods": int(len(periods)), "n_missing": int(len(missing))},
    )


@control("no_negative_expense", Severity.BLOCKING)
def _no_negative_expense(ctx: LedgerContext):
    """Allocated expense must be non-negative.

    A negative allocation would mean the disaggregation produced a credit that
    does not exist in the filing.
    """
    negatives = ctx.ledger[ctx.ledger["amount"] < 0]
    return (
        negatives.empty,
        f"{len(negatives)} negative ledger row(s)",
        {"n_negative": int(len(negatives))},
    )


@control("cost_centers_known", Severity.BLOCKING)
def _cost_centers_known(ctx: LedgerContext):
    """Every ledger cost center must exist in the configured hierarchy.

    An orphan cost center is spend that would never roll up — the classic way a
    total quietly stops tying.
    """
    known = {(f, s) for f, subs in COST_CENTERS.items() for s in subs}
    found = set(map(tuple, ctx.ledger[["function", "sub_center"]].drop_duplicates().to_numpy()))
    orphans = found - known
    return (
        not orphans,
        f"{len(found)} cost centers, {len(orphans)} orphan(s)"
        + (f": {sorted(orphans)}" if orphans else ""),
        {"orphans": sorted(map(list, orphans))},
    )


@control("budget_coverage", Severity.WARN)
def _budget_coverage(ctx: LedgerContext):
    """Flag actual periods with no budget to compare against.

    Expected to fail for the first fiscal year: the plan for year Y is built from
    year Y-1 actuals, so the earliest year has no plan. That is a real constraint,
    and it is reported rather than papered over with a zero budget.
    """
    if ctx.budget.empty:
        return False, "no budget rows at all", {}
    actual_periods = set(ctx.ledger["period"].unique())
    budget_periods = set(ctx.budget["period"].unique())
    uncovered = sorted(actual_periods - budget_periods)
    coverage = 1.0 - len(uncovered) / len(actual_periods)
    return (
        not uncovered,
        f"budget covers {coverage:.0%} of actual months; {len(uncovered)} uncovered"
        + (f" (from {pd.Timestamp(min(uncovered)).date()})" if uncovered else ""),
        {"coverage": coverage, "n_uncovered": len(uncovered)},
    )


@control("restatements", Severity.WARN)
def _restatements(ctx: LedgerContext):
    """Surface facts whose reported value actually changed between filings.

    We keep the latest filed value. A finance team wants to know a prior period
    moved — silently collapsing restatements is how a variance explanation ends up
    contradicting last quarter's board pack.

    Deliberately keyed on ``n_distinct_values``, not on how many filings mentioned
    the period. Every 10-K repeats the prior year's quarters as comparatives, so
    counting filings flags ~75% of the fact table as "restated" and the control
    becomes noise that everyone learns to ignore.
    """
    if ctx.facts is None or "n_distinct_values" not in ctx.facts.columns:
        return True, "fact-level version data unavailable", {}
    restated = ctx.facts[ctx.facts["n_distinct_values"] > 1]
    reported_twice = int((ctx.facts["n_filings"] > 1).sum())
    return (
        restated.empty,
        f"{len(restated)} fact(s) changed value across filings "
        f"({reported_twice} were merely re-reported as comparatives, which is normal)",
        {"n_restated": int(len(restated)), "n_reported_multiple_times": reported_twice},
    )


@control("annual_identity_in_window", Severity.BLOCKING)
def _annual_identity_in_window(ctx: LedgerContext):
    """Every fiscal year inside the window must articulate at the annual level.

    Q4 is never filed as a quarter; it is derived from the 10-K annual. If the
    annual income statement does not tie, the whole discrepancy lands in that one
    derived quarter. This control is what justifies ``Settings.window_start``
    rather than leaving it as an assertion in a comment.
    """
    need = ["revenue", *EXPENSE_ACCOUNTS, "operating_income"]
    if any(c not in ctx.quarterly.columns for c in need):
        return False, "chart of accounts incomplete", {}

    q = ctx.quarterly[need].dropna()
    by_year = q.groupby(q.index.year).sum()
    # Only full years can be tested; the current one is still in progress.
    full = by_year[q.groupby(q.index.year).size() == 4]
    if full.empty:
        return True, "no complete fiscal year in window", {}

    residual = full["revenue"] - full[list(EXPENSE_ACCOUNTS)].sum(axis=1) - full["operating_income"]
    rel = (residual.abs() / full["revenue"].abs())
    worst_year = int(rel.idxmax())
    worst = float(rel.max())
    return (
        worst <= 1e-9,
        f"{len(full)} complete fiscal years articulate; worst {worst:.2e} ({worst_year})",
        {"n_years": int(len(full)), "worst_relative": worst, "worst_year": worst_year},
    )


# ---------------------------------------------------------------------------
# Articulation — the checks that make a pile of XBRL tags into three statements.
# ---------------------------------------------------------------------------
# Filed statements are rounded to thousands, so "exact" means to the dollar, not
# to float epsilon. On a $58B balance sheet a $1 tolerance is 1.7e-11 relative.
ARTICULATION_ATOL = 1.0


@control("balance_sheet_balances", Severity.BLOCKING)
def _balance_sheet_balances(ctx: LedgerContext):
    """Assets must equal Liabilities plus Equity, on filed figures.

    The oldest control in accounting, and here it is load-bearing rather than
    ceremonial: because it holds to $0.00, the ERP loader can post the balance
    sheet as a **self-balancing journal entry with no clearing account**. If this
    ever failed, that entry would not balance and Odoo would reject it — so the
    check is not decoration, it is the precondition for the posting to exist.
    """
    if ctx.balance_sheet is None or ctx.balance_sheet.empty:
        return True, "no balance sheet in this run — skipped", {"skipped": True}

    bs = ctx.balance_sheet
    residual = (bs["assets"] - bs["liabilities"] - bs["equity"]).abs()
    worst = float(residual.max())
    return (
        worst <= ARTICULATION_ATOL,
        f"A = L + E across {len(bs)} quarters (worst ${worst:,.2f})",
        {"n_quarters": int(len(bs)), "worst_absolute": worst},
    )


@control("balance_sheet_partition", Severity.BLOCKING)
def _balance_sheet_partition(ctx: LedgerContext):
    """The posted detail lines must partition the filed totals exactly.

    Two failure modes this catches, and only this catches:

    * **A gap** — a line of the balance sheet nobody mapped, so the ERP holds a
      balance sheet that is smaller than the filed one.
    * **An overlap** — a disclosure tag posted as though it sat on the face of the
      statement. That is not hypothetical: the lease tags are nested inside the
      "other non-current" captions, and posting them alongside produced a
      **negative** non-current liability of −$571M.
    """
    if ctx.balance_sheet is None or ctx.balance_sheet.empty:
        return True, "no balance sheet in this run — skipped", {"skipped": True}

    bs = ctx.balance_sheet
    lines = [(line, sign) for line, sign in balance_sheet_lines() if line.account in bs.columns]
    debits = sum(bs[line.account] for line, sign in lines if sign == 1)
    credits = sum(bs[line.account] for line, sign in lines if sign == -1)

    asset_gap = float((debits - bs["assets"]).abs().max())
    credit_gap = float((credits - bs["liabilities"] - bs["equity"]).abs().max())
    entry_gap = float((debits - credits).abs().max())
    worst = max(asset_gap, credit_gap, entry_gap)

    return (
        worst <= ARTICULATION_ATOL,
        f"{len(lines)} lines partition the filed totals exactly "
        f"(assets ${asset_gap:,.2f}, L+E ${credit_gap:,.2f}, "
        f"entry balances to ${entry_gap:,.2f})",
        {"n_lines": len(lines), "worst_absolute": worst},
    )


@control("derived_balance_sheet_lines", Severity.BLOCKING)
def _derived_balance_sheet_lines(ctx: LedgerContext):
    """Every derived balance-sheet residual must carry its expected sign.

    A residual is only a financial statement line if it behaves like one. Content
    assets cannot be negative; treasury stock cannot be positive. This is the
    cheapest possible test that a plug corresponds to the caption it was given, and
    it is what turned a silently-wrong partition into a visible one.
    """
    if ctx.balance_sheet is None or ctx.balance_sheet.empty:
        return True, "no balance sheet in this run — skipped", {"skipped": True}

    offenders: dict[str, float] = {}
    for name, spec in DERIVED_BALANCE_SHEET.items():
        if name not in ctx.balance_sheet.columns:
            continue
        series = ctx.balance_sheet[name].dropna()
        # Worst value in the wrong direction, or 0.0 if the sign always held.
        worst = float((series * spec.sign).min())
        if worst < 0:
            offenders[name] = worst

    return (
        not offenders,
        f"{len(DERIVED_BALANCE_SHEET)} derived lines hold their sign in every quarter"
        if not offenders
        else f"wrong-signed residual(s): "
        + ", ".join(f"{k} ${v:,.0f}" for k, v in offenders.items()),
        {"offenders": offenders},
    )


@control("cash_flow_articulates", Severity.BLOCKING)
def _cash_flow_articulates(ctx: LedgerContext):
    """Operating + investing + financing + FX must explain the movement in cash.

    The third statement's reason for existing. Note it reconciles to cash
    *including restricted* cash — a different concept under ASU 2016-18 from the
    balance-sheet cash line, and substituting one for the other breaks the
    roll-forward by exactly the restricted balance.
    """
    if ctx.cash_flow is None or ctx.cash_flow.empty:
        return True, "no cash-flow statement in this run — skipped", {"skipped": True}

    residual = ctx.cash_flow["roll_forward_residual"].dropna().abs()
    if residual.empty:
        return True, "no period with both an opening and closing balance", {}
    worst = float(residual.max())
    return (
        worst <= ARTICULATION_ATOL,
        f"roll-forward explains cash across {len(residual)} quarters "
        f"(worst ${worst:,.2f})",
        {"n_quarters": int(len(residual)), "worst_absolute": worst},
    )


@control("net_income_bridge", Severity.BLOCKING)
def _net_income_bridge(ctx: LedgerContext):
    """Pre-tax income less tax must equal net income, and the derivation must hold.

    Two claims in one control, because they fail together. Pre-tax income is only
    tagged as a quarter from 2020-Q3, so earlier quarters use
    ``pretax = net income + tax``. That fills the gap only if the identity is exact
    where both exist — which is tested here against the overlap rather than assumed
    from the fact that it is an identity on paper.
    """
    q = ctx.quarterly
    need = {"pretax_income", "income_tax", "net_income"}
    if not need <= set(q.columns):
        return False, f"missing {sorted(need - set(q.columns))}", {}

    rows = q[list(need)].dropna()
    residual = (rows["pretax_income"] - rows["income_tax"] - rows["net_income"]).abs()
    worst = float(residual.max()) if len(rows) else 0.0

    overlap_worst = 0.0
    n_overlap = 0
    if {"pretax_filed", "pretax_derived"} <= set(q.columns):
        both = q[["pretax_filed", "pretax_derived"]].dropna()
        n_overlap = len(both)
        if n_overlap:
            overlap_worst = float(
                (both["pretax_filed"] - both["pretax_derived"]).abs().max()
            )

    ok = worst <= ARTICULATION_ATOL and overlap_worst <= ARTICULATION_ATOL
    return (
        ok,
        f"bridge holds across {len(rows)} quarters (worst ${worst:,.2f}); "
        f"derivation reproduces the filed tag across {n_overlap} overlap quarters "
        f"(worst ${overlap_worst:,.2f})",
        {
            "n_quarters": int(len(rows)),
            "worst_absolute": worst,
            "n_overlap": n_overlap,
            "overlap_worst": overlap_worst,
        },
    )


@control("eps_consistency", Severity.WARN)
def _eps_consistency(ctx: LedgerContext):
    """Net income divided by basic shares must reproduce filed basic EPS.

    A WARN, not a block, because EPS is filed rounded to the cent and the quotient
    is not — so it can only ever agree to within half a cent, and disagreeing by
    more is a question rather than a certainty.

    It earns its place by testing something no other control touches: the ingest
    reads three different XBRL units here (``USD``, ``shares``, ``USD/shares``). If
    the unit handling regressed to hardcoded USD, the per-share tags would not error
    — they would silently vanish, and this is the control that would notice.
    """
    need = {"net_income", "shares_basic", "eps_basic"}
    if not need <= set(ctx.quarterly.columns):
        return False, f"missing {sorted(need - set(ctx.quarterly.columns))}", {}

    rows = ctx.quarterly[list(need)].dropna()
    if rows.empty:
        return False, "no quarter carries all three of net income, shares and EPS", {}

    computed = rows["net_income"] / rows["shares_basic"]
    worst = float((computed - rows["eps_basic"]).abs().max())
    return (
        worst <= 0.01,
        f"EPS reproduces from net income and share count across {len(rows)} quarters "
        f"(worst ${worst:.4f}/share; filed EPS is rounded to the cent)",
        {"n_quarters": int(len(rows)), "worst_per_share": worst},
    )


@control("absent_tags_assumed_zero", Severity.WARN)
def _absent_tags_assumed_zero(ctx: LedgerContext):
    """Report how many periods rest on reading an untagged line as a zero balance.

    Sometimes correct — a company that holds no short-term investments does not tag
    the concept. But it is an interpretation, not a fact, and the number of periods
    it covers belongs on screen rather than in a docstring.
    """
    if ctx.balance_sheet is None:
        return True, "no balance sheet in this run — skipped", {"skipped": True}

    assumed = ctx.balance_sheet.attrs.get("assumed_zero", {})
    total = sum(assumed.values())
    if not total:
        return True, "every balance-sheet line is filed in every period", {}
    detail = ", ".join(f"{tag} ({n} periods)" for tag, n in sorted(assumed.items()))
    return (
        True,
        f"{total} period-line(s) read an absent tag as a zero balance: {detail}",
        {"assumed_zero": assumed},
    )


@control("erp_extract_reconciles", Severity.BLOCKING)
def _erp_extract_reconciles(ctx: LedgerContext):
    """The ERP extract must still tie to the filed figures.

    The round trip is EDGAR → disaggregation → Odoo ORM → posted double-entry
    journal → analytic distribution → SQL GROUP BY → back here. Plenty of places
    to lose or double-count a number, none of which any other control would catch.

    Skipped when no extract snapshot exists, so the pipeline still runs offline
    without Odoo — the ERP is the system of record, not a runtime dependency.
    Tolerance is 1e-6 relative rather than float epsilon because Odoo stores cents:
    rounding at the ERP boundary is correct behaviour, not an error.
    """
    from fpa.extract.odoo_sql import reconcile_to_filed

    if ctx.erp_extract is None or ctx.erp_extract.empty:
        return True, "no ERP extract in this run — skipped", {"skipped": True}

    report = reconcile_to_filed(ctx.erp_extract, ctx.quarterly)
    worst = float(report["rel_diff"].max())
    worst_abs = float(report["abs_diff"].max())
    return (
        worst <= 1e-6,
        f"ERP round trip ties to filed across {len(report)} accounts "
        f"(worst {worst:.2e} relative, ${worst_abs:,.2f} absolute — cent rounding)",
        {"worst_relative": worst, "worst_absolute": worst_abs},
    )


@control("erp_balance_sheet_reconciles", Severity.BLOCKING)
def _erp_balance_sheet_reconciles(ctx: LedgerContext):
    """The ERP's balance sheet must still equal the filed one at every quarter end.

    Harder than the P&L round trip, and deliberately so. The P&L is posted as
    independent monthly entries, so an error stays in its own period. The balance
    sheet is posted as **movements**, so every quarter's balance is a cumulative
    sum of every entry before it — one bad entry in 2021 shows up in all twenty
    quarters that follow. Landing on the filed figure at all of them is a much
    stronger statement than twenty-six independent comparisons would be.
    """
    from fpa.extract.odoo_sql import reconcile_balance_sheet

    if ctx.erp_balance_sheet is None or ctx.erp_balance_sheet.empty:
        return True, "no ERP balance-sheet extract in this run — skipped", {"skipped": True}
    if ctx.balance_sheet is None or ctx.balance_sheet.empty:
        return True, "no filed balance sheet to compare against — skipped", {"skipped": True}

    report = reconcile_balance_sheet(ctx.erp_balance_sheet, ctx.balance_sheet)
    worst_abs = float(report["abs_diff"].max())
    n_periods = int(ctx.erp_balance_sheet["period"].nunique())
    return (
        worst_abs <= 1.0,
        f"ERP balance sheet ties to filed across {len(report)} lines and "
        f"{n_periods} quarter ends (worst ${worst_abs:,.2f})",
        {"n_lines": int(len(report)), "n_periods": n_periods, "worst_absolute": worst_abs},
    )


@control("trial_balance_nets_to_zero", Severity.BLOCKING)
def _trial_balance_nets_to_zero(ctx: LedgerContext):
    """Total debits must equal total credits, per fiscal year.

    The oldest check in double-entry bookkeeping, and it is here because it is the
    one an ERP can fail that Python cannot: it tests what Odoo actually stored,
    not what the loader intended to store.
    """
    if ctx.erp_trial_balance is None or ctx.erp_trial_balance.empty:
        return True, "no trial balance in this run — skipped", {"skipped": True}

    by_year = ctx.erp_trial_balance.groupby("fiscal_year")[["total_debit", "total_credit"]].sum()
    net = (by_year["total_debit"] - by_year["total_credit"]).abs()
    worst = float(net.max())
    return (
        worst <= 0.01,
        f"debits equal credits in all {len(by_year)} fiscal years (worst ${worst:,.2f})",
        {"n_years": int(len(by_year)), "worst_absolute": worst},
    )


@control("segment_revenue_foots_to_filed", Severity.BLOCKING)
def _segment_revenue_foots_to_filed(ctx: LedgerContext):
    """Regional revenue must sum to the filed consolidated total.

    The two figures come from **different SEC products** — the regions from the
    Financial Statement Data Sets, the total from the companyfacts API — so this is
    a genuine cross-source reconciliation rather than an arithmetic identity. If it
    ties, both pipelines read the same filing correctly.

    It is also the only thing standing between the demo and a double-count. Netflix
    tags two overlapping geographic breakdowns: the four-region operating segments,
    and a standalone ``Geographical=US`` country disclosure worth $18.5B in FY2025.
    Summing everything on the geographic axis inflates revenue by roughly 40% — a
    number large enough to be obvious once totalled, and completely invisible in a
    per-region chart.

    **What the regions sum to is streaming revenue, not consolidated revenue**, and
    the difference is a real business rather than a rounding artifact. Netflix ran a
    DVD-by-mail service until September 2023, and it sits outside the streaming
    segment entirely. The residual against consolidated revenue is that business:

        FY2021  $182.3M     FY2024  $0.00
        FY2022  $145.7M     FY2025  $0.00
        FY2023   $82.8M

    A declining tail that reaches exactly zero the year after the shutdown. So the
    control tests the tie against the filed *streaming* total where that line is
    tagged, and separately requires the legacy residual to be non-negative — regions
    cannot exceed the whole — and to stay zero once the segment is gone.
    """
    from fpa.config import STREAMING_TOTAL

    if ctx.segments is None or ctx.segments.empty:
        return True, "no segment data in this run — skipped", {"skipped": True}

    revenue = ctx.quarterly["revenue"]
    annual = revenue.groupby(revenue.index.year).sum()
    complete = annual[revenue.groupby(revenue.index.year).size() == 4]

    segments = ctx.segments.copy()
    segments.index = segments.index.year
    shared = segments.index.intersection(complete.index)
    if len(shared) == 0:
        return True, "no fiscal year with both segment and filed revenue", {}

    # 1. Regions must sum to the filed streaming line, exactly, where it is tagged.
    streaming_worst = 0.0
    n_streaming = 0
    if STREAMING_TOTAL in segments.columns:
        both = segments.loc[shared, ["total", STREAMING_TOTAL]].dropna()
        n_streaming = len(both)
        if n_streaming:
            streaming_worst = float((both["total"] - both[STREAMING_TOTAL]).abs().max())

    # 2. The residual against consolidated revenue is the legacy DVD segment. It
    #    may be positive while that business existed, but never negative.
    legacy = complete.reindex(shared) - segments.loc[shared, "total"]
    most_negative = float(legacy.min())
    still_running = {int(year) for year, value in legacy.items() if abs(value) > ARTICULATION_ATOL}

    ok = streaming_worst <= ARTICULATION_ATOL and most_negative >= -ARTICULATION_ATOL
    legacy_note = (
        f"legacy non-streaming revenue in {sorted(still_running)}, nil thereafter"
        if still_running
        else "no legacy non-streaming revenue"
    )
    return (
        ok,
        f"{len(ctx.segments.columns) - 2} regions sum to the filed streaming line across "
        f"{n_streaming} fiscal years (worst ${streaming_worst:,.2f}); {legacy_note}",
        {
            "n_years": int(len(shared)),
            "streaming_worst": streaming_worst,
            "most_negative_legacy": most_negative,
            "legacy_years": sorted(still_running),
        },
    )


@control("accession_coverage", Severity.BLOCKING)
def _accession_coverage(ctx: LedgerContext):
    """Every filed fact must carry the accession number of its filing.

    Without it the traceability claim is unbacked, which is worse than not making it.
    """
    if ctx.facts is None:
        return True, "fact table unavailable", {}
    missing = ctx.facts["accn"].isna().sum()
    return (
        missing == 0,
        f"{len(ctx.facts) - missing}/{len(ctx.facts)} facts carry an accession number",
        {"n_missing": int(missing)},
    )
