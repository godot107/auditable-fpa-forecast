"""Centralized settings, the chart of accounts, and the cost-center hierarchy.

Read config in one place so the CLI, the Streamlit app, and the tests all get
identical behavior from the same environment variables — the pattern used across
this workspace (see ``financial-forecasting-engine/fce/config.py``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - optional dependency
    pass


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


# Repo root = parent of the ``fpa`` package directory.
_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Chart of accounts — the filed line items we build the ledger from.
# ---------------------------------------------------------------------------
class Tag(NamedTuple):
    """A us-gaap concept and the XBRL unit its facts are reported in.

    The unit is not decoration. ``companyfacts`` keys every fact by unit, and a
    per-share concept (``USD/shares``) or a share count (``shares``) simply is not
    present under ``USD``. Hardcoding ``USD`` therefore does not raise — it makes
    the tag silently vanish, which is the failure mode this codebase keeps finding
    and the reason the unit is declared here rather than assumed downstream.
    """

    tag: str
    unit: str = "USD"


# Each entry maps an internal account code to the us-gaap tag that carries it.
# Every tag below was verified present for NFLX (CIK 0001065280) with facts
# through 2026-06-30, unless annotated otherwise.
#
# Tags deliberately absent from this map, and why:
#   * ``GrossProfit`` — Netflix stopped tagging it after 2020-12-31. Derived as
#     Revenue − CostOfRevenue instead, which is its definition anyway.
#   * ``SellingGeneralAndAdministrativeExpense`` — never filed by Netflix; the
#     company splits Marketing and G&A into separate income-statement lines.
#   * ``Goodwill``, ``AccountsReceivableNetCurrent``, ``LongTermDebtCurrent``,
#     ``AdditionalPaidInCapital`` — never filed by this registrant.
#   * ``DeferredRevenueCurrent`` — superseded by the ASC 606 tag
#     ``ContractWithCustomerLiabilityCurrent`` after 2018.
#   * Content assets — Netflix reports them under a company extension tag, and
#     ``companyfacts`` exposes only us-gaap/dei/srt/ecd/ffd. Derived instead; see
#     ``DERIVED_CONTENT_ASSETS``.
EDGAR_TAGS: dict[str, Tag] = {
    # --- Income statement -------------------------------------------------
    "revenue": Tag("Revenues"),
    "cost_of_revenue": Tag("CostOfRevenue"),
    "research_development": Tag("ResearchAndDevelopmentExpense"),
    "general_administrative": Tag("GeneralAndAdministrativeExpense"),
    "marketing": Tag("MarketingExpense"),
    "operating_income": Tag("OperatingIncomeLoss"),
    # Filed as a quarter only from 2020-Q3 on; derived as NI + tax before that.
    "pretax_income": Tag(
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItems"
        "NoncontrollingInterest"
    ),
    "income_tax": Tag("IncomeTaxExpenseBenefit"),
    "net_income": Tag("NetIncomeLoss"),
    # Discontinued after 2024-09-30, like MarketingExpense. Kept because the
    # overlap is what ``non_operating_bridge`` tests against.
    "interest_expense": Tag("InterestExpense"),
    # --- Per share (non-USD units — the reason Tag carries one) -----------
    "eps_basic": Tag("EarningsPerShareBasic", "USD/shares"),
    "eps_diluted": Tag("EarningsPerShareDiluted", "USD/shares"),
    "shares_basic": Tag("WeightedAverageNumberOfSharesOutstandingBasic", "shares"),
    "shares_diluted": Tag("WeightedAverageNumberOfDilutedSharesOutstanding", "shares"),
    # --- Cash flow --------------------------------------------------------
    "cash_from_operations": Tag("NetCashProvidedByUsedInOperatingActivities"),
    "cash_from_investing": Tag("NetCashProvidedByUsedInInvestingActivities"),
    "cash_from_financing": Tag("NetCashProvidedByUsedInFinancingActivities"),
    "fx_effect_on_cash": Tag(
        "EffectOfExchangeRateOnCashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
    ),
    "depreciation_amortization": Tag("DepreciationDepletionAndAmortization"),
    "stock_compensation": Tag("ShareBasedCompensation"),
    "capex": Tag("PaymentsToAcquirePropertyPlantAndEquipment"),
    "buybacks": Tag("PaymentsForRepurchaseOfCommonStock"),
    # --- Balance sheet: totals -------------------------------------------
    "assets": Tag("Assets"),
    "assets_current": Tag("AssetsCurrent"),
    "liabilities": Tag("Liabilities"),
    "liabilities_current": Tag("LiabilitiesCurrent"),
    "equity": Tag("StockholdersEquity"),
    # --- Balance sheet: detail -------------------------------------------
    "cash": Tag("CashAndCashEquivalentsAtCarryingValue"),
    # The cash-flow statement reconciles to cash *including restricted* cash,
    # which is a different concept from the balance-sheet "cash" line above.
    # Using the wrong one breaks the roll-forward by the restricted balance.
    "cash_including_restricted": Tag(
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
    ),
    "short_term_investments": Tag("ShortTermInvestments"),
    "ppe_net": Tag("PropertyPlantAndEquipmentNet"),
    "rou_asset": Tag("OperatingLeaseRightOfUseAsset"),
    "other_assets_noncurrent": Tag("OtherAssetsNoncurrent"),
    "accounts_payable": Tag("AccountsPayableCurrent"),
    "accrued_liabilities": Tag("AccruedLiabilitiesCurrent"),
    "deferred_revenue": Tag("ContractWithCustomerLiabilityCurrent"),
    "lease_liability_current": Tag("OperatingLeaseLiabilityCurrent"),
    "long_term_debt": Tag("LongTermDebtNoncurrent"),
    "lease_liability_noncurrent": Tag("OperatingLeaseLiabilityNoncurrent"),
    "other_liabilities_noncurrent": Tag("OtherLiabilitiesNoncurrent"),
    "common_stock": Tag("CommonStockValue"),
    "retained_earnings": Tag("RetainedEarningsAccumulatedDeficit"),
    "aoci": Tag("AccumulatedOtherComprehensiveIncomeLossNetOfTax"),
}

# Expense accounts that make up operating expense, in income-statement order.
# These are the lines allocated down to cost centers.
EXPENSE_ACCOUNTS: tuple[str, ...] = (
    "cost_of_revenue",
    "research_development",
    "marketing",
    "general_administrative",
)

# ``MarketingExpense`` was last tagged for the period ending 2024-09-30. After
# that Netflix stopped filing it, so it is recovered from the income-statement
# identity:
#
#     Marketing = Revenue − CostOfRevenue − R&D − G&A − OperatingIncome
#
# This is not an approximation. Verified against every period where both the
# filed tag and all five inputs exist: the derivation reproduces the filed value
# **to the dollar** for every quarter from 2018-Q1 onward. It breaks by $17–25M
# in 2016–2017 (a presentation change) and by ~$1–2M pre-2010, which is why the
# usable window starts in 2018. ``fpa.controls`` re-asserts this every run rather
# than trusting the comment.
DERIVED_MARKETING_FROM: tuple[str, ...] = (
    "revenue",
    "cost_of_revenue",
    "research_development",
    "general_administrative",
    "operating_income",
)


# ---------------------------------------------------------------------------
# The balance sheet — a non-overlapping partition of Assets and of Liabilities
# and Equity, so the two sides can be posted as a self-balancing journal entry.
# ---------------------------------------------------------------------------
# ``Assets = Liabilities + StockholdersEquity`` holds to **$0.00** across all 26
# quarters in the window (verified, not assumed — see ``balance_sheet_balances``).
# That single fact is what lets the ERP loader post the balance sheet with no
# clearing account: debits and credits net to zero because the filed statement
# says they do.
#
# Getting there needs the detail lines to partition each total exactly, and three
# of Netflix's largest lines are not obtainable as tags:
#
#   * **Content assets** — reported under a company extension tag, which
#     ``companyfacts`` does not expose.
#   * **Treasury stock** — ``TreasuryStockValue`` stops at 2022-03-31 while the
#     buyback programme continues; by 2026-06-30 the unreported balance is $28.4B.
#   * **Other current assets / liabilities** — an aggregation Netflix does not tag.
#
# Each is therefore derived as the residual of a filed subtotal against filed
# detail — the same pattern as the marketing derivation, and subject to the same
# discipline: the residual is *named* as the line it represents, its sign is
# checked every run, and where an overlap with a filed tag exists the derivation
# is proved against it rather than trusted.
class Derived(NamedTuple):
    """A balance-sheet line recovered as ``total − sum(parts)``.

    ``sign`` is the residual's expected direction: +1 for a normal balance, −1 for
    a contra account. It is not documentation — ``derived_balance_sheet_lines``
    blocks the pipeline on it, which is how the first partition's −$571M
    "liability" was caught instead of being posted.
    """

    total: str
    parts: tuple[str, ...]
    sign: int = 1


# Tags whose *absence* is itself information: the company had no such line, so the
# balance is zero. Netflix reported no short-term investments before 2021-Q4 and
# does not tag the concept in those quarters.
#
# This set exists to stop the assumption being made by accident. ``DataFrame.sum``
# skips NaN by default, so a missing part is silently treated as zero and the
# derived line comes out looking complete — 26 of 26 quarters populated, 10 of them
# resting on an assumption nobody declared. Every other part is required, and a
# derivation missing one yields NaN rather than a number.
ABSENT_MEANS_ZERO: frozenset[str] = frozenset({"short_term_investments"})


# A subtlety that a naive partition gets wrong, silently and in the wrong
# direction: **not every filed tag is a line on the face of the statement.** The
# lease tags are ASC 842 *disclosures* nested inside the "other non-current"
# captions, not siblings of them. Including them double-counts, which showed up as
# a non-current liability residual of −$571M — a negative liability, and the only
# reason the error was visible at all.
#
# Settled empirically rather than by reading the filing, via the ASC 842 adoption
# step in 2019-Q1, when both tags appear for the first time:
#
#     OtherAssetsNoncurrent      stepped +$816M   ROU recognised   $812M
#     OtherLiabilitiesNoncurrent stepped +$663M   lease liability  $765M
#
# A caption that jumps by the amount of the thing being adopted contains it. So
# the lease tags are carried as disclosure columns and excluded from the posting
# partition. See ``fpa.controls.derived_balance_sheet_lines``, which re-proves
# every residual is positive on every run.
DERIVED_BALANCE_SHEET: dict[str, Derived] = {
    # Cash + short-term investments are filed; prepaid and other receivables are
    # the remainder.
    "other_current_assets": Derived(
        "assets_current", ("cash", "short_term_investments")
    ),
    # The residual that matters: $33.8B at 2026-06-30, and Netflix's single
    # largest asset. Magnitude is checked, not merely asserted.
    "content_assets": Derived(
        "assets", ("assets_current", "ppe_net", "other_assets_noncurrent")
    ),
    # Current content liabilities and short-term debt, neither tagged separately.
    "content_liabilities_current": Derived(
        "liabilities_current",
        ("accounts_payable", "accrued_liabilities", "deferred_revenue"),
    ),
    # Non-current content liabilities: $1.6B, which is the right order for this
    # filer and — unlike the first attempt — positive in every quarter.
    "content_liabilities_noncurrent": Derived(
        "liabilities",
        ("liabilities_current", "long_term_debt", "other_liabilities_noncurrent"),
    ),
    # Negative by nature: treasury stock is contra-equity.
    "treasury_stock": Derived(
        "equity", ("common_stock", "retained_earnings", "aoci"), sign=-1
    ),
}

# Filed, real, and useful — but *inside* the captions above, so they are reported
# and never posted. Keeping them visible is the point: a lease disclosure is what
# an IFRS-16 / ASC 842 leverage KPI is built from.
DISCLOSURE_ONLY: tuple[str, ...] = (
    "rou_asset",
    "lease_liability_current",
    "lease_liability_noncurrent",
    "cash_including_restricted",
)


# ---------------------------------------------------------------------------
# Reportable segments — a different API, because companyfacts has no dimensions.
# ---------------------------------------------------------------------------
# The four operating regions Netflix reports revenue for, mapped from the XBRL
# member name to the abbreviation the company itself uses in its shareholder
# letters. ``Geographical=US`` is deliberately absent: it is a standalone country
# disclosure that overlaps UCAN, and summing it with these would double-count.
REGION_LABELS: dict[str, str] = {
    "UnitedStatesAndCanada": "UCAN",
    "EMEA": "EMEA",
    "LatinAmerica": "LATAM",
    "AsiaPacific": "APAC",
}

SEGMENT_REVENUE_TAG = "Revenues"

# The consolidated streaming line — streaming revenue with no geographic axis.
# Captured alongside the regions because it is what they actually sum to, which is
# not the same as consolidated revenue: Netflix ran a DVD-by-mail business until
# September 2023, and its revenue sits outside the streaming segment entirely.
STREAMING_TOTAL = "Streaming"


class Line(NamedTuple):
    """One balance-sheet line: an ERP account code, a caption, and its side."""

    code: str
    name: str
    account: str  # internal account key, filed or derived
    account_type: str  # Odoo 18 ``account.account.account_type``


# Debit-balance side. Sums to filed ``assets`` exactly, by construction.
BALANCE_SHEET_ASSETS: tuple[Line, ...] = (
    Line("100000", "Cash and Cash Equivalents", "cash", "asset_cash"),
    Line("101000", "Short-term Investments", "short_term_investments", "asset_current"),
    Line("102000", "Other Current Assets", "other_current_assets", "asset_current"),
    Line("110000", "Content Assets, Net", "content_assets", "asset_non_current"),
    Line("120000", "Property and Equipment, Net", "ppe_net", "asset_fixed"),
    Line(
        "130000",
        "Other Non-Current Assets (incl. right-of-use)",
        "other_assets_noncurrent",
        "asset_non_current",
    ),
)

# Credit-balance side. Sums to filed ``liabilities`` + ``equity`` exactly.
BALANCE_SHEET_LIABILITIES: tuple[Line, ...] = (
    Line("200000", "Accounts Payable", "accounts_payable", "liability_payable"),
    Line("210000", "Accrued Expenses (incl. current leases)", "accrued_liabilities", "liability_current"),
    Line("220000", "Deferred Revenue", "deferred_revenue", "liability_current"),
    Line("230000", "Current Content Liabilities and Short-term Debt", "content_liabilities_current", "liability_current"),
    Line("240000", "Non-Current Content Liabilities", "content_liabilities_noncurrent", "liability_non_current"),
    Line("250000", "Long-term Debt", "long_term_debt", "liability_non_current"),
    Line(
        "260000",
        "Other Non-Current Liabilities (incl. leases)",
        "other_liabilities_noncurrent",
        "liability_non_current",
    ),
)

BALANCE_SHEET_EQUITY: tuple[Line, ...] = (
    Line("300000", "Common Stock and Paid-in Capital", "common_stock", "equity"),
    Line("310000", "Treasury Stock", "treasury_stock", "equity"),
    Line("320000", "Retained Earnings", "retained_earnings", "equity"),
    Line("330000", "Accumulated Other Comprehensive Income", "aoci", "equity"),
)


def balance_sheet_lines() -> tuple[tuple[Line, int], ...]:
    """Every balance-sheet line paired with its natural sign (+1 debit, −1 credit)."""
    return (
        *((line, 1) for line in BALANCE_SHEET_ASSETS),
        *((line, -1) for line in BALANCE_SHEET_LIABILITIES),
        *((line, -1) for line in BALANCE_SHEET_EQUITY),
    )


# The cash-flow statement is deliberately **not** posted to the ERP. No ledger
# journalizes a cash-flow statement: it is derived from the movement in balance
# sheet accounts and the income statement. It is built in Python
# (``fpa.ingest.statements.cash_flow``), reconciled against the filed roll-forward,
# and reported — which is exactly where it belongs.
# Accounts that must never be derived as ``annual − (Q1+Q2+Q3)``, because they do
# not add across periods. A weighted-average share count is not the sum of four
# quarterly ones, and EPS is a ratio — differencing either produces a number that
# looks plausible and means nothing. Flow accounts (revenue, cash flow) are
# additive; these are not.
NON_ADDITIVE_ACCOUNTS: frozenset[str] = frozenset(
    {"eps_basic", "eps_diluted", "shares_basic", "shares_diluted"}
)

CASH_FLOW_SECTIONS: tuple[str, ...] = (
    "cash_from_operations",
    "cash_from_investing",
    "cash_from_financing",
    "fx_effect_on_cash",
)


# ---------------------------------------------------------------------------
# Cost-center hierarchy — total → function → sub-center.
# ---------------------------------------------------------------------------
# Three levels, so the forecast is a genuine hierarchical time series and can be
# bottom-up reconciled: forecast the leaves, aggregate up, coherent by
# construction. The Technology & Product sub-centers are the ones an ITFM /
# FinOps platform actually reports on, which is where the (scaffolded) unit-cost
# KPIs in ``fpa.kpi.finops`` attach.
COST_CENTERS: dict[str, tuple[str, ...]] = {
    "Content": ("Licensed Content", "Original Productions"),
    "Technology & Product": (
        "Cloud Infrastructure",
        "CDN & Delivery",
        "Platform Engineering",
    ),
    "Marketing": ("Brand & Media", "Performance Marketing"),
    "G&A": ("Corporate Functions", "Facilities"),
}

# Which filed expense account each function draws from. A streaming company's
# cost of revenue is content amortization plus delivery infrastructure, so
# Content and Technology & Product both sit inside it; R&D is Netflix's
# "technology and development" line.
FUNCTION_ACCOUNTS: dict[str, tuple[str, ...]] = {
    "Content": ("cost_of_revenue",),
    "Technology & Product": ("cost_of_revenue", "research_development"),
    "Marketing": ("marketing",),
    "G&A": ("general_administrative",),
}


def leaf_cost_centers() -> tuple[tuple[str, str], ...]:
    """Return every ``(function, sub_center)`` leaf in the hierarchy."""
    return tuple(
        (function, sub) for function, subs in COST_CENTERS.items() for sub in subs
    )


@dataclass
class Settings:
    """Runtime configuration for one pipeline invocation."""

    # --- Company under analysis ------------------------------------------
    ticker: str = field(default_factory=lambda: _env("FPA_TICKER", "NFLX"))
    # Zero-padded 10-digit CIK, as the EDGAR XBRL API expects it.
    cik: str = field(default_factory=lambda: _env("FPA_CIK", "0001065280"))

    # --- Data / cache -----------------------------------------------------
    data_dir: Path = field(default_factory=lambda: _ROOT / "data")
    # Pinned data vintage: cached parquet snapshots are keyed by this tag, so the
    # demo reproduces offline regardless of "today" and never depends on EDGAR
    # being reachable while someone is watching.
    data_vintage: str = field(default_factory=lambda: _env("FPA_DATA_VINTAGE", "2026-07"))

    # SEC requires a descriptive User-Agent; generic ones get throttled/blocked.
    edgar_user_agent: str = field(
        default_factory=lambda: _env(
            "EDGAR_USER_AGENT", "auditable-fpa-forecast contact@example.com"
        )
    )

    # Series start — chosen by what the data supports, not by preference.
    #
    # The *quarterly* opex identity holds to the dollar from 2018-Q1. But Q4 is
    # never filed as a quarter, so it must be derived from the 10-K annual, and
    # the *annual* identity does not articulate for FY2017 (-$27.3M), FY2018
    # (+$3.8M) or FY2019 (-$127.9M) against this chart of accounts. It ties
    # exactly from FY2020 on. Deriving Q4 from a year that does not articulate
    # dumps the entire discrepancy into that one quarter, so those years are
    # excluded and ``fpa.controls.annual_identity_in_window`` re-proves the
    # boundary every run.
    window_start: str = "2020-01-01"

    # --- Ledger -----------------------------------------------------------
    seed: int = 42  # allocation model is seeded; the demo must reproduce exactly

    # --- Forecast ---------------------------------------------------------
    horizon_months: int = 12
    seasonal_period: int = 12
    # Rolling-origin backtest: gap between train and test equals the horizon, so
    # no fold can see inside its own forecast window.
    backtest_folds: int = 6

    # --- Narrative --------------------------------------------------------
    # claudecode = headless `claude -p` (no API key); fixture = offline replay;
    # anthropic = direct API.
    narrative_provider: str = field(
        default_factory=lambda: _env("FPA_NARRATIVE_PROVIDER", "claudecode")
    )
    narrative_model: str = field(
        default_factory=lambda: _env("FPA_NARRATIVE_MODEL", "sonnet")
    )
    audit_dir: Path = field(default_factory=lambda: _ROOT / "audit")
    reports_dir: Path = field(default_factory=lambda: _ROOT / "reports")

    # --- Segments (SEC Financial Statement Data Sets) ---------------------
    # Which quarterly archives to pull for the regional revenue series. Each is
    # ~85 MB, so this is a deliberate, declared cost rather than "fetch
    # everything". Ordered oldest-first: the dedupe keeps the last occurrence of a
    # period, which is the most recently filed restatement of it.
    segment_quarters: tuple[tuple[int, int], ...] = (
        (2024, 1),
        (2025, 1),
        (2025, 3),
        (2026, 1),
    )

    # --- Odoo (seeding only; the app reads the materialized extract) ------
    odoo_url: str = field(default_factory=lambda: _env("ODOO_URL", "http://localhost:8069"))
    odoo_db: str = field(default_factory=lambda: _env("ODOO_DB", "fpa_demo"))
    odoo_user: str = field(default_factory=lambda: _env("ODOO_USER", "admin"))
    odoo_password: str = field(default_factory=lambda: _env("ODOO_PASSWORD", "admin"))

    def vintage_path(self, name: str) -> Path:
        """Cache path for a named parquet snapshot at the pinned vintage."""
        return self.data_dir / f"{name}.{self.data_vintage}.parquet"


def get_settings() -> Settings:
    """Build a fresh :class:`Settings` from the current environment."""
    return Settings()
