"""SEC EDGAR XBRL ingest — filed actuals, each carrying its filing's accession number.

This is the layer that makes the whole demo auditable: every actual displayed
downstream can be traced to the specific 10-Q or 10-K it was tagged in. Nothing
here invents a number.

Three real-world XBRL problems are handled explicitly, because each one silently
corrupts a naive pull:

1. **Q4 is never a 10-Q.** Companies file Q1–Q3 on 10-Qs and roll Q4 into the
   10-K as a full year. Pulling "quarterly duration" facts therefore yields three
   quarters a year and a hole every December. Q4 is derived as FY − (Q1+Q2+Q3).

2. **Year-to-date facts share the tag with quarterly ones.** A 10-Q tags both the
   three-month and the six/nine-month figure. Filtering on duration is mandatory
   or Q2 gets double-counted.

3. **Restatements duplicate periods.** The same (tag, period) is re-reported in
   later filings. We keep the most recently *filed* value and count how many
   versions existed — a period with several is a restatement, which is a signal
   a finance team wants surfaced, not silently collapsed.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import httpx
import pandas as pd

from fpa.config import (
    DERIVED_MARKETING_FROM,
    EDGAR_TAGS,
    EXPENSE_ACCOUNTS,
    NON_ADDITIVE_ACCOUNTS,
    Settings,
)
from shared.cache import cached_parquet

logger = logging.getLogger(__name__)

COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Duration windows, in days, used to classify a duration fact.
_QUARTER_DAYS = (80, 100)
_ANNUAL_DAYS = (350, 380)


def _fetch_company_facts(settings: Settings) -> dict:
    """Download the full companyfacts document for the configured CIK."""
    url = COMPANY_FACTS_URL.format(cik=settings.cik)
    # SEC blocks or throttles generic agents; a descriptive one is required.
    headers = {"User-Agent": settings.edgar_user_agent, "Accept-Encoding": "gzip, deflate"}
    logger.info("fetching %s", url)
    response = httpx.get(url, headers=headers, timeout=60.0, follow_redirects=True)
    response.raise_for_status()
    return response.json()


def _tidy_facts(document: dict, settings: Settings) -> pd.DataFrame:
    """Flatten the companyfacts JSON into one tidy row per filed fact."""
    gaap = document.get("facts", {}).get("us-gaap", {})
    rows: list[dict] = []

    for account, spec in EDGAR_TAGS.items():
        # Unit comes from the tag spec, never assumed. ``companyfacts`` keys facts
        # by unit, so reading a per-share concept out of ``units["USD"]`` returns
        # an empty list rather than raising — the tag would simply disappear.
        entries = gaap.get(spec.tag, {}).get("units", {}).get(spec.unit, [])
        if not entries:
            logger.warning("tag %s (%s) has no %s facts", spec.tag, account, spec.unit)
            continue

        for entry in entries:
            end = date.fromisoformat(entry["end"])
            start_raw = entry.get("start")
            if start_raw is None:
                # Instant fact (balance-sheet item): a point-in-time balance.
                period_type, span = "instant", None
            else:
                start = date.fromisoformat(start_raw)
                span = (end - start).days
                if _QUARTER_DAYS[0] <= span <= _QUARTER_DAYS[1]:
                    period_type = "quarter"
                elif _ANNUAL_DAYS[0] <= span <= _ANNUAL_DAYS[1]:
                    period_type = "annual"
                else:
                    # Year-to-date (6- or 9-month) figures share the tag with the
                    # quarterly ones. Keeping them would double-count Q2 and Q3.
                    continue

            rows.append(
                {
                    "account": account,
                    "tag": spec.tag,
                    "unit": spec.unit,
                    "start": pd.Timestamp(start_raw) if start_raw else pd.NaT,
                    "end": pd.Timestamp(entry["end"]),
                    "value": float(entry["val"]),
                    "period_type": period_type,
                    "span_days": span,
                    "form": entry.get("form"),
                    "fy": entry.get("fy"),
                    "fp": entry.get("fp"),
                    # The audit trail. This column is why the demo can claim
                    # traceability rather than assert it.
                    "accn": entry.get("accn"),
                    "filed": pd.Timestamp(entry["filed"]) if entry.get("filed") else pd.NaT,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No facts found for CIK {settings.cik}")
    return frame.sort_values(["account", "end", "filed"]).reset_index(drop=True)


def _dedupe_restatements(facts: pd.DataFrame) -> pd.DataFrame:
    """Collapse re-reported periods to the latest filed value.

    Adds two columns, and the distinction between them matters:

    * ``n_filings`` — how many filings reported this period. Almost always >1 and
      almost always uninteresting: every 10-K repeats the prior year's quarters as
      comparatives. Counting these as "restatements" flags ~75% of the fact table
      and trains everyone to ignore the control.
    * ``n_distinct_values`` — how many *different numbers* were reported for it.
      Above 1 means the figure actually moved between filings, which is a genuine
      restatement and the only version worth a human's attention.
    """
    keys = ["account", "period_type", "start", "end"]
    grouped = facts.groupby(keys, dropna=False)["value"]
    stats = pd.DataFrame(
        {"n_filings": grouped.size(), "n_distinct_values": grouped.nunique()}
    ).reset_index()

    latest = (
        facts.sort_values("filed")
        .groupby(keys, dropna=False, as_index=False)
        .last()
        .merge(stats, on=keys, how="left")
    )
    return latest.sort_values(["account", "end"]).reset_index(drop=True)


def load_facts(settings: Settings, *, refresh: bool = False) -> pd.DataFrame:
    """Return the tidy, deduped fact table, reading a pinned parquet snapshot.

    The snapshot is what makes the demo offline-safe: EDGAR is hit once per
    vintage, not once per run.
    """
    path = settings.vintage_path(f"edgar_facts_{settings.ticker.lower()}")

    def fetch() -> pd.DataFrame:
        document = _fetch_company_facts(settings)
        return _dedupe_restatements(_tidy_facts(document, settings))

    return cached_parquet(path, fetch, refresh=refresh)


def _pivot(facts: pd.DataFrame, period_type: str) -> pd.DataFrame:
    """Pivot tidy facts to one row per period end, one column per account."""
    subset = facts[facts["period_type"] == period_type]
    return subset.pivot_table(index="end", columns="account", values="value", aggfunc="last")


def _annual_identity_status(annual_row: pd.Series, *, rtol: float = 1e-9) -> str:
    """Classify whether a fiscal year's annual income statement can be trusted.

    Q4 is never filed as a quarter, so it is derived from the 10-K annual. If the
    annual does not articulate

        Revenue - (the four expense lines) - OperatingIncome == 0

    then the entire discrepancy lands in that one derived quarter. So the year is
    classified before anything is derived from it:

    * ``"verified"`` — every line present and the identity ties. Derive freely.
    * ``"broken"`` — every line present and it does **not** tie. Netflix's FY2017
      (-$27.3M), FY2018 (+$3.8M) and FY2019 (-$127.9M) land here. Derive nothing.
    * ``"marketing_implied"`` — only Marketing is missing, because the tag was
      discontinued after 2024-09-30 (FY2024, FY2025). The identity cannot be
      tested, but the other five lines are filed and independent, so they are
      derived normally and Marketing follows from the quarterly identity in
      :func:`_derive_marketing`. That mechanism is not taken on trust either —
      ``fpa.controls.marketing_identity`` re-proves it against every quarter where
      the filed tag still exists.
    * ``"incomplete"`` — something else is missing. Derive nothing.
    """
    core = ["revenue", "cost_of_revenue", "research_development", "general_administrative", "operating_income"]
    if any(c not in annual_row.index or pd.isna(annual_row[c]) for c in core):
        return "incomplete"

    if "marketing" not in annual_row.index or pd.isna(annual_row["marketing"]):
        return "marketing_implied"

    residual = (
        annual_row["revenue"]
        - sum(annual_row[a] for a in EXPENSE_ACCOUNTS)
        - annual_row["operating_income"]
    )
    return "verified" if abs(residual) <= rtol * abs(annual_row["revenue"]) else "broken"


def _derive_q4(quarterly: pd.DataFrame, annual: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fill each fiscal Q4 as FY minus the three quarters filed on 10-Qs.

    Netflix's fiscal year is the calendar year, so Q4 ends 31 December and the
    annual fact covers the same year.

    Two guards, both learned from real breaks in this filer's data:

    * **Per account, not all-or-nothing.** ``Revenues`` is filed as a Q4 quarter
      for 2018-2020 while every other line is not. Re-deriving an account that was
      already filed replaces a filed number with an arithmetic one — and worse,
      mixes filing vintages, because the annual fact may come from a much later
      10-K than the quarters it is differenced against.
    * **Only from years that articulate.** See :func:`_annual_identity_holds`.

    Returns the filled frame plus a provenance dict for the audit trail.
    """
    shared = [c for c in annual.columns if c in quarterly.columns]
    if not shared:
        return quarterly, {}

    filled = quarterly.copy()
    provenance: dict[int, dict] = {}

    for annual_end, annual_row in annual.iterrows():
        year = int(annual_end.year)
        q4_end = pd.Timestamp(year=year, month=12, day=31)

        in_year = filled.index[(filled.index.year == year) & (filled.index != q4_end)]
        if len(in_year) != 3:
            # Incomplete year (e.g. the current one). Controls report the gap.
            provenance[year] = {"derived": [], "reason": "incomplete year"}
            continue

        status = _annual_identity_status(annual_row)
        if status in ("broken", "incomplete"):
            provenance[year] = {"derived": [], "reason": f"annual identity {status}"}
            continue

        derived: list[str] = []
        incomplete: list[str] = []
        for account in shared:
            if account in NON_ADDITIVE_ACCOUNTS:
                # A weighted-average share count and a per-share ratio do not
                # difference. Leave Q4 empty rather than fabricate it.
                continue
            if status == "marketing_implied" and account == "marketing":
                # Not filed annually either; comes from the quarterly identity.
                continue
            already_filed = q4_end in filled.index and pd.notna(
                filled.loc[q4_end, account]
            )
            if already_filed:
                continue  # keep the filed quarter; never overwrite it

            # skipna=False, and this is the whole ballgame. Q4 = FY − (Q1+Q2+Q3)
            # is only true if all three quarters are actually there. Pandas' sum
            # skips NaN, so an account filed for one quarter out of three yields
            # FY − Q3 — a number with two missing quarters' worth of the year
            # buried in it, and nothing to distinguish it from a real one.
            #
            # This is not hypothetical: pre-tax income is tagged as a quarter only
            # from 2020-Q3, so 2020-Q4 came out $1.83B too high and the
            # ``net_income_bridge`` control is what surfaced it.
            prior = filled.loc[in_year, account].sum(skipna=False)
            if pd.isna(prior):
                incomplete.append(account)
                continue
            filled.loc[q4_end, account] = annual_row[account] - prior
            derived.append(account)

        provenance[year] = {"derived": derived, "reason": status}
        if incomplete:
            provenance[year]["incomplete_quarters"] = incomplete

    return filled.sort_index(), provenance


def _derive_marketing(quarterly: pd.DataFrame) -> pd.DataFrame:
    """Recover Marketing where the tag was discontinued, via the opex identity.

        Marketing = Revenue − CostOfRevenue − R&D − G&A − OperatingIncome

    Exact, not approximate: it reproduces the filed value to the dollar for every
    period from 2018-Q1 on. ``fpa.controls.check_marketing_identity`` re-verifies
    this against the periods where the filed tag still exists.
    """
    needed = [c for c in DERIVED_MARKETING_FROM if c in quarterly.columns]
    if len(needed) != len(DERIVED_MARKETING_FROM):
        return quarterly

    revenue, cost, rnd, ga, ebit = (quarterly[c] for c in DERIVED_MARKETING_FROM)
    derived = revenue - cost - rnd - ga - ebit

    out = quarterly.copy()
    if "marketing" not in out.columns:
        out["marketing"] = pd.NA
    out["marketing_filed"] = out["marketing"]
    out["marketing_derived"] = derived
    # Prefer the filed figure where it exists; fall back to the identity.
    out["marketing"] = out["marketing"].fillna(derived)
    return out


def _derive_pretax(quarterly: pd.DataFrame, filed: pd.DataFrame) -> pd.DataFrame:
    """Fill pre-tax income from the tax bridge where the tag was not filed.

        Pre-tax income = Net income + Income tax expense

    Netflix only began tagging ``IncomeLossFromContinuingOperationsBefore...`` as a
    *quarter* in 2020-Q3, so eight quarters in the window need filling.

    Two derivations are available and they are not equally good:

    * **The tax bridge** — an identity within one filing's own quarterly facts.
    * **Annual differencing** (``FY − Q1..Q3``) — mixes filing vintages, because the
      10-K's annual figure may be restated relative to the 10-Qs it is differenced
      against. Residuals of $100–400K show up on $10B+ quarters from exactly that.

    So the identity is preferred wherever both are possible, and differencing is the
    last resort. ``filed`` is the raw quarterly pivot from *before* any Q4
    derivation, which is what makes ``pretax_filed`` mean genuinely filed — the
    comparison ``fpa.controls.net_income_bridge`` needs in order to be a test rather
    than a tautology.
    """
    if not {"net_income", "income_tax"} <= set(quarterly.columns):
        return quarterly

    derived = quarterly["net_income"] + quarterly["income_tax"]
    out = quarterly.copy()
    if "pretax_income" not in out.columns:
        out["pretax_income"] = pd.NA

    filed_pretax = (
        filed["pretax_income"].reindex(out.index)
        if "pretax_income" in filed.columns
        else pd.Series(pd.NA, index=out.index, dtype="float64")
    )
    out["pretax_filed"] = filed_pretax
    out["pretax_derived"] = derived
    # Precedence: filed, then the identity, then whatever differencing produced.
    out["pretax_income"] = filed_pretax.fillna(derived).fillna(out["pretax_income"])
    return out


def quarterly_actuals(settings: Settings, *, refresh: bool = False) -> pd.DataFrame:
    """Return filed quarterly actuals: one row per fiscal quarter, accounts as columns.

    Q4 is derived from the 10-K, Marketing is derived past its tag's
    discontinuation, and the series is trimmed to the window where the
    income-statement presentation is stable (see ``Settings.window_start``).
    """
    facts = load_facts(settings, refresh=refresh)

    quarterly = _pivot(facts, "quarter")
    annual = _pivot(facts, "annual")
    instant = _pivot(facts, "instant")

    # Kept before any derivation runs: "filed" has to mean filed for the
    # provenance badges and the identity controls to mean anything.
    filed_quarters = quarterly.copy()

    quarterly, q4_provenance = _derive_q4(quarterly, annual)
    quarterly = _derive_marketing(quarterly)
    quarterly = _derive_pretax(quarterly, filed_quarters)

    # Balance-sheet instants attach to the quarter that ends on the same date.
    if not instant.empty:
        quarterly = quarterly.join(instant, how="left", rsuffix="_instant")

    quarterly = quarterly[quarterly.index >= pd.Timestamp(settings.window_start)]
    quarterly = quarterly.sort_index()
    # Set attrs last: pandas does not reliably carry them through join/filter.
    quarterly.attrs["q4_provenance"] = q4_provenance
    return quarterly


def accession_index(settings: Settings, *, refresh: bool = False) -> pd.DataFrame:
    """Map each (account, period) to the filing it came from — the on-screen audit trail."""
    facts = load_facts(settings, refresh=refresh)
    columns = [
        "account",
        "end",
        "period_type",
        "unit",
        "form",
        "accn",
        "filed",
        "n_filings",
        "n_distinct_values",
    ]
    return facts[columns].sort_values(["end", "account"]).reset_index(drop=True)
