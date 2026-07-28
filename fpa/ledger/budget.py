"""Build the annual plan (budget) the actuals are measured against.

A variance analysis is only interesting if the plan was wrong in the way real
plans are wrong. So the budget here is not actuals-plus-noise: it is built the way
an FP&A team builds one — **set once, before the year starts, from last year's
actuals plus a planned growth rate** — and it carries deliberate, named planning
biases that the variance bridge then has to explain.

Because the plan is locked at the start of each fiscal year and never revised, it
diverges from actuals exactly as a real annual plan does: small early, compounding
through the year, and worst where the planning assumption was wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpa.config import Settings

# Planned year-over-year growth applied when the plan is set, by function.
# These are the *assumptions*, and some of them are wrong on purpose — that is
# the point of a variance analysis.
PLANNED_GROWTH: dict[str, float] = {
    "Content": 0.09,
    "Technology & Product": 0.11,
    "Marketing": 0.06,
    "G&A": 0.05,
}

# Systematic planning bias by sub-center, applied on top of the growth rate.
# Negative = the plan under-provisions, so actuals come in over budget.
#
# Cloud Infrastructure is deliberately the worst offender: chronically
# under-budgeted consumption-based spend is the single most common finding in
# real technology-cost management, and it gives the FinOps hooks something true
# to point at.
PLANNING_BIAS: dict[str, float] = {
    "Cloud Infrastructure": -0.14,
    "CDN & Delivery": -0.06,
    "Platform Engineering": 0.02,
    "Licensed Content": 0.03,
    "Original Productions": -0.04,
    "Brand & Media": 0.05,
    "Performance Marketing": -0.03,
    "Corporate Functions": 0.01,
    "Facilities": 0.06,
}

# Planned revenue growth. Set optimistically, as revenue plans usually are.
PLANNED_REVENUE_GROWTH: float = 0.14

# Noise on the monthly phasing of the plan. Plans are phased on a seasonal
# profile, not spread evenly, and the profile is never exactly right.
_PHASING_NOISE = 0.03


def _fiscal_year(period: pd.Series) -> pd.Series:
    return period.dt.year


def build_budget(
    settings: Settings, ledger: pd.DataFrame, *, trim_to_actuals: bool = True
) -> pd.DataFrame:
    """Return the cost-center budget at the same grain as ``ledger``.

    The plan for fiscal year Y is built from year Y−1 actuals, so the first year
    in the window has no budget — a real constraint, not a bug. Rows for that year
    are simply absent, and ``fpa.controls`` reports the coverage gap rather than
    silently treating a missing budget as zero.

    ``trim_to_actuals`` distinguishes the two things a plan is used for, which want
    opposite behaviour:

    * **Variance reporting** (default, ``True``) — drop plan months with no actual
      to compare against. Keeping them would report a month that has not happened
      as 100% under budget, which is worse than reporting nothing.
    * **Publishing to the ERP** (``False``) — keep the whole fiscal year. A plan is
      committed in advance for twelve months; truncating it at today's date would
      leave nothing for the rolling forecast to be compared *against*, which is the
      entire reason both versions are loaded.
    """
    rng = np.random.default_rng(settings.seed + 10)

    work = ledger.copy()
    work["year"] = _fiscal_year(work["period"])
    work["month"] = work["period"].dt.month

    years = sorted(work["year"].unique())
    rows: list[pd.DataFrame] = []

    for year in years[1:]:
        prior = work[work["year"] == year - 1]
        if prior.empty:
            continue

        # Prior-year actuals by month x cost center x account = the planning base.
        base = prior.groupby(
            ["month", "account", "function", "sub_center"], as_index=False
        )["amount"].sum()

        growth = base["function"].map(PLANNED_GROWTH).fillna(0.07)
        bias = base["sub_center"].map(PLANNING_BIAS).fillna(0.0)
        phasing = 1.0 + rng.normal(0.0, _PHASING_NOISE, len(base))

        planned = base["amount"] * (1.0 + growth) * (1.0 + bias) * phasing

        year_rows = base[["month", "account", "function", "sub_center"]].copy()
        year_rows["year"] = year
        year_rows["budget"] = planned.to_numpy()
        rows.append(year_rows)

    if not rows:
        return pd.DataFrame(
            columns=["period", "account", "function", "sub_center", "budget"]
        )

    budget = pd.concat(rows, ignore_index=True)
    budget["period"] = pd.to_datetime(
        dict(year=budget["year"], month=budget["month"], day=1)
    ) + pd.offsets.MonthEnd(0)

    if trim_to_actuals:
        # A plan for a month that has not happened yet belongs to the forecast,
        # not the variance report.
        actual_periods = set(ledger["period"].unique())
        budget = budget[budget["period"].isin(actual_periods)]

    return budget[
        ["period", "account", "function", "sub_center", "budget"]
    ].sort_values(["period", "account", "function", "sub_center"]).reset_index(drop=True)


def build_revenue_budget(settings: Settings, revenue: pd.DataFrame) -> pd.DataFrame:
    """Return the revenue plan, built the same way: prior-year actual x planned growth."""
    rng = np.random.default_rng(settings.seed + 11)

    work = revenue.copy()
    work["year"] = _fiscal_year(work["period"])
    work["month"] = work["period"].dt.month

    years = sorted(work["year"].unique())
    rows: list[pd.DataFrame] = []
    for year in years[1:]:
        prior = work[work["year"] == year - 1]
        if prior.empty:
            continue
        base = prior.groupby("month", as_index=False)["amount"].sum()
        phasing = 1.0 + rng.normal(0.0, _PHASING_NOISE, len(base))
        year_rows = base[["month"]].copy()
        year_rows["year"] = year
        year_rows["budget"] = (base["amount"] * (1.0 + PLANNED_REVENUE_GROWTH) * phasing).to_numpy()
        rows.append(year_rows)

    if not rows:
        return pd.DataFrame(columns=["period", "budget"])

    budget = pd.concat(rows, ignore_index=True)
    budget["period"] = pd.to_datetime(
        dict(year=budget["year"], month=budget["month"], day=1)
    ) + pd.offsets.MonthEnd(0)
    budget = budget[budget["period"].isin(set(revenue["period"].unique()))]
    return budget[["period", "budget"]].sort_values("period").reset_index(drop=True)
