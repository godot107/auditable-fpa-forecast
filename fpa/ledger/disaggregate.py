"""Disaggregate filed quarterly actuals into a monthly, cost-center-level ledger.

No public company discloses monthly figures or cost-center detail, so this layer
is **modeled** — and it says so. What makes it defensible rather than invented is
that the model is constrained: every allocation is forced to foot exactly back to
the filed quarterly figure it came from. The detail is a hypothesis about *shape*;
the totals remain the filed truth.

That constraint is what makes the forecast a **hierarchical time series**
(total → function → sub-center) that can be **bottom-up reconciled**: forecast the
leaves, aggregate up, and the total is coherent by construction. Bottom-up keeps
leaf-level signal and guarantees the cost-center forecasts sum to the number the
CFO sees; the cost is noisier leaf series than a top-down split would give.

Everything here is seeded. The same vintage always produces the same ledger.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpa.config import COST_CENTERS, EXPENSE_ACCOUNTS, Settings

# ---------------------------------------------------------------------------
# Allocation basis: which cost centers consume each filed expense account.
# ---------------------------------------------------------------------------
# Explicit rather than clever, because this table is the thing an auditor would
# ask to see. Weights are a starting basis; they drift slowly over time (below)
# and are renormalized per period so each account still foots exactly.
#
# The shape reflects how a streaming business actually spends: cost of revenue is
# content amortization plus delivery infrastructure; R&D ("technology and
# development" on Netflix's income statement) is mostly platform engineering
# headcount but funds cloud and delivery work too.
ALLOCATION_BASIS: dict[str, dict[tuple[str, str], float]] = {
    "cost_of_revenue": {
        ("Content", "Licensed Content"): 0.46,
        ("Content", "Original Productions"): 0.32,
        ("Technology & Product", "Cloud Infrastructure"): 0.09,
        ("Technology & Product", "CDN & Delivery"): 0.13,
    },
    "research_development": {
        ("Technology & Product", "Platform Engineering"): 0.70,
        ("Technology & Product", "Cloud Infrastructure"): 0.20,
        ("Technology & Product", "CDN & Delivery"): 0.10,
    },
    "marketing": {
        ("Marketing", "Brand & Media"): 0.58,
        ("Marketing", "Performance Marketing"): 0.42,
    },
    "general_administrative": {
        ("G&A", "Corporate Functions"): 0.74,
        ("G&A", "Facilities"): 0.26,
    },
}

# How far an allocation weight may drift over the full history, as a fraction of
# its base. Cost mixes shift (Netflix moved toward originals; cloud spend grew),
# and a perfectly static mix would make the variance analysis trivially boring.
_DRIFT = 0.25

# Noise on the intra-quarter split, as a fraction of an even third.
#
# Kept small on purpose. The plan is built from prior-year same-month actuals, so
# any noise here enters the budget variance *twice* — once via the current month
# and once via the base — and two independent draws at 10% relative would produce
# ~14% of pure artifact variance. That would swamp the 2-14% planning biases in
# ``budget.py``, and the variance bridge would end up explaining this module's
# random numbers instead of the planning story. The seasonal tilt below is
# deterministic and repeats each year, which is what makes a year-over-year
# comparison mean something.
_MONTH_NOISE = 0.03

# Revenue is phased separately and far more smoothly than expenses. Subscription
# revenue accrues roughly evenly — it does not spike at quarter-end the way
# expenses do — and phasing it with the expense tilt made implied ARPU swing ~11%
# month to month, which is an artifact, not a rate signal.
_REVENUE_NOISE = 0.01


def _month_ends(quarter_end: pd.Timestamp) -> list[pd.Timestamp]:
    """The three calendar month-ends making up the quarter ending at ``quarter_end``."""
    start = quarter_end - pd.offsets.QuarterBegin(startingMonth=1)
    return list(pd.date_range(start=start, end=quarter_end, freq="ME"))[-3:]


def _intra_quarter_weights(
    rng: np.random.Generator, n_quarters: int, *, noise: float = _MONTH_NOISE
) -> np.ndarray:
    """Weights splitting each quarter across its three months; each row sums to 1.

    Months inside a quarter are not equal for expenses — spend ramps, and the last
    month of a quarter carries period-end activity. The tilt is deterministic and
    identical every quarter, so the seasonal shape repeats year over year; only the
    small ``noise`` term varies. Pass a smaller ``noise`` for series that accrue
    smoothly (see ``monthly_revenue``).
    """
    tilt = np.array([0.32, 0.33, 0.35])
    weights = np.clip(tilt + rng.normal(0.0, noise / 3.0, size=(n_quarters, 3)), 0.15, 0.55)
    return weights / weights.sum(axis=1, keepdims=True)


def _drifting_weights(
    rng: np.random.Generator, base: dict[tuple[str, str], float], n_periods: int
) -> pd.DataFrame:
    """Allocation weights per period for one account; every row sums to exactly 1."""
    centers = list(base)
    base_vector = np.array([base[c] for c in centers])

    # A smooth, seeded drift path per cost center: random direction, monotone in
    # time, so the mix evolves instead of jittering.
    direction = rng.uniform(-1.0, 1.0, size=len(centers))
    ramp = np.linspace(0.0, 1.0, n_periods).reshape(-1, 1)
    drifted = base_vector * (1.0 + _DRIFT * direction * ramp)
    drifted = np.clip(drifted, 1e-6, None)

    weights = drifted / drifted.sum(axis=1, keepdims=True)
    return pd.DataFrame(weights, columns=pd.MultiIndex.from_tuples(centers))


def _exact_split(total: float, weights: np.ndarray) -> np.ndarray:
    """Split ``total`` by ``weights`` so the parts sum to ``total`` exactly.

    Float multiplication leaves a residual of order 1e-10 relative, which would
    make the footing control fail on rounding rather than on substance. The last
    bucket absorbs the remainder, so the sum is exact by construction.
    """
    parts = total * weights
    parts[-1] = total - parts[:-1].sum()
    return parts


def monthly_ledger(settings: Settings, quarterly: pd.DataFrame) -> pd.DataFrame:
    """Return the modeled monthly ledger at (period, function, sub_center, account) grain.

    Columns: ``period``, ``quarter_end``, ``account``, ``function``,
    ``sub_center``, ``amount``, ``basis``. ``quarter_end`` is retained so any
    ledger row can be traced back to the filed quarter — and thence to its
    accession number — that constrains it.
    """
    rng = np.random.default_rng(settings.seed)
    quarters = quarterly.index
    month_weights = _intra_quarter_weights(rng, len(quarters))

    # One drift path per account, shared across all its quarters.
    account_weights = {
        account: _drifting_weights(rng, ALLOCATION_BASIS[account], len(quarters))
        for account in EXPENSE_ACCOUNTS
        if account in ALLOCATION_BASIS
    }

    rows: list[dict] = []
    for q_idx, quarter_end in enumerate(quarters):
        months = _month_ends(quarter_end)
        for account, weight_frame in account_weights.items():
            quarter_total = quarterly.at[quarter_end, account]
            if pd.isna(quarter_total):
                continue

            # Split across months first, then across cost centers within a month.
            month_amounts = _exact_split(float(quarter_total), month_weights[q_idx])
            centers = list(weight_frame.columns)
            center_weights = weight_frame.iloc[q_idx].to_numpy()

            for month, month_amount in zip(months, month_amounts):
                center_amounts = _exact_split(month_amount, center_weights)
                for (function, sub_center), amount in zip(centers, center_amounts):
                    rows.append(
                        {
                            "period": month,
                            "quarter_end": quarter_end,
                            "account": account,
                            "function": function,
                            "sub_center": sub_center,
                            "amount": amount,
                            "basis": "MODELED",
                        }
                    )

    ledger = pd.DataFrame(rows)
    return ledger.sort_values(["period", "account", "function", "sub_center"]).reset_index(
        drop=True
    )


def monthly_revenue(settings: Settings, quarterly: pd.DataFrame) -> pd.DataFrame:
    """Split filed quarterly revenue into months, footing exactly to each quarter.

    Revenue is not allocated to cost centers — it is a company-level line here —
    but it does need the same monthly grain as the expense ledger.
    """
    rng = np.random.default_rng(settings.seed + 1)
    quarters = quarterly.index
    # Near-even split, built directly rather than by reusing the expense tilt:
    # subscription revenue accrues smoothly and does not spike at period end.
    # Phasing it on the expense profile made implied ARPU swing ~11% month to
    # month, which is a modeling artifact rather than a rate signal.
    weights = (1.0 / 3.0) * (1.0 + rng.normal(0.0, _REVENUE_NOISE, size=(len(quarters), 3)))
    weights = weights / weights.sum(axis=1, keepdims=True)

    rows: list[dict] = []
    for q_idx, quarter_end in enumerate(quarters):
        total = quarterly.at[quarter_end, "revenue"]
        if pd.isna(total):
            continue
        for month, amount in zip(_month_ends(quarter_end), _exact_split(float(total), weights[q_idx])):
            rows.append(
                {
                    "period": month,
                    "quarter_end": quarter_end,
                    "account": "revenue",
                    "amount": amount,
                    "basis": "MODELED",
                }
            )
    return pd.DataFrame(rows).sort_values("period").reset_index(drop=True)


def monthly_drivers(settings: Settings, quarterly: pd.DataFrame) -> pd.DataFrame:
    """Return the rate x volume driver decomposition behind monthly revenue.

    Netflix does not tag member counts in XBRL (verified: no membership or
    subscriber concept exists in its facts), and stopped disclosing them publicly
    from 2025. So **members is modeled** — a seeded growth-plus-seasonality curve —
    and **ARPU is then implied** as revenue / members.

    Only one of the two is invented. Their product reproduces the filed revenue
    exactly, which is what lets the variance bridge decompose a revenue movement
    into a rate effect and a volume effect without either being fabricated.
    """
    revenue = monthly_revenue(settings, quarterly)
    rng = np.random.default_rng(settings.seed + 2)

    n = len(revenue)
    t = np.arange(n)
    # Members: compounding growth that decelerates, plus mild seasonality and seeded
    # noise. Anchored to Netflix's ~183M paid memberships at the start of the window
    # (Q1 2020) and calibrated so the series lands near the ~300M last disclosed
    # before the company stopped reporting membership in 2025. Getting this anchor
    # right matters beyond realism: ARPU is *implied* as revenue / members, so a
    # wrong member level silently produces a wrong-looking rate.
    base = 183e6 * np.cumprod(1.0 + np.linspace(0.011, 0.003, n))
    seasonal = 1.0 + 0.012 * np.sin(2 * np.pi * (t % 12) / 12.0)
    members = base * seasonal * (1.0 + rng.normal(0.0, 0.004, n))

    out = revenue[["period", "quarter_end"]].copy()
    out["members"] = members
    out["revenue"] = revenue["amount"].to_numpy()
    # ARPU is implied, not modeled — this is the identity that keeps rate x volume
    # tied to the filed number.
    out["arpu"] = out["revenue"] / out["members"]
    out["basis_members"] = "MODELED"
    out["basis_arpu"] = "IMPLIED"
    return out
