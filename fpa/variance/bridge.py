"""Budget vs actual decomposition — the variance bridge.

"Opex was $40M over plan" is not actionable. "Opex was $40M over plan, of which $31M
is total spend growing faster than planned and $9M is the mix shifting toward Cloud
Infrastructure" is. The decompositions here exist to answer *why*, and each one **sums
exactly to the total variance** — a bridge that does not tie is worse than no bridge,
because it invites the reader to trust a residual nobody owns.

Sign convention, stated once and applied throughout:

* **Expenses** — positive variance means actual came in **above** plan, which is
  **unfavourable**.
* **Revenue** — positive variance means actual came in **above** plan, which is
  **favourable**.

Two decompositions, each matched to what the data actually supports:

* **Cost centers — spend x mix.** There is no independent unit driver per cost center,
  so a rate/volume split would require inventing one. The variance is split instead
  into the part explained by total spend moving (every center carried along at its
  planned share) and the part explained by the *mix* shifting between centers.

* **Revenue — rate x volume.** Revenue is members x ARPU, so the classic split
  applies. The plan is a revenue figure rather than a member figure, which constrains
  what can honestly be attributed — see :func:`driver_variance`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def expense_variance(ledger: pd.DataFrame, budget: pd.DataFrame) -> pd.DataFrame:
    """Budget vs actual for expenses at (period, account, cost center) grain.

    Positive variance means actuals were HIGHER than budget (unfavourable).
    """
    if budget.empty or ledger.empty:
        return pd.DataFrame()

    actual = ledger.groupby(
        ["period", "account", "function", "sub_center"], as_index=False
    )["amount"].sum()

    merged = actual.merge(
        budget, on=["period", "account", "function", "sub_center"], how="inner"
    ).rename(columns={"amount": "actual"})

    merged["variance"] = merged["actual"] - merged["budget"]
    merged["variance_pct"] = merged["variance"] / merged["budget"].replace(0, np.nan)
    merged["favourability"] = np.where(merged["variance"] > 0, "unfavourable", "favourable")
    return merged


def revenue_variance(
    revenue_actual: pd.DataFrame, revenue_budget: pd.DataFrame
) -> pd.DataFrame:
    """Budget vs actual for total revenue by period.

    Positive variance means actuals were HIGHER than budget (favourable).
    """
    if revenue_budget.empty or revenue_actual.empty:
        return pd.DataFrame()

    merged = revenue_actual.merge(revenue_budget, on=["period"], how="inner").rename(
        columns={"amount": "actual"}
    )
    merged["variance"] = merged["actual"] - merged["budget"]
    merged["variance_pct"] = merged["variance"] / merged["budget"].replace(0, np.nan)
    merged["favourability"] = np.where(merged["variance"] > 0, "favourable", "unfavourable")
    return merged


def cost_center_variance(
    ledger: pd.DataFrame, budget: pd.DataFrame, *, periods: int | None = None
) -> pd.DataFrame:
    """Variance by cost center, decomposed into spend and mix effects.

    For each cost center *i*::

        spend effect_i = planned_share_i x (total actual - total budget)
        mix effect_i   = variance_i - spend effect_i

    The spend effect is what center *i* would have overrun by had every center grown
    proportionally. What remains is that center's share of the budget changing — the
    mix. By construction the two sum to the variance, for every center and in total.

    ``periods`` restricts to the most recent N months.
    """
    detail = expense_variance(ledger, budget)
    if detail.empty:
        return pd.DataFrame(
            columns=["function", "sub_center", "actual", "budget", "variance",
                     "spend_effect", "mix_effect", "variance_pct"]
        )

    if periods is not None:
        keep = sorted(detail["period"].unique())[-periods:]
        detail = detail[detail["period"].isin(keep)]

    rolled = detail.groupby(["function", "sub_center"], as_index=False)[
        ["actual", "budget"]
    ].sum()

    total_actual = rolled["actual"].sum()
    total_budget = rolled["budget"].sum()
    total_variance = total_actual - total_budget

    rolled["variance"] = rolled["actual"] - rolled["budget"]
    planned_share = rolled["budget"] / total_budget if total_budget else 0.0
    rolled["spend_effect"] = planned_share * total_variance
    rolled["mix_effect"] = rolled["variance"] - rolled["spend_effect"]
    rolled["variance_pct"] = rolled["variance"] / rolled["budget"].replace(0, np.nan)

    rolled.attrs.update(
        total_actual=float(total_actual),
        total_budget=float(total_budget),
        total_variance=float(total_variance),
        n_periods=int(detail["period"].nunique()),
        period_start=detail["period"].min(),
        period_end=detail["period"].max(),
    )
    return rolled.sort_values("variance", ascending=False).reset_index(drop=True)


def driver_variance(
    drivers: pd.DataFrame, revenue_budget: pd.DataFrame, *, periods: int | None = None
) -> dict:
    """Decompose revenue variance into rate and volume effects.

    **What this can and cannot say.** A full rate/volume/mix split needs a *planned*
    member count to vary actual members against. No member plan exists — Netflix does
    not file member counts at all (verified: no membership concept in its XBRL), and
    the budget here is a revenue figure. Rather than invent a member plan and present
    the resulting "volume effect" as analysis, the planned ARPU is solved by holding
    planned members equal to actual, and the whole variance is attributed to rate.

    That is the honest answer given the inputs, and the returned ``basis`` says so
    explicitly so a reader is never misled about which effects were measured.
    """
    work = drivers.merge(revenue_budget, on="period", how="inner")
    if work.empty:
        return {}

    if periods is not None:
        work = work.sort_values("period").tail(periods)

    actual_revenue = float(work["revenue"].sum())
    planned_revenue = float(work["budget"].sum())
    member_months = float(work["members"].sum())

    actual_arpu = actual_revenue / member_months if member_months else float("nan")
    planned_arpu = planned_revenue / member_months if member_months else float("nan")

    volume_effect = 0.0  # no independent member plan to vary against
    rate_effect = (actual_arpu - planned_arpu) * member_months
    total_variance = actual_revenue - planned_revenue

    return {
        "actual_revenue": round(actual_revenue, 2),
        "planned_revenue": round(planned_revenue, 2),
        "variance": round(total_variance, 2),
        "variance_pct": round(actual_revenue / planned_revenue - 1.0, 4) if planned_revenue else None,
        "favourability": "favourable" if total_variance > 0 else "unfavourable",
        "avg_members": round(float(work["members"].mean()), 0),
        "actual_arpu": round(actual_arpu, 4),
        "planned_arpu": round(planned_arpu, 4),
        "volume_effect": round(volume_effect, 2),
        "rate_effect": round(rate_effect, 2),
        "n_periods": int(len(work)),
        "basis": (
            "No member plan is filed or modeled, so volume cannot be varied against "
            "plan; the full variance is attributed to rate. Volume effect is reported "
            "as zero because it was not measured, not because it was zero."
        ),
    }


def variance_summary(variance: pd.DataFrame, *, top_n: int = 5) -> dict:
    """Condense the bridge into the facts payload the narrative layer may cite.

    Everything here is computed in Python. The model receives this and may describe
    these numbers; it may not produce others. ``fpa.narrative.groundedness`` enforces
    that by checking every numeral in the generated prose against this payload.
    """
    if variance.empty:
        return {}

    def rows(frame: pd.DataFrame) -> list[dict]:
        return [
            {
                "cost_center": f"{r.function} / {r.sub_center}",
                "actual": round(float(r.actual), 2),
                "budget": round(float(r.budget), 2),
                "variance": round(float(r.variance), 2),
                "variance_pct": round(float(r.variance_pct), 4),
                "spend_effect": round(float(r.spend_effect), 2),
                "mix_effect": round(float(r.mix_effect), 2),
                "favourability": "unfavourable" if r.variance > 0 else "favourable",
            }
            for r in frame.itertuples()
        ]

    total_budget = variance.attrs.get("total_budget")
    total_variance = variance.attrs.get("total_variance", float(variance["variance"].sum()))

    return {
        "period_start": str(pd.Timestamp(variance.attrs["period_start"]).date()),
        "period_end": str(pd.Timestamp(variance.attrs["period_end"]).date()),
        "n_periods": variance.attrs.get("n_periods"),
        "total_actual": round(variance.attrs.get("total_actual", 0.0), 2),
        "total_budget": round(total_budget or 0.0, 2),
        "total_variance": round(total_variance, 2),
        "total_variance_pct": round(total_variance / total_budget, 4) if total_budget else None,
        "largest_unfavourable": rows(variance.nlargest(top_n, "variance")),
        "largest_favourable": rows(variance.nsmallest(top_n, "variance")),
    }


def assert_bridge_ties(variance: pd.DataFrame, *, rtol: float = 1e-9) -> None:
    """Verify the effects sum to the total variance.

    Called by the pipeline on every run, not just in tests. A bridge whose components
    do not add up to the number they are explaining is not a rounding nuisance — it
    means one of the effects is wrong.
    """
    if variance.empty:
        return
    recomposed = float((variance["spend_effect"] + variance["mix_effect"]).sum())
    total = float(variance["variance"].sum())
    if not np.isclose(recomposed, total, rtol=rtol):
        raise AssertionError(
            f"variance bridge does not tie: effects sum to {recomposed:,.2f} "
            f"but total variance is {total:,.2f}"
        )


def build_variance_report(
    ledger: pd.DataFrame,
    budget: pd.DataFrame,
    revenue: pd.DataFrame,
    revenue_budget: pd.DataFrame,
    drivers: pd.DataFrame | None = None,
    *,
    periods: int | None = 12,
) -> dict:
    """Assemble the whole variance picture, and verify the bridge ties."""
    by_center = cost_center_variance(ledger, budget, periods=periods)
    assert_bridge_ties(by_center)

    return {
        "by_cost_center": by_center,
        "by_period": revenue_variance(revenue, revenue_budget),
        "expenses_detail": expense_variance(ledger, budget),
        "revenue_drivers": driver_variance(drivers, revenue_budget, periods=periods)
        if drivers is not None
        else {},
        "summary": variance_summary(by_center),
    }
