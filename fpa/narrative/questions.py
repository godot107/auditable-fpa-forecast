"""A fixed question set a CFO can ask, and the facts that answer each one.

This is the feature most likely to recreate the defect the project was built to
remove. The scaffold this replaced shipped an ``if/elif`` chatbot that asserted a
*"statistically significant correlation 45 days later"* which was never computed
anywhere — a plausible sentence with no arithmetic behind it. A question-and-answer
panel is structurally the same feature, so it is built the other way round:

1. The **question set is fixed**. A CFO picks from questions the pipeline can actually
   answer, rather than typing free text at a model that will answer regardless.
2. Each question maps to a Python function that **computes** its facts from the
   pipeline result. The model never sees the ledger, only the payload.
3. The model writes prose about that payload and nothing else. Every numeral it
   returns is checked back against the facts by ``fpa.narrative.groundedness``.
4. The exchange is appended to the audit log with the question, the provider, the
   prompt version, the groundedness verdict and the reviewer's decision.

So the panel is not a chatbot. It is a **receipted transcript**: for every answer, the
figures it may contain were computed first, and there is a record of who accepted it.

**One question is deliberately unanswerable.** ``price_elasticity`` asks what churn
would do if prices rose, and is refused — not because the model would fail to produce
a fluent answer, but because it would produce one. Nothing in filed data identifies how
members respond to a price change: the company filed one price path, and one realised
history contains no counterfactual. A system that answers every question asked of it
cannot be trusted on the ones it happens to get right, and the refusal is the single
most honest thing in the narrative layer.

Future work, noted rather than hidden: an agent with a financial-analyst system prompt
could *select* which question a free-text request maps to. That is a routing problem and
a safe place for a model — the answer would still be computed in Python and still pass
the groundedness gate. What must never move behind the model is the arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Question:
    """A question, and either the facts that answer it or the reason it is refused."""

    id: str
    text: str
    rationale: str
    facts: Callable[..., dict] | None = None
    refusal: str | None = None

    @property
    def answerable(self) -> bool:
        return self.facts is not None


# ---------------------------------------------------------------------------
# Fact builders. Each returns only computed values — no prose, no adjectives.
# ---------------------------------------------------------------------------
def _variance_shape(result, report) -> dict:
    """Was the miss a level effect or a mix effect?"""
    frame = report["by_cost_center"]
    worst_mix = frame.loc[frame["mix_effect"].idxmax()]
    best_mix = frame.loc[frame["mix_effect"].idxmin()]
    summary = report["summary"]

    return {
        "question": "spend_vs_mix",
        "totals": {
            "actual": round(float(summary["total_actual"]), 2),
            "budget": round(float(summary["total_budget"]), 2),
            "variance": round(float(summary["total_variance"]), 2),
            "variance_pct": round(float(summary["total_variance_pct"]), 4),
        },
        "largest_unfavourable_mix": {
            "cost_center": f"{worst_mix['function']} / {worst_mix['sub_center']}",
            "mix_effect": round(float(worst_mix["mix_effect"]), 2),
            "variance": round(float(worst_mix["variance"]), 2),
        },
        "largest_favourable_mix": {
            "cost_center": f"{best_mix['function']} / {best_mix['sub_center']}",
            "mix_effect": round(float(best_mix["mix_effect"]), 2),
            "variance": round(float(best_mix["variance"]), 2),
        },
        "basis": (
            "Spend effect is each centre's planned share of the total variance; mix "
            "effect is the remainder. Spend effects sum to the total variance and mix "
            "effects sum to zero, so mix is a redistribution rather than new money."
        ),
    }


def _planning_range(result, report, *, posterior=None, series=None) -> dict:
    """What range should the next twelve months be planned against?"""
    if posterior is None or series is None:
        return {
            "question": "planning_range",
            "available": False,
            "basis": (
                "No pinned posterior for this vintage. Build it with "
                "python -m fpa.forecast.posterior."
            ),
        }

    from fpa.forecast.posterior import posterior_intervals

    bands = posterior_intervals(posterior, series, result.settings.horizon_months)
    first, last = bands.iloc[0], bands.iloc[-1]
    return {
        "question": "planning_range",
        "cost_center": f"{series[0]} / {series[1]}",
        "month_1": {
            "downside_p10": round(float(first["p10"]), 2),
            "base_p50": round(float(first["p50"]), 2),
            "upside_p90": round(float(first["p90"]), 2),
        },
        "month_12": {
            "downside_p10": round(float(last["p10"]), 2),
            "base_p50": round(float(last["p50"]), 2),
            "upside_p90": round(float(last["p90"]), 2),
        },
        "widening": round(
            float((last["p90"] - last["p10"]) / (first["p90"] - first["p10"])), 2
        ),
        "converged": bool(bands.attrs["converged"]),
        "worst_rhat": round(float(bands.attrs["worst_rhat"]), 3),
        "min_ess": round(float(bands.attrs["min_ess"]), 0),
        "basis": (
            "p10/p50/p90 are the downside, base and upside cases measured from the "
            "posterior predictive rather than chosen. They describe an unchanged "
            "future, not the effect of a decision."
        ),
    }


def _forecast_trust(result, report) -> dict:
    """How much confidence does the backtest support?"""
    if result.backtest_filed is None:
        return {"question": "forecast_trust", "available": False}

    summary = result.backtest_filed.summary()
    best = summary.iloc[0]
    losses = result.backtest_filed.losses_to_benchmark()

    payload = {
        "question": "forecast_trust",
        "best_model": str(best["model"]),
        "mase_filed_quarterly": round(float(best["mase"]), 3),
        "models_compared": int(len(summary)),
        "series_losing_to_naive": int(len(losses)),
        "basis": (
            "MASE is scaled by the in-sample seasonal-naive error: below 1.0 beats "
            "'same quarter last year', above 1.0 loses to it. The quoted figure is the "
            "best of the models compared on the same backtest, so it is the optimistic "
            "end of a narrow range rather than an unbiased estimate."
        ),
    }
    if len(losses):
        worst = losses.iloc[0]
        payload["worst_series"] = {
            "series": str(worst["series"]),
            "model": str(worst["model"]),
            "mase": round(float(worst["mase"]), 3),
        }
    if result.backtest_monthly is not None:
        payload["mase_modeled_monthly"] = round(
            float(result.backtest_monthly.summary().iloc[0]["mase"]), 3
        )
        payload["why_two_backtests"] = (
            "The monthly figure is measured on a ledger this project disaggregated "
            "itself, so a model scores partly by rediscovering our own allocation "
            "weights. The filed-quarterly figure is the honest one."
        )
    return payload


def _operating_leverage(result, report) -> dict:
    """Is the cost base growing faster than revenue?

    Entirely filed data — no modeled layer touches this — and it is the question
    underneath most others. Operating leverage is positive when revenue outgrows cost,
    which is the only way margin expands without a price rise.
    """
    quarterly = result.quarterly.sort_index()
    expense_columns = [
        c
        for c in ("cost_of_revenue", "research_development", "general_administrative", "marketing")
        if c in quarterly.columns
    ]
    revenue = quarterly["revenue"]
    opex = quarterly[expense_columns].sum(axis=1)

    # Trailing four quarters against the four before them: a full year each side, so
    # seasonality cancels rather than being extrapolated from one quarter.
    if len(revenue) < 8:
        return {"question": "operating_leverage", "available": False}

    rev_now, rev_prior = float(revenue.iloc[-4:].sum()), float(revenue.iloc[-8:-4].sum())
    opex_now, opex_prior = float(opex.iloc[-4:].sum()), float(opex.iloc[-8:-4].sum())
    # Round first, then derive. The payload has to be self-consistent at the precision
    # it publishes: a reader — or a model — subtracting the two growth figures it was
    # given must land on the gap the payload states. Deriving from full precision and
    # rounding afterwards put 0.27 in the payload beside figures that subtract to 0.28,
    # which the groundedness checker would reject as a fabricated number.
    rev_growth = round(rev_now / rev_prior - 1.0, 4)
    opex_growth = round(opex_now / opex_prior - 1.0, 4)

    return {
        "question": "operating_leverage",
        "period": f"{quarterly.index[-4]:%Y-%m-%d} to {quarterly.index[-1]:%Y-%m-%d}",
        "revenue_ttm": round(rev_now, 2),
        "revenue_growth": rev_growth,
        "opex_ttm": round(opex_now, 2),
        "opex_growth": opex_growth,
        "leverage_gap_pts": round((rev_growth - opex_growth) * 100, 2),
        "operating_margin_now": round((rev_now - opex_now) / rev_now, 4),
        "operating_margin_prior": round((rev_prior - opex_prior) / rev_prior, 4),
        "basis": (
            "Trailing twelve months against the prior twelve, so seasonality cancels on "
            "both sides. Positive leverage means revenue grew faster than cost, which is "
            "the only route to margin expansion that does not require raising price. "
            "Every figure here is filed; no modeled layer is involved."
        ),
    }


def _margin_reliability(result, report) -> dict:
    """Which line is least predictable — and why margin guidance is the riskiest promise?"""
    if result.backtest_filed is None:
        return {"question": "margin_reliability", "available": False}

    by_model = result.backtest_filed.by_model
    best_per_series = (
        by_model[by_model["model"] != "seasonal_naive"]
        .sort_values("mase")
        .groupby("series", as_index=False)
        .first()
        .sort_values("mase", ascending=False)
    )
    hardest = best_per_series.iloc[0]
    easiest = best_per_series.iloc[-1]

    payload = {
        "question": "margin_reliability",
        "hardest_series": {
            "series": str(hardest["series"]),
            "model": str(hardest["model"]),
            "mase": round(float(hardest["mase"]), 3),
        },
        "easiest_series": {
            "series": str(easiest["series"]),
            "model": str(easiest["model"]),
            "mase": round(float(easiest["mase"]), 3),
        },
        "series_scored": int(len(best_per_series)),
        "basis": (
            "MASE below 1.0 beats 'same quarter last year'. Operating income is "
            "consistently hardest because it is a small difference between two large "
            "numbers: proportionally modest errors in revenue and in cost compound into "
            "a large error in the residual. That is the argument for forecasting the "
            "drivers and letting margin fall out, rather than guiding to margin directly."
        ),
    }
    row = best_per_series[best_per_series["series"] == "operating_income"]
    if len(row):
        payload["operating_income_mase"] = round(float(row.iloc[0]["mase"]), 3)
    return payload


def _ledger_ties(result, report) -> dict:
    """Does the modeled detail reconcile to the filings?"""
    ledger_total = float(result.ledger["amount"].sum())
    quarterly = result.quarterly
    expense_columns = [
        c
        for c in ("cost_of_revenue", "research_development", "general_administrative", "marketing")
        if c in quarterly.columns
    ]
    filed_total = float(quarterly[expense_columns].sum().sum())
    difference = ledger_total - filed_total

    return {
        "question": "ledger_ties",
        "modeled_ledger_total": round(ledger_total, 2),
        "filed_expense_total": round(filed_total, 2),
        "difference": round(difference, 2),
        "relative_difference": (
            round(abs(difference) / filed_total, 12) if filed_total else 0.0
        ),
        "months_modeled": int(result.ledger["period"].nunique()),
        "cost_centers": int(
            result.ledger.groupby(["function", "sub_center"]).ngroups
        ),
        "basis": (
            "The cost-center split and the intra-quarter phasing are modeled by this "
            "project; no filer discloses spend by department. Only the totals are "
            "filed, and the modeled detail is forced to sum to them."
        ),
    }


# ---------------------------------------------------------------------------
# The set
# ---------------------------------------------------------------------------
QUESTIONS: tuple[Question, ...] = (
    Question(
        id="operating_leverage",
        text="Is our cost base growing faster than revenue?",
        rationale=(
            "The question underneath most others, and answerable entirely from filed "
            "data. Positive operating leverage is the only route to margin expansion "
            "that does not require raising price."
        ),
        facts=_operating_leverage,
    ),
    Question(
        id="spend_vs_mix",
        text="We missed plan. Did we spend more, or spend it differently?",
        rationale=(
            "One number cannot separate the two, and they call for different action: a "
            "level miss is a budget problem, a mix miss is a strategy that changed "
            "without a re-plan."
        ),
        facts=_variance_shape,
    ),
    Question(
        id="ledger_ties",
        text="How much of this ledger is filed, and how much did we model?",
        rationale=(
            "The question a CFO should ask of any management ledger, and the one this "
            "pipeline answers most precisely: filed at the total, modeled at the line."
        ),
        facts=_ledger_ties,
    ),
    Question(
        id="planning_range",
        text="What range should I plan this cost centre against for the next year?",
        rationale=(
            "Downside, base and upside measured from the posterior predictive rather "
            "than set by convention, each with a stated probability."
        ),
        facts=_planning_range,
    ),
    Question(
        id="forecast_trust",
        text="How much should I trust that forecast?",
        rationale=(
            "Accuracy against a benchmark, including the series where the model loses. "
            "A figure quoted without a benchmark can be neither trusted nor distrusted."
        ),
        facts=_forecast_trust,
    ),
    Question(
        id="margin_reliability",
        text="Which line is least predictable — and what does that mean for guidance?",
        rationale=(
            "Turns backtest error into a decision: the least predictable series is the "
            "riskiest thing to promise a board, and here it is margin itself."
        ),
        facts=_margin_reliability,
    ),
)

# Kept apart from the answerable set on purpose. It is not a seventh option competing
# for a slot — it is the demonstration that the set is bounded, shown alongside the
# answers rather than buried in them.
REFUSED = Question(
    id="price_elasticity",
    text="If we raised the standard price by $2, what would happen to churn?",
    rationale=(
        "Included precisely because it cannot be answered. A system that answers "
        "everything asked of it cannot be trusted on the ones it gets right."
    ),
    refusal=(
        "**Refused — no fact in this pipeline supports an answer.**\n\n"
        "This asks for an *elasticity*: how members respond to a price change. "
        "Estimating one needs variation in price alongside the response to it, and the "
        "filings contain a single realised price path. One history holds no "
        "counterfactual, so there is nothing to measure against.\n\n"
        "A language model would answer this fluently, and the number would look exactly "
        "like the ones above — which is why the question set is fixed and every figure "
        "is computed before the model sees it.\n\n"
        "What *can* be answered: how margin moves **if** content spend grows at a stated "
        "rate. That is assumption propagation, and the hierarchy re-foots by "
        "construction. What cannot be answered is what happens **when you decide** to "
        "change it."
    ),
)

BY_ID = {q.id: q for q in QUESTIONS + (REFUSED,)}


def build_answer_facts(question: Question, result, report, **kwargs) -> dict:
    """Compute the facts for one question. Raises if the question is refused."""
    if not question.answerable:
        raise ValueError(f"question {question.id!r} is refused and has no facts")
    return question.facts(result, report, **kwargs)
