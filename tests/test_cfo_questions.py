"""Tests for the CFO question set — mostly guarding against becoming a chatbot.

The scaffold this project replaced shipped an ``if/elif`` panel that asserted a
correlation nobody had computed. A question-and-answer feature is structurally the same
thing, so what these tests pin is the *inversion*: facts are computed before the model
is involved, and every figure an answer may contain is already in the payload.

``test_every_answer_is_checkable`` is the one that matters. If a fact builder ever
returns prose containing a figure that is not also a value in the payload, the
groundedness checker cannot verify it, and the panel has quietly become the thing it
was built to avoid.
"""

from __future__ import annotations

import pytest

from fpa.config import get_settings
from fpa.narrative.facts import to_json
from fpa.narrative.groundedness import check
from fpa.narrative.questions import BY_ID, QUESTIONS, REFUSED, build_answer_facts
from fpa.pipeline import run
from fpa.variance import build_variance_report


@pytest.fixture(scope="module")
def context():
    result = run(get_settings())
    report = build_variance_report(
        result.ledger, result.budget, result.revenue, result.revenue_budget,
        result.drivers, periods=12,
    )
    return result, report


def _facts_for(question, context):
    result, report = context
    kwargs = {}
    if question.id == "planning_range":
        from fpa.forecast.models import leaf_series
        from fpa.forecast.posterior import load_posterior

        kwargs = {
            "posterior": load_posterior(result.settings),
            "series": list(leaf_series(result.ledger).columns)[0],
        }
    return build_answer_facts(question, result, report, **kwargs)


def test_there_are_six_answerable_questions():
    assert len(QUESTIONS) == 6
    assert all(q.answerable for q in QUESTIONS)
    assert len({q.id for q in QUESTIONS}) == 6


def test_every_question_computes_facts(context):
    for question in QUESTIONS:
        facts = _facts_for(question, context)
        assert isinstance(facts, dict) and facts, question.id
        assert facts["question"] == question.id


def test_every_answer_is_checkable(context):
    """Prose in a payload must not contain a figure the payload cannot vouch for.

    Each fact builder carries a ``basis`` string explaining the metric. If one of those
    sentences cites a number, the checker will see it in a generated answer and have
    nothing to match it against — a false rejection at best, and at worst a figure that
    looks sourced because it appeared in the payload as *text*.
    """
    for question in QUESTIONS:
        facts = _facts_for(question, context)
        basis = facts.get("basis", "")
        outcome = check(basis, facts)
        assert outcome.passed, f"{question.id} basis cites an unverifiable figure: {outcome.message}"


def test_payloads_are_json_serializable(context):
    for question in QUESTIONS:
        assert to_json(_facts_for(question, context))


def test_refusal_is_not_in_the_answerable_set():
    """It sits beside the answers, not among them — and it has no facts to compute."""
    assert REFUSED not in QUESTIONS
    assert not REFUSED.answerable
    assert REFUSED.refusal

    with pytest.raises(ValueError, match="refused"):
        build_answer_facts(REFUSED, None, None)


def test_the_refusal_explains_the_identification_problem():
    """A refusal that does not say *why* teaches nothing and reads as a limitation."""
    text = REFUSED.refusal.lower()
    assert "elasticity" in text
    assert "counterfactual" in text
    assert "one realised price path" in text or "single realised price path" in text


def test_operating_leverage_is_internally_consistent(context):
    """The one question with no modeled layer under it at all.

    The invariant worth asserting is the *sign relationship*: margin can only expand
    when revenue outgrows cost. If those two ever disagree the metric is telling a CFO
    the opposite of what the totals say, which is worse than not computing it.
    """
    facts = _facts_for(BY_ID["operating_leverage"], context)

    # Payload values are rounded for display, so compare at the rounding, not below it.
    implied_margin = (facts["revenue_ttm"] - facts["opex_ttm"]) / facts["revenue_ttm"]
    assert facts["operating_margin_now"] == pytest.approx(implied_margin, abs=5e-5)

    gap = (facts["revenue_growth"] - facts["opex_growth"]) * 100
    assert facts["leverage_gap_pts"] == pytest.approx(gap, abs=5e-3)

    margin_moved = facts["operating_margin_now"] - facts["operating_margin_prior"]
    assert (facts["leverage_gap_pts"] > 0) == (margin_moved > 0), (
        "positive operating leverage must coincide with margin expansion"
    )


def test_ledger_ties_reports_the_footing_it_claims(context):
    """The answer to 'how much did we make up' must be measured, not asserted."""
    facts = _facts_for(BY_ID["ledger_ties"], context)
    assert facts["relative_difference"] < 1e-9
    assert facts["cost_centers"] == 9
    assert "no filer discloses spend by department" in facts["basis"]


def test_planning_range_carries_its_diagnostics(context):
    """An interval whose R-hat nobody inspected is not evidence.

    The payload must carry convergence, so a generated answer can be held to it and a
    reviewer is never handed a range from a fit the sampler did not explore.
    """
    facts = _facts_for(BY_ID["planning_range"], context)
    if not facts.get("available", True):
        pytest.skip("no pinned posterior in this vintage")

    assert {"converged", "worst_rhat", "min_ess"} <= facts.keys()
    assert facts["widening"] > 1.0, "a 12-month band must be wider than a 1-month band"


def test_forecast_trust_reports_where_the_model_loses(context):
    """Never publish a metric without its benchmark, and never hide the losses."""
    facts = _facts_for(BY_ID["forecast_trust"], context)
    assert facts["series_losing_to_naive"] > 0
    assert "worst_series" in facts
    assert facts["worst_series"]["mase"] > 1.0
    # Both backtests, with the flattering one named as the artifact it is.
    assert facts["mase_modeled_monthly"] < facts["mase_filed_quarterly"]
    assert "rediscovering our own allocation weights" in facts["why_two_backtests"]
