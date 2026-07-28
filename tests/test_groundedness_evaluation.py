"""Tests for the groundedness *evaluation* — one level up from the checker.

The evaluation reports 0% false acceptance and 100% parse coverage. On its own that
is not evidence of anything: an evaluation that cannot detect a broken checker
produces exactly the same numbers as one facing a working checker.

So these tests break the checker on purpose — reverting it to the two bugs actually
found in it by hand, plus a widened tolerance — and assert the evaluation goes red.
Same convention as everywhere else in this repo, applied one level higher: **the
tests validate the validator's validator.**

They also guard against the mistake made while building this module. The first
version labelled cases by running the checker's own matching rule over them, which
made positives pass and negatives fail by construction and turned both rates into
tautologies. ``test_labels_do_not_consult_the_checkers_tolerance`` exists so that
cannot come back.
"""

from __future__ import annotations

import re

import pytest

from fpa.narrative import evaluation as ev
from fpa.narrative import groundedness as gr

PAYLOAD = {
    "variance": {
        "actual": 2_654_321_000.0,
        "budget": 2_050_000_000.0,
        "variance": 604_321_000.0,
        "variance_pct": 0.229,
    },
    "drivers": [
        {"cost_center": "Cloud Infrastructure", "amount": 1_785_400_000.0},
        {"cost_center": "Licensed Content", "amount": -495_473_475.79},
    ],
    "period_end": "2026-06-30",
}


@pytest.fixture
def corpus():
    return ev.build_corpus(PAYLOAD)


def test_corpus_contains_both_labels_in_useful_numbers(corpus):
    positives = [c for c in corpus if c.grounded]
    negatives = [c for c in corpus if not c.grounded]
    assert len(positives) >= 40
    assert len(negatives) >= 20
    # Several distinct mutation kinds, or the negatives only probe one mechanism.
    assert len({c.kind for c in negatives}) >= 4


def test_a_working_checker_scores_clean(corpus):
    report = ev.evaluate(corpus, PAYLOAD)
    assert report.false_acceptance_rate == 0.0
    assert report.parse_coverage == 1.0
    assert report.clean


# ---------------------------------------------------------------------------
# The evaluation must go red when the checker is broken
# ---------------------------------------------------------------------------
def test_evaluation_catches_the_sentence_final_regex_bug(corpus, monkeypatch):
    """The real bug: ``(?![\\w.])`` meant ``$999.9M.`` never matched at all.

    It is invisible to accept/reject accuracy, because an unmatched numeral gets no
    verdict — it is simply absent from ``checked``. Only parse coverage sees it, and
    if this test ever stops failing the coverage metric has stopped working.
    """
    buggy = re.compile(
        gr._NUMERAL.pattern.replace(r"(?!\w)(?!\.\d)", r"(?![\w.])"),
        re.IGNORECASE | re.VERBOSE,
    )
    monkeypatch.setattr(gr, "_NUMERAL", buggy)
    monkeypatch.setattr(ev, "_NUMERAL", buggy)

    report = ev.evaluate(corpus, PAYLOAD)
    assert report.parse_coverage < 1.0, "parse coverage failed to notice a blind parser"
    assert report.parse_misses
    assert not report.clean


def test_evaluation_catches_a_tolerance_that_is_too_loose(corpus, monkeypatch):
    """A checker that accepts anything within 50% accepts fabrications.

    This is the direction that matters: a figure nobody computed reaching a reviewer.
    """
    monkeypatch.setattr(gr, "MATCH_RTOL", 0.5)

    report = ev.evaluate(corpus, PAYLOAD)
    assert report.false_acceptance_rate > 0.0
    assert not report.clean


def test_evaluation_catches_a_tolerance_that_is_too_tight(corpus, monkeypatch):
    """A checker demanding exactness rejects correct prose, because prose rounds.

    A cost rather than a safety failure — so it must show up in the false *rejection*
    rate and must **not** be treated as making the report unclean.
    """
    monkeypatch.setattr(gr, "MATCH_RTOL", 1e-12)

    report = ev.evaluate(corpus, PAYLOAD)
    assert report.false_rejection_rate > 0.0
    assert report.false_acceptance_rate == 0.0
    assert report.clean, "a false rejection is a cost, not a fabrication reaching a reviewer"


def test_evaluation_catches_a_checker_that_ignores_scale_suffixes(corpus, monkeypatch):
    """Dropping the scale multiplier makes ``$604.4M`` claim 604.4, which is wrong."""
    monkeypatch.setattr(gr, "_candidates", lambda raw, scale, percent: [float(raw.replace(",", ""))])

    report = ev.evaluate(corpus, PAYLOAD)
    assert report.false_rejection_rate > 0.0


# ---------------------------------------------------------------------------
# The labels must not be derived from the thing under test
# ---------------------------------------------------------------------------
def test_labels_do_not_consult_the_checkers_tolerance():
    """Ground truth must be independent, or both rates are tautologies.

    The first version of this module labelled cases with the checker's own matching
    rule. Positives then passed by construction and negatives failed by construction,
    and it reported a perfect score while measuring nothing at all.

    The bands must stay separated with ``MATCH_RTOL`` strictly inside the gap, so the
    checker is free to disagree with the corpus.
    """
    assert ev.POSITIVE_PRECISION < gr.MATCH_RTOL < ev.NEGATIVE_MARGIN
    assert gr.MATCH_RTOL / ev.POSITIVE_PRECISION >= 5
    assert ev.NEGATIVE_MARGIN / gr.MATCH_RTOL >= 5


def test_changing_the_checkers_tolerance_does_not_change_the_corpus():
    """The decisive property: labels are fixed, verdicts are what vary.

    If widening the tolerance also widened the corpus's notion of grounded, the
    evaluation would move in lockstep with the checker and could never disagree
    with it.
    """
    before = {(c.text, c.grounded) for c in ev.build_corpus(PAYLOAD)}

    original = gr.MATCH_RTOL
    try:
        gr.MATCH_RTOL = 0.5
        after = {(c.text, c.grounded) for c in ev.build_corpus(PAYLOAD)}
    finally:
        gr.MATCH_RTOL = original

    assert before == after


# ---------------------------------------------------------------------------
# Parse-miss detection
# ---------------------------------------------------------------------------
def test_parse_misses_are_found_by_span_not_by_count():
    """Equal counts can still mean the parsers matched different things."""
    assert ev.parse_misses("Spend was $604.4M this quarter.") == []
    # A malformed numeral the strict parser deliberately refuses is still a span it
    # gave no verdict on, and the crude parser should surface it for adjudication.
    assert ev.parse_misses("Version 1.2.3 shipped")


def test_report_names_the_failure_direction_that_matters(corpus):
    report = ev.evaluate(corpus, PAYLOAD)
    markdown = ev.report_markdown(report)
    assert "False acceptance rate" in markdown
    assert "Parse coverage" in markdown
    # The limitation has to survive into the rendered report, not just the docstring.
    assert "does not measure" in markdown
