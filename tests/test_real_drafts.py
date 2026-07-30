"""Tests for the real-draft corpus — the half of the groundedness evaluation that was open.

``fpa.narrative.evaluation`` scores the checker on 364 synthetic cases and has always carried
the limitation in its own report: whether synthetic phrasing matches what a real model writes
is *not measured*, and that is the only thing testing generalisation. A generator can only
produce the phrasings it was written to produce.

These tests run against a **frozen corpus of real drafts** so the measurement is reproducible
without calling a model. The one that matters is
``test_the_report_refuses_to_overstate_a_digit_only_corpus``: 100% parse coverage over prose
that happens to be entirely digit-form says the regex handles digits, not that it handles
prose, and a report that does not say so is worse than no report.
"""

from __future__ import annotations

import pytest

from fpa.config import get_settings
from fpa.narrative import real_drafts as RD


@pytest.fixture(scope="module")
def corpus():
    drafts = RD.load(get_settings())
    if not drafts:
        pytest.skip("no frozen corpus; run fpa.narrative.real_drafts to build one")
    return drafts


def test_the_corpus_is_real_prose_not_a_fixture_of_ours(corpus):
    """Every entry has to carry the payload it was generated against, or the drafts cannot
    be re-checked later against the facts that produced them."""
    assert len(corpus) >= 10
    for entry in corpus:
        assert "draft" in entry and "facts" in entry
        assert entry["draft"].get("headline"), "a draft with no prose measures nothing"


def test_prose_is_collected_from_every_free_text_field(corpus):
    """A numeral can hide in any field a model filled in — headline, driver comment,
    watch item. Checking only the headline would report coverage over a fraction."""
    fields = sum(len(RD._prose(entry["draft"])) for entry in corpus)
    assert fields > len(corpus), "expected more prose fields than drafts"


def test_parse_coverage_is_measured_over_real_numerals(corpus):
    report = RD.parse_report(corpus)

    assert report.numerals > 100, "too few numerals to say anything"
    assert report.drafts == len(corpus)
    assert 0.0 <= report.coverage <= 1.0


def test_any_unparsed_span_is_named_not_counted(corpus):
    """A parse miss is a numeral the checker gave *no verdict* on — the failure mode the
    accept/reject rates are structurally blind to. If one exists it has to be shown."""
    report = RD.parse_report(corpus)
    if report.unparsed:
        markdown = RD.report_markdown(report, RD.phrasing_census(corpus))
        for span in report.unparsed[:5]:
            assert span in markdown


def test_the_report_refuses_to_overstate_a_digit_only_corpus(corpus):
    """The honesty guard on this whole exercise.

    Real drafts came back at 100% parse coverage, which reads like the gap is closed. It is
    not: the census showed the model wrote in digits almost exclusively, so the hard case —
    a figure spelled out in words — never occurred. Untested is not the same as passed, and
    the report must say which one it is.
    """
    census = RD.phrasing_census(corpus)
    markdown = RD.report_markdown(RD.parse_report(corpus), census)

    assert "digit form" in markdown and "word form" in markdown
    if census["word_form"] == 0:
        assert "untested rather than passed" in markdown
        # And the structural reason, because it is a design decision doing the work.
        assert "JSON schema" in markdown


def test_the_report_keeps_the_open_half_open(corpus):
    """Accept/reject accuracy on real drafts is *not* measured, and claiming otherwise
    would rebuild the tautology evaluation.py exists to prevent."""
    markdown = RD.report_markdown(RD.parse_report(corpus), RD.phrasing_census(corpus))
    assert "stays open" in markdown
    assert "tautology" in markdown


def test_the_census_counts_digits_and_words_separately(corpus):
    census = RD.phrasing_census(corpus)
    assert set(census) == {"digit_form", "word_form"}
    assert census["digit_form"] > 0
    # "six periods" is a count of periods, not a figure needing verification, and the
    # census must not inflate itself by counting it.
    assert census["word_form"] < census["digit_form"]
