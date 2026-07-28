"""Tests for the anti-hallucination check.

These are the most important tests in the suite, because they **validate the
validator**. A groundedness check that passes everything is worse than none at all:
it produces a false assurance that every figure was verified. So the suite proves
the check actually rejects fabricated numbers, not merely that it accepts real ones.
"""

from __future__ import annotations

import pytest

from fpa.narrative.groundedness import (
    GroundingError,
    assert_grounded,
    check,
    check_draft,
    collect_values,
)

PAYLOAD = {
    "variance": {
        "total_variance": 2_419_281_959.81,
        "total_variance_pct": 0.0766,
        "n_periods": 12,
        "largest_unfavourable": [
            {
                "cost_center": "Technology & Product / Cloud Infrastructure",
                "variance": 604_398_923.71,
                "variance_pct": 0.2292,
                "spend_effect": 201_879_667.91,
                "mix_effect": 402_519_255.81,
            },
            {
                "cost_center": "Content / Licensed Content",
                "variance": 394_812_247.40,
                "mix_effect": -495_473_475.79,
            },
        ],
        "period_end": "2026-06-30",
    }
}


def test_rejects_a_fabricated_dollar_figure():
    """The whole point. A number nobody computed must not pass."""
    result = check("Cloud Infrastructure overran by $999.9M.", PAYLOAD)
    assert not result.passed
    assert "$999.9M" in result.ungrounded


def test_rejects_a_fabricated_percentage():
    result = check("That is a 47.3% overrun.", PAYLOAD)
    assert not result.passed


def test_rejects_a_plausible_but_uncomputed_derivation():
    """A model that does its own arithmetic is hallucinating, even if the maths is right.

    604,398,923.71 - 201,879,667.91 = 402,519,255.80, which happens to be in the
    payload. But 604,398,923.71 + 394,812,247.40 is not, and must not pass merely
    because both operands were.
    """
    result = check("The two largest overruns together came to $999,211,171.11.", PAYLOAD)
    assert not result.passed


def test_accepts_figures_present_in_the_payload():
    result = check("Spend was $604,398,923.71 over plan.", PAYLOAD)
    assert result.passed, result.message


def test_accepts_scaled_renderings():
    """Models write $604.4M, not 604398923.71. Both denote the same fact."""
    for rendering in ("$604.4M", "604.4 million", "$2.42B", "$2,419.3M"):
        assert check(f"The overrun was {rendering}.", PAYLOAD).passed, rendering


def test_accepts_ratio_written_as_percentage():
    """0.2292 in the payload may legitimately be written as 22.92%."""
    assert check("A 22.92% overrun.", PAYLOAD).passed


def test_accepts_magnitude_of_a_negative_fact():
    """Prose states magnitude and carries direction in words.

    The mix effect is stored as -495,473,475.79. "a $495.5M reduction" is correct
    FP&A writing; requiring the minus sign would reject it as fabrication.
    """
    assert check("offset by a $495.5M reduction in mix", PAYLOAD).passed


def test_accepts_small_integers_as_counts():
    """Counting things in the payload is not fabrication."""
    assert check("All 9 cost centers across 12 periods were reviewed.", PAYLOAD).passed


def test_thousands_separator_with_decimal_parses_as_one_number():
    """Regression: '$2,419.3M' once parsed as '2' then '419.3M'.

    419.3M is a figure nobody computed, so a valid draft was rejected for a number
    it never wrote.
    """
    result = check("Spend was $2,419.3M over plan.", PAYLOAD)
    assert result.passed
    assert result.checked == 1


def test_checks_every_prose_field_of_a_structured_draft():
    draft = {
        "headline": "Spend was $604.4M over plan.",
        "drivers": [{"cost_center": "Cloud", "comment": "An invented $12,345,678.90 appears here."}],
    }
    assert not check_draft(draft, PAYLOAD).passed


def test_collect_values_ignores_booleans():
    """Booleans are ints in Python; treating True as the figure 1 would be wrong."""
    assert 1.0 not in collect_values({"gate_passed": True, "other": 5000.0})


def test_collect_values_does_not_harvest_arbitrary_digits_from_strings():
    """Only years. Otherwise payload prose could launder an invented figure."""
    allowed = collect_values({"note": "reference 8675309 applies", "period_end": "2026-06-30"})
    assert 8675309.0 not in allowed
    assert 2026.0 in allowed


def test_assert_grounded_raises_on_fabrication():
    with pytest.raises(GroundingError):
        assert_grounded("An invented $777.7M figure.", PAYLOAD)
