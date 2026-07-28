"""Factual-consistency checking for LLM output — the anti-hallucination guarantee.

**The rule: the model writes the commentary, never the number.**

Every figure is computed in Python and handed to the model as a facts payload. The
model may describe those figures in prose. If any numeral it returns does not
correspond to a value in the payload, the draft is **rejected** — not flagged, not
footnoted, rejected.

This is factual-consistency checking, the standard NLG approach to hallucination
detection (Huyen, *AI Engineering*, pp.219-225). It is stronger than RAG here in one
specific way: the reference is **computed**, not retrieved, so the reference itself
cannot be wrong. A RAG system can faithfully cite a bad document; a figure here either
equals something the pipeline calculated or it does not exist.

Three things make the difference between a check that works and one that rejects
everything:

* **Relative tolerance.** These are billion-dollar figures. An absolute tolerance of a
  couple of cents means ``$604.4M`` never matches ``604_432_117.88`` and every draft
  fails.
* **Scale suffixes.** Models write ``$604.4M``, not ``604432117.88``. ``604.4`` alone
  appears nowhere in the payload; ``604.4 x 1e6`` does.
* **Percent forms.** A ratio stored as ``0.229`` is written as ``22.9%``.

Bare small integers pass without matching: counting the things in the payload ("all
nine cost centers", "the top three") is not fabrication.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_FREE_INTEGER = 100
MATCH_RTOL = 2e-3  # generous enough for "$604.4M", tight enough to reject "$610M"

_SCALES = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "mm": 1e6, "million": 1e6,
    "bn": 1e9, "b": 1e9, "billion": 1e9,
}

_NUMERAL = re.compile(
    # The grouped alternative must allow a decimal tail. Without it "2,419.3M"
    # matches as "2" and then "419.3M" — and 419.3M is a figure nobody computed,
    # so a perfectly valid draft gets rejected for a number it never wrote.
    r"""(?<![\w.])
        \$?\s?
        (-?\d{1,3}(?:,\d{3})+(?:\.\d+)? | -?\d+(?:\.\d+)?)
        \s*
        (thousand|million|billion|mm|bn|k|m|b)?
        \s*
        (%)?
        # Reject only a period that begins another number ("1.2.3"), not one that
        # ends a sentence. A blanket (?![\w.]) meant "$999.9M." never matched at
        # all — so a fabricated figure passed by never being checked, which is the
        # worst possible failure mode for a validator.
        (?!\w)(?!\.\d)
    """,
    re.IGNORECASE | re.VERBOSE,
)


class GroundingError(ValueError):
    """Raised when a draft cites a number absent from the facts payload."""


@dataclass
class GroundednessResult:
    passed: bool
    ungrounded: list[str] = field(default_factory=list)
    checked: int = 0

    @property
    def message(self) -> str:
        if self.passed:
            return f"all {self.checked} figure(s) trace to the computed facts"
        return (
            f"{len(self.ungrounded)} of {self.checked} figure(s) absent from the facts "
            f"payload: {', '.join(self.ungrounded[:6])}"
        )


def collect_values(payload) -> set[float]:
    """Recursively gather every numeric value the model is permitted to cite."""
    allowed: set[float] = set()

    def walk(node) -> None:
        if isinstance(node, bool):
            return  # bools are ints in Python, but they are not figures
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)
        elif isinstance(node, (int, float)):
            allowed.add(float(node))
        elif isinstance(node, str):
            # Only years, deliberately. Harvesting every digit out of arbitrary
            # strings would let the payload's own text launder an invented figure.
            for year in re.findall(r"\b((?:19|20)\d{2})\b", node):
                allowed.add(float(year))

    walk(payload)

    # A ratio may legitimately be written as a percentage.
    for value in list(allowed):
        if -10.0 < value < 10.0:
            allowed.add(value * 100.0)
    return allowed


def _candidates(raw: str, scale: str | None, percent: str | None) -> list[float]:
    """Every value a rendered numeral could plausibly denote."""
    magnitude = float(raw.replace(",", ""))
    values = [magnitude]
    if scale:
        values.append(magnitude * _SCALES[scale.lower()])
    if percent:
        values.append(magnitude / 100.0)  # "22.9%" may be stored as the ratio 0.229
    return values


def check(text: str, payload) -> GroundednessResult:
    """Verify every numeral in ``text`` corresponds to a value in ``payload``."""
    allowed = collect_values(payload)
    ungrounded: list[str] = []
    checked = 0

    for match in _NUMERAL.finditer(text):
        raw, scale, percent = match.group(1), match.group(2), match.group(3)
        checked += 1
        values = _candidates(raw, scale, percent)

        if (
            not scale
            and not percent
            and float(values[0]).is_integer()
            and abs(values[0]) <= MAX_FREE_INTEGER
        ):
            continue  # a count, not a claim

        # Matched on magnitude, not signed value. The payload holds the signed truth
        # (Licensed Content's mix effect is -$495.5M); prose conventionally states the
        # magnitude and carries direction in words — "a $495.5M reduction". Requiring
        # the sign to match rejects correct writing as fabrication. Magnitude matching
        # still means a figure absent from the payload cannot appear.
        if any(
            abs(abs(candidate) - abs(permitted)) <= MATCH_RTOL * max(abs(permitted), 1.0)
            for candidate in values
            for permitted in allowed
        ):
            continue

        ungrounded.append(match.group(0).strip())

    return GroundednessResult(passed=not ungrounded, ungrounded=ungrounded, checked=checked)


def check_draft(draft, payload) -> GroundednessResult:
    """Check every prose field of a structured draft."""
    texts: list[str] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            texts.append(node)

    walk(draft)
    return check("\n".join(texts), payload)


def assert_grounded(text: str, payload) -> bool:
    """Raise :class:`GroundingError` unless every figure in ``text`` is grounded."""
    result = check(text, payload)
    if not result.passed:
        raise GroundingError(result.message)
    return True
