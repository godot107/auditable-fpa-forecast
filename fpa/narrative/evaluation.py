"""Measuring the groundedness checker, because a validator is not exempt from validation.

``fpa.narrative.groundedness`` is the centerpiece of this project's claim that an LLM
can write finance copy safely. Until now it was supported by tests asserting it
catches *particular* fabrications. That proves it catches those. It says nothing
about its error rate, and two real bugs have already been found in it by hand.

The evaluation is built on a distinction that matters more than it first appears.
``check()`` reports ``checked``, the number of numerals the regex **matched**, so a
verdict only exists for numerals the parser saw. That gives three failure modes, not
two:

===================  =============================================  ==================
Mode                 What happens                                   Visible in output?
===================  =============================================  ==================
False rejection      a grounded figure is marked ungrounded          yes
False acceptance     a fabricated figure is judged grounded          yes
**Parse miss**       a numeral is never matched at all               **no**
===================  =============================================  ==================

The worst bug found in this module was the third kind. The regex ended with
``(?![\\w.])``, so ``$999.9M.`` at the end of a sentence never matched — a fabricated
figure passed not because it was verified but because it was **never checked**. An
accuracy metric computed over matched numerals would have scored that checker 100%
while it was blind. Silence is the failure mode, so silence needs its own metric.

Labels come from **construction rather than annotation**. Positives are composed
directly from payload values across the surface forms a model actually emits;
negatives are grounded drafts with exactly one numeral mutated in a controlled way.
Both are exact by construction, with no annotator and no judgement calls. This is
mutation testing pointed at a validator.

What this deliberately does **not** cover: whether the synthetic phrasing matches what
a real model writes. That needs a sample of genuine drafts, adjudicated once by hand
and frozen — a separate exercise, and the one place generalisation is actually tested.
Recorded here so the gap is not mistaken for coverage.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from fpa.narrative.groundedness import (
    MATCH_RTOL,
    MAX_FREE_INTEGER,
    _candidates,
    _NUMERAL,
    check,
    collect_values,
)

# Deliberately crude and over-inclusive: any digit run with optional separators.
# Its only job is to disagree with the real parser. Where a strict parser and a dumb
# one disagree is exactly where the strict one has gone blind, and finding that
# needs no labels at all.
_ANY_NUMERAL = re.compile(r"-?\$?\s?\d[\d,]*(?:\.\d+)?\s*(?:%|[a-zA-Z]{1,7})?")

# Surface forms a model plausibly emits for the same quantity. Each takes a value in
# units and returns prose. The awkward ones are the point: a figure at the end of a
# sentence, inside parentheses, or with a comma-grouped mantissa are all cases that
# have broken this parser before.
SURFACE_FORMS: dict[str, callable] = {
    "plain": lambda v: f"{v:,.1f}",
    "dollars_millions": lambda v: f"${v / 1e6:,.1f}M",
    "dollars_billions": lambda v: f"${v / 1e9:,.2f}B",
    "dollars_full": lambda v: f"${v:,.2f}",
    "million_word": lambda v: f"${v / 1e6:,.1f} million",
    "billion_lower": lambda v: f"{v / 1e9:,.2f}bn",
    "comma_grouped_scaled": lambda v: f"{v / 1e6:,.1f}M",
    "sentence_final": lambda v: f"${v / 1e6:,.1f}M.",
    "parenthesised": lambda v: f"(${v / 1e6:,.1f}M)",
    "trailing_comma": lambda v: f"${v / 1e6:,.1f}M,",
}

PERCENT_FORMS: dict[str, callable] = {
    "percent": lambda r: f"{r * 100:,.1f}%",
    "percent_sentence_final": lambda r: f"{r * 100:,.1f}%.",
}


@dataclass
class Case:
    """One labelled evaluation case."""

    text: str
    grounded: bool  # True = the checker must accept
    kind: str
    note: str = ""


@dataclass
class EvaluationReport:
    """Error rates for the groundedness checker."""

    n_positive: int
    n_negative: int
    false_rejections: list[Case] = field(default_factory=list)
    false_acceptances: list[Case] = field(default_factory=list)
    parse_misses: list[tuple[str, str]] = field(default_factory=list)
    n_candidates: int = 0
    n_matched: int = 0

    @property
    def false_rejection_rate(self) -> float:
        return len(self.false_rejections) / self.n_positive if self.n_positive else 0.0

    @property
    def false_acceptance_rate(self) -> float:
        return len(self.false_acceptances) / self.n_negative if self.n_negative else 0.0

    @property
    def parse_coverage(self) -> float:
        """Share of numeral-looking spans the strict parser actually inspected."""
        return self.n_matched / self.n_candidates if self.n_candidates else 1.0

    @property
    def clean(self) -> bool:
        """No fabrication accepted and nothing silently unparsed.

        A false *rejection* is a cost — a good draft thrown away. A false
        acceptance or a parse miss is a fabricated number reaching a reviewer,
        which is the failure this project exists to prevent. They are not
        weighted equally and the gate reflects that.
        """
        return not self.false_acceptances and not self.parse_misses


def claimed_values(text: str) -> list[float] | None:
    """What the checker will understand a single-numeral sentence to claim.

    Returns ``None`` unless the text contains exactly one numeral, because a case
    with two is not a controlled test of either.
    """
    matches = list(_NUMERAL.finditer(text))
    if len(matches) != 1:
        return None
    raw, scale, percent = matches[0].group(1), matches[0].group(2), matches[0].group(3)
    return _candidates(raw, scale, percent)


# Labelling thresholds, deliberately **independent of the checker's own tolerance**.
#
# The first version of this module labelled cases by running the checker's matching
# rule over them. That made the corpus definitionally consistent with the thing it
# was meant to test: positives passed by construction, negatives failed by
# construction, and the reported error rates were tautologies. A test that cannot
# fail is the exact failure this project warns about, arrived at while building the
# instrument meant to detect it.
#
# So labels now come from geometry instead. A case is only emitted if it sits
# unambiguously on one side:
#
#     |----positive----|         ambiguous         |----negative----|
#     0            2e-4        MATCH_RTOL 2e-3   2e-2            infinity
#              relative distance from the nearest payload value
#
# ``MATCH_RTOL`` sits inside the gap and is never consulted when labelling, so the
# checker is free to disagree with the corpus — which is what makes the resulting
# rates measurements rather than restatements. Cases in the ambiguous middle are not
# generated at all, because their correct label genuinely depends on which tolerance
# you pick, and inventing an answer there would be dishonest in either direction.
POSITIVE_PRECISION = 2e-4   # a rendering must round-trip at least this faithfully
NEGATIVE_MARGIN = 2e-2      # a fabrication must be at least this far from every fact


def _relative_distance(value: float, permitted) -> float:
    """Smallest relative distance from ``value`` to any permitted figure."""
    if not permitted:
        return float("inf")
    return min(
        abs(abs(value) - abs(p)) / max(abs(p), 1.0) for p in permitted
    )


def _is_faithful_positive(text: str, intended: float) -> bool:
    """True when the rendering still denotes ``intended`` to within formatting noise.

    Formatting rounds, and sometimes ruinously. ``$-0.50B`` renders -$495,473,475 at
    two decimal places — 0.9% from the figure it was meant to express. Labelling that
    as *grounded* would blame the checker for a defect in the corpus, so it is
    dropped instead.

    The test is whether the rendered string round-trips to within
    ``POSITIVE_PRECISION``, which is a property of the formatting alone. The checker's
    tolerance is ten times looser and plays no part in the decision.
    """
    claimed = claimed_values(text)
    if claimed is None:
        return False
    return _relative_distance(intended, claimed) <= POSITIVE_PRECISION


def _is_faithful_negative(text: str, permitted) -> bool:
    """True when no reading of the rendering is anywhere near a real figure.

    Stricter than checking the intended mutation alone, for two reasons.
    ``_candidates`` treats a scale suffix as optional, so ``$199.7B`` also claims the
    bare 199.7 — and if *that* appears in the payload the case is grounded after all,
    whatever the generator intended. And a free integer is waved through as a count
    rather than a claim, so it cannot serve as a negative either.

    ``NEGATIVE_MARGIN`` is ten times the checker's tolerance, so a case only counts as
    a fabrication when it would be one under any reasonable tolerance.
    """
    claimed = claimed_values(text)
    if claimed is None:
        return False
    first = claimed[0]
    if float(first).is_integer() and abs(first) <= MAX_FREE_INTEGER and len(claimed) == 1:
        return False
    return all(_relative_distance(value, permitted) > NEGATIVE_MARGIN for value in claimed)


def _sample_values(payload, *, limit: int = 40) -> list[float]:
    """Payload values big enough that a citation of them is a real claim.

    Small integers are skipped because ``check`` deliberately waves them through as
    counts rather than claims (``MAX_FREE_INTEGER``), so they cannot discriminate
    between a working checker and a broken one.
    """
    values = sorted(
        v for v in collect_values(payload) if abs(v) > MAX_FREE_INTEGER and abs(v) < 1e13
    )
    if len(values) <= limit:
        return values
    step = len(values) / limit
    return [values[int(i * step)] for i in range(limit)]


def compose_positives(payload) -> list[Case]:
    """Grounded prose, one case per (value, surface form).

    Every case cites a figure that is in the payload, so a correct checker accepts
    all of them. Anything rejected here is a false rejection — a real cost, because
    it means a valid draft was thrown away over a number the model was entitled to
    write.
    """
    cases: list[Case] = []
    values = _sample_values(payload)

    for value in values:
        for name, render in SURFACE_FORMS.items():
            text = f"Cloud Infrastructure came in at {render(value)} for the period."
            if not _is_faithful_positive(text, value):
                continue  # the formatting, not the checker, lost the value
            cases.append(Case(text=text, grounded=True, kind=f"positive:{name}"))

    # Ratios written as percentages — a documented allowance in ``collect_values``.
    for ratio in sorted(r for r in collect_values(payload) if 0.001 < abs(r) < 1.0)[:12]:
        for name, render in PERCENT_FORMS.items():
            text = f"That is {render(ratio)} against plan."
            if _is_faithful_positive(text, ratio):
                cases.append(Case(text=text, grounded=True, kind=f"positive:{name}"))

    # Values inside the tolerance band must still be accepted: prose rounds, and a
    # checker that demanded exactness would reject correct writing.
    for value in values[:10]:
        nudged = value * (1 + MATCH_RTOL * 0.4)
        text = f"Spend reached ${nudged / 1e6:,.1f}M."
        if _is_faithful_positive(text, value):
            cases.append(
                Case(
                    text=text,
                    grounded=True,
                    kind="positive:within_tolerance",
                    note="rounded, inside MATCH_RTOL",
                )
            )

    return cases


def compose_negatives(payload, *, seed: int = 42) -> list[Case]:
    """Fabrications the checker must reject, each mutated from a grounded figure.

    Four mutation kinds, chosen because each probes a different mechanism:

    * ``digit`` — a value just outside ``MATCH_RTOL``. Probes the tolerance.
    * ``magnitude`` — the right mantissa at the wrong scale. Probes ``_SCALES``.
    * ``derived`` — a value that looks computed (a sum, a ratio of two real figures)
      but appears nowhere in the payload. The realistic case, and the hardest.
    * ``fabricated`` — an invented round number in plausible prose.
    """
    rng = random.Random(seed)
    allowed = collect_values(payload)
    values = _sample_values(payload)
    cases: list[Case] = []

    def emit(text: str, kind: str, note: str = "") -> None:
        """Add a case only if *every* reading of it is genuinely ungrounded."""
        if _is_faithful_negative(text, allowed):
            cases.append(Case(text=text, grounded=False, kind=kind, note=note))

    for value in values:
        # Well outside the tolerance band, but visually close — the kind of typo or
        # hallucinated digit a human reviewer skims past.
        mutated = value * 1.05
        emit(
            f"Cloud Infrastructure came in at ${mutated / 1e6:,.1f}M for the period.",
            "negative:digit",
        )

        # Right mantissa, wrong scale: "$604.4B" where the payload holds $604.4M.
        if abs(value) < 1e10:
            emit(
                f"The overrun reached ${value / 1e6:,.1f}B against plan.",
                "negative:magnitude",
                "mantissa correct, scale wrong",
            )

    # Plausible-looking derivations of two real figures that nobody computed. This is
    # the mutation that most resembles what a model actually does wrong: it does
    # arithmetic in prose instead of reading the number it was given.
    for _ in range(30):
        first, second = rng.sample(values, 2) if len(values) >= 2 else (0.0, 0.0)
        for label, derived in (("sum", first + second), ("difference", first - second)):
            if derived:
                emit(
                    f"Together the two centers account for ${abs(derived) / 1e6:,.1f}M.",
                    f"negative:derived_{label}",
                    "arithmetic on real figures, absent from the payload",
                )

    # Round invented numbers in confident prose — the classic hallucination shape.
    for invented in (999.9, 1234.5, 4200.0, 87.6, 3141.6):
        emit(
            f"Marketing overran by ${invented:,.1f}M.",
            "negative:fabricated_sentence_final",
            "the shape of the bug that once passed unchecked",
        )
        emit(
            f"Marketing overran by ${invented:,.1f}M, a material miss.",
            "negative:fabricated_mid_sentence",
        )

    return cases


def parse_misses(text: str) -> list[str]:
    """Numeral-looking spans the strict parser did not inspect.

    Found by disagreement with a deliberately over-inclusive parser rather than by
    labelling. Any span the crude regex finds that the strict one never covered is a
    numeral with **no verdict at all** — neither accepted nor rejected, simply
    invisible. That is how ``$999.9M.`` slipped through, and it is the only failure
    mode an accuracy metric cannot see.

    Compared by **character span**, not by count. Equal counts can still mean the two
    parsers matched different things, and a miss that coincides with a spurious extra
    match would cancel out and read as clean.
    """
    covered: set[int] = set()
    for match in _NUMERAL.finditer(text):
        covered.update(range(match.start(1), match.end(1)))

    missed: list[str] = []
    for candidate in _ANY_NUMERAL.finditer(text):
        digits = [i for i in range(candidate.start(), candidate.end()) if text[i].isdigit()]
        if digits and not any(i in covered for i in digits):
            missed.append(candidate.group(0).strip())
    return missed


def evaluate(cases: list[Case], payload) -> EvaluationReport:
    """Score the checker over a labelled corpus.

    Returns rates rather than a pass/fail, because the two error directions are not
    equally bad and collapsing them into one number would hide which is which.
    """
    report = EvaluationReport(
        n_positive=sum(1 for c in cases if c.grounded),
        n_negative=sum(1 for c in cases if not c.grounded),
    )

    for case in cases:
        result = check(case.text, payload)

        if case.grounded and not result.passed:
            report.false_rejections.append(case)
        elif not case.grounded and result.passed:
            report.false_acceptances.append(case)

        # Coverage is measured on every case regardless of its label: a numeral the
        # parser never saw is a defect whether the surrounding draft was honest or not.
        report.n_matched += result.checked
        missed = parse_misses(case.text)
        report.n_candidates += result.checked + len(missed)
        for span in missed:
            report.parse_misses.append((case.text, span))

    return report


def build_corpus(payload, *, seed: int = 42) -> list[Case]:
    """The full synthetic corpus: composed positives plus mutated negatives."""
    return compose_positives(payload) + compose_negatives(payload, seed=seed)


def report_markdown(report: EvaluationReport) -> str:
    """Render the evaluation, leading with the direction that matters."""
    lines = [
        "## Groundedness checker — measured error rates",
        "",
        "Labels are generated by construction, not annotation: positives are composed "
        "from payload values across the surface forms a model actually emits, negatives "
        "are grounded drafts with exactly one numeral mutated. Mutation testing, pointed "
        "at a validator.",
        "",
        "| metric | value | why it matters |",
        "|---|---|---|",
        f"| **False acceptance rate** | **{report.false_acceptance_rate:.2%}** "
        f"({len(report.false_acceptances)}/{report.n_negative}) | "
        "A fabricated figure reaching a reviewer. The failure this project exists to prevent. |",
        f"| False rejection rate | {report.false_rejection_rate:.2%} "
        f"({len(report.false_rejections)}/{report.n_positive}) | "
        "A valid draft thrown away. A cost, not a safety failure. |",
        f"| **Parse coverage** | **{report.parse_coverage:.2%}** "
        f"({report.n_matched:,}/{report.n_candidates:,}) | "
        "Numerals the parser actually inspected. Anything below 100% is a figure with "
        "*no verdict at all*. |",
        "",
    ]

    if report.clean:
        lines.append(
            "**No fabrication accepted, and no numeral left uninspected** across "
            f"{report.n_positive + report.n_negative:,} cases."
        )
    else:
        lines.append("**Failures found:**")
        for case in report.false_acceptances[:8]:
            lines.append(f"- accepted a fabrication (`{case.kind}`): {case.text}")
        for text, span in report.parse_misses[:8]:
            lines.append(f"- never inspected `{span}` in: {text}")

    if report.false_rejections:
        kinds = sorted({c.kind for c in report.false_rejections})
        lines += [
            "",
            f"_Rejected {len(report.false_rejections)} grounded case(s), in: "
            + ", ".join(f"`{k}`" for k in kinds)
            + ". Each is a surface form a model may legitimately write and this checker "
            "would refuse — worth knowing, and less dangerous than the alternative._",
        ]

    lines += [
        "",
        "**What this does not measure.** Whether synthetic phrasing matches what a real "
        "model writes. That needs a sample of genuine drafts adjudicated by hand, and is "
        "the only thing that tests generalisation. Recorded as an open gap rather than "
        "counted as coverage.",
    ]
    return "\n".join(lines)
