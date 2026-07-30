"""Measure the checker against prose a real model actually wrote.

``fpa.narrative.evaluation`` scores the groundedness checker on 364 **synthetic** cases and
reports 0% false acceptance with 100% parse coverage. Its own report has always carried the
limitation verbatim: *"whether synthetic phrasing matches what a real model writes"* is not
measured, and that is the only thing testing generalisation. A generated corpus can only
contain the phrasings its generator thought of, so a model writing "roughly six hundred
million" in words, or "$604.4 million" spelled out, would defeat every regex in the checker
and the synthetic score would never notice.

This closes that gap in the one direction that can be measured without circularity.

**What is measured: parse coverage.** For every numeral in a real draft, does the checker's
strict parser give a verdict at all? A numeral it fails to match is not judged wrong — it is
*not judged*, which is the failure mode the accept/reject rates are structurally blind to and
the one a fabricated figure would exploit. Disagreement with a deliberately over-inclusive
second parser finds those spans.

**What is not measured: accept/reject accuracy on real drafts.** Ground truth for that would
have to come from reading each numeral and deciding whether the payload supports it. Doing
that with the checker's own matching rule is exactly the tautology
``fpa.narrative.evaluation`` was rewritten to remove, and doing it by hand is a human task
this module cannot honestly automate. So it is left open and said so, rather than measured
badly.

Drafts are generated once and **frozen** in ``data/real_drafts.<vintage>.json`` so the
measurement is reproducible without re-running a model, and so a reader can see the prose
that produced the number.
"""

from __future__ import annotations

import json
import re
import logging
from dataclasses import dataclass

from fpa.config import Settings

logger = logging.getLogger(__name__)


def corpus_path(settings: Settings):
    return settings.data_dir / f"real_drafts.{settings.data_vintage}.json"


@dataclass(frozen=True)
class ParseReport:
    """Parse coverage over real model prose."""

    drafts: int
    numerals: int
    unparsed: tuple[str, ...]

    @property
    def coverage(self) -> float:
        return 1.0 - len(self.unparsed) / self.numerals if self.numerals else 1.0

    @property
    def clean(self) -> bool:
        return not self.unparsed


def _prose(draft: dict) -> list[str]:
    """Every free-text field a model filled in. Numbers can hide in any of them."""
    texts = [draft.get("headline", ""), draft.get("watch_item", "") or ""]
    texts += [item.get("comment", "") for item in draft.get("drivers", [])]
    return [t for t in texts if t]


def generate(settings: Settings, result, report, *, count: int = 12, model: str = "sonnet") -> list[dict]:
    """Ask a real model for ``count`` drafts, varying the payload so the prose varies.

    The variance window is changed between calls rather than the prompt: different periods
    produce genuinely different figures, so the model has different numbers to phrase and the
    corpus does not collapse into one sentence rewritten.
    """
    from fpa.narrative.facts import from_pipeline
    from fpa.narrative.provider import get_provider
    from fpa.variance import build_variance_report

    provider = get_provider("claudecode", model=model)
    drafts: list[dict] = []

    for index in range(count):
        periods = 3 + index  # 3..14 trailing months
        varied = build_variance_report(
            result.ledger, result.budget, result.revenue, result.revenue_budget,
            result.drivers, periods=periods,
        )
        facts = from_pipeline(result, varied)
        try:
            draft = provider.generate(facts)
        except Exception as exc:  # noqa: BLE001 - one bad call must not lose the batch
            logger.warning("draft %d failed: %s", index, exc)
            continue

        drafts.append({"periods": periods, "draft": draft, "facts": facts})
        logger.info("draft %d/%d (%d trailing months)", index + 1, count, periods)

    return drafts


def parse_report(drafts: list[dict]) -> ParseReport:
    """Parse coverage across every prose field of every draft."""
    from fpa.narrative.evaluation import parse_misses
    from fpa.narrative.groundedness import _NUMERAL

    numerals = 0
    unparsed: list[str] = []

    for entry in drafts:
        for text in _prose(entry["draft"]):
            numerals += len(list(_NUMERAL.finditer(text)))
            unparsed.extend(parse_misses(text))

    return ParseReport(drafts=len(drafts), numerals=numerals, unparsed=tuple(unparsed))


def save(settings: Settings, drafts: list[dict]) -> None:
    from fpa.narrative.facts import to_json

    path = corpus_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(drafts), encoding="utf-8")


def load(settings: Settings) -> list[dict] | None:
    path = corpus_path(settings)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


_WORD_NUMERAL = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|"
    r"forty|fifty|hundred|thousand|million|billion)\b(?!\s*(?:periods?|quarters?|months?|"
    r"controls?|series|cost centers?))",
    re.IGNORECASE,
)


def phrasing_census(drafts: list[dict]) -> dict[str, int]:
    """How the model actually wrote its numbers.

    The coverage figure is only as strong as the phrasings that occurred. A corpus of pure
    digit-form numerals proves the regex handles digits, not that it handles prose.
    """
    from fpa.narrative.groundedness import _NUMERAL

    digits = words = 0
    for entry in drafts:
        for text in _prose(entry["draft"]):
            digits += len(list(_NUMERAL.finditer(text)))
            words += len(_WORD_NUMERAL.findall(text))
    return {"digit_form": digits, "word_form": words}


def report_markdown(report: ParseReport, census: dict[str, int] | None = None) -> str:
    lines = [
        "## Checker parse coverage on real model prose",
        "",
        f"**{report.drafts} drafts** from the `claudecode` provider, "
        f"**{report.numerals} numerals**, parse coverage "
        f"**{report.coverage:.1%}**.",
        "",
        "The synthetic corpus reports 100% parse coverage over 364 generated cases, but a "
        "generator can only produce the phrasings it was written to produce. This measures "
        "the same thing on prose a model actually wrote.",
    ]

    if report.unparsed:
        lines += [
            "",
            f"**{len(report.unparsed)} span(s) the checker gave no verdict on:**",
            "",
            *[f"- `{span}`" for span in report.unparsed[:20]],
        ]

    # The result is easy to overstate, so the phrasing census goes next to it. 100% coverage
    # over digit-form numerals says the regex handles digits — nothing more.
    if census:
        digits, words = census["digit_form"], census["word_form"]
        lines += [
            "",
            "### How strong is that, actually",
            "",
            f"| phrasing | count |",
            f"|---|---|",
            f"| digit form — `$553,358,946.90`, `6.94%`, `$525.49M` | **{digits}** |",
            f"| word form — *six hundred million* | **{words}** |",
            "",
        ]
        if words == 0:
            lines.append(
                "**The hard case did not occur, so it is untested rather than passed.** Every "
                "numeral this model wrote was digit form. A draft saying *\"roughly six hundred "
                "million\"* would defeat the regex entirely, and none appeared — which is a "
                "fact about the model's output, not evidence about the checker."
            )
        else:
            lines.append(
                f"**{words} word-form numeral(s) appeared** and the parser still covered "
                "everything, which is the stronger version of this result."
            )
        lines += [
            "",
            "There is a reason digits dominate, and it is a design decision rather than luck: "
            "the provider constrains output with a **JSON schema** and caps length, which is "
            "Huyen's second hallucination mitigation (*AI Engineering*, pp.219–225). A bounded, "
            "structured field pushes a model toward compact figures. So the schema is carrying "
            "part of the load the regex appears to carry.",
        ]

    lines += [
        "",
        "**Why parse coverage and not accept/reject.** A numeral the strict parser fails to "
        "match gets *no verdict* — it is absent from `checked` rather than judged wrong — so "
        "the accept and reject rates are structurally blind to it. That silence is what a "
        "fabricated figure would exploit, and it is measurable here without circularity. "
        "Accept/reject accuracy on real drafts would need ground truth from reading each "
        "numeral by hand; deriving it from the checker's own rule is the tautology "
        "`evaluation.py` was rewritten to remove. **That half stays open.**",
    ]
    return "\n".join(lines)
