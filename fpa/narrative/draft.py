"""Generate a commentary draft and gate it on groundedness.

This is where the guarantee is actually enforced. A draft that cites an ungrounded
figure never reaches a human — it is rejected and, optionally, regenerated once. The
rejection is itself recorded, because "how often does the model try to make something
up" is a number a finance team should be able to see.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from fpa.narrative.groundedness import GroundednessResult, check_draft
from fpa.narrative.provider import PROMPT_VERSION, NarrativeProvider

logger = logging.getLogger(__name__)


@dataclass
class DraftResult:
    """A commentary draft plus the evidence about whether it may be shown."""

    draft: dict | None
    grounded: GroundednessResult
    provider: str
    prompt_version: str = PROMPT_VERSION
    attempts: int = 1
    rejected_drafts: list[dict] = field(default_factory=list)
    error: str | None = None

    @property
    def publishable(self) -> bool:
        """A draft may be shown to a reviewer only if every figure traces to the facts."""
        return self.draft is not None and self.grounded.passed and self.error is None


def generate_draft(
    facts: dict, provider: NarrativeProvider, *, retries: int = 1
) -> DraftResult:
    """Generate commentary, rejecting any draft that cites an ungrounded figure.

    One retry by default: a model that hallucinates a figure will often not do so
    twice, and a single retry is cheap. It is not retried indefinitely — a provider
    that cannot stay grounded should surface as a failure, not be papered over.
    """
    rejected: list[dict] = []
    last: GroundednessResult | None = None

    for attempt in range(1, retries + 2):
        try:
            draft = provider.generate(facts)
        except Exception as exc:
            logger.error("narrative provider %s failed: %s", provider.name, exc)
            return DraftResult(
                draft=None,
                grounded=GroundednessResult(passed=False, checked=0),
                provider=provider.name,
                attempts=attempt,
                rejected_drafts=rejected,
                error=str(exc),
            )

        result = check_draft(draft, facts)
        last = result
        if result.passed:
            return DraftResult(
                draft=draft,
                grounded=result,
                provider=provider.name,
                attempts=attempt,
                rejected_drafts=rejected,
            )

        logger.warning(
            "draft rejected (attempt %d): %s", attempt, result.message
        )
        rejected.append(draft)

    return DraftResult(
        draft=None,
        grounded=last or GroundednessResult(passed=False, checked=0),
        provider=provider.name,
        attempts=retries + 1,
        rejected_drafts=rejected,
        error="all attempts cited ungrounded figures",
    )
