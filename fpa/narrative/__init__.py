"""Narrative layer — LLM-drafted variance commentary under a groundedness guarantee.

The governing rule, borrowed from this workspace's ``energy-batch-trader`` (where the
LLM may only run the anomaly gate, never decide a trade): **the model writes the
commentary, never the number.** Figures are computed in Python, handed over as a facts
payload, and every numeral in the returned prose is checked back against it.

The output is a *draft*. A human approves or rejects it, and that decision is written
to an append-only audit log with the model and prompt version attached.
"""

from fpa.narrative.facts import build_facts_payload, from_pipeline, to_json
from fpa.narrative.groundedness import (
    GroundednessResult,
    GroundingError,
    assert_grounded,
    check,
    check_draft,
)
from fpa.narrative.provider import (
    NARRATIVE_SCHEMA,
    PROMPT_VERSION,
    AnthropicProvider,
    ClaudeCodeProvider,
    FixtureProvider,
    NarrativeProvider,
    get_provider,
)
from fpa.narrative.draft import DraftResult, generate_draft

__all__ = [
    "NARRATIVE_SCHEMA",
    "PROMPT_VERSION",
    "AnthropicProvider",
    "ClaudeCodeProvider",
    "DraftResult",
    "FixtureProvider",
    "GroundednessResult",
    "GroundingError",
    "NarrativeProvider",
    "assert_grounded",
    "build_facts_payload",
    "check",
    "check_draft",
    "from_pipeline",
    "generate_draft",
    "get_provider",
    "to_json",
]
