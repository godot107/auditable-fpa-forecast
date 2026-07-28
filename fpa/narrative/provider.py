"""Narrative providers — where the commentary text comes from.

Three implementations behind one Protocol, so swapping the engine is a one-line
change and nothing downstream knows the difference:

* :class:`ClaudeCodeProvider` — headless ``claude -p``. No API key: it uses the
  existing Claude Code authentication.
* :class:`FixtureProvider` — deterministic, offline, no subprocess. Composed from the
  facts payload itself, so it is grounded by construction and the demo works with no
  network.
* :class:`AnthropicProvider` — the documented seam for moving to the API later.

The output schema caps length on purpose. Fewer generated tokens means less surface to
fabricate on (Huyen, *AI Engineering*, pp.219-225), and a variance comment that runs
past three sentences is not being read by anyone anyway.
"""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Protocol

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v1"

# Bounded by construction: maxLength on every prose field, a capped array.
NARRATIVE_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["headline", "drivers"],
    "properties": {
        "headline": {
            "type": "string",
            "maxLength": 240,
            "description": "One sentence on the total variance and whether it is favourable.",
        },
        "drivers": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["cost_center", "comment"],
                "properties": {
                    "cost_center": {"type": "string", "maxLength": 80},
                    "comment": {
                        "type": "string",
                        "maxLength": 320,
                        "description": "One or two sentences explaining this variance.",
                    },
                },
            },
        },
        "watch_item": {
            "type": "string",
            "maxLength": 240,
            "description": "Optional: the single thing to watch next period.",
        },
    },
}

SYSTEM_RULE = (
    "You are an FP&A analyst writing variance commentary for a CFO review.\n"
    "\n"
    "ABSOLUTE RULE: every number you write MUST appear in the facts payload below. "
    "Do not compute, infer, round differently, or estimate any figure. Do not add "
    "percentages that are not given. If you want to say something you have no number "
    "for, say it qualitatively without a number.\n"
    "\n"
    "Explain WHY variances happened using the spend_effect and mix_effect breakdown: "
    "spend_effect is the part explained by total spend moving, mix_effect is the part "
    "explained by budget share shifting between cost centers. Be concise and concrete. "
    "Return JSON matching the schema."
)


class NarrativeProvider(Protocol):
    """Anything that can turn a facts payload into a structured draft."""

    name: str

    def generate(self, facts: dict) -> dict: ...


def _build_prompt(facts: dict) -> str:
    return f"{SYSTEM_RULE}\n\nFACTS PAYLOAD:\n{json.dumps(facts, indent=2, default=str)}\n"


class ClaudeCodeProvider:
    """Generate commentary via the headless Claude Code CLI."""

    name = "claude-code"

    def __init__(self, model: str = "sonnet", timeout: int = 180) -> None:
        self.model = model
        self.timeout = timeout

    def generate(self, facts: dict) -> dict:
        prompt = _build_prompt(facts)

        # The schema is passed inline — `--json-schema` parses its argument as JSON,
        # not as a path. It is ~1KB so argv is fine. The *prompt* goes over stdin,
        # because it carries the whole facts payload and would risk ARG_MAX; a
        # truncated prompt fails in a way that looks like a model problem.
        try:
            completed = subprocess.run(
                [
                    "claude", "-p",
                    "--output-format", "json",
                    "--json-schema", json.dumps(NARRATIVE_SCHEMA),
                    "--model", self.model,
                ],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Claude Code CLI not found on PATH.") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"claude -p failed: {exc.stderr[:400]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"claude -p timed out after {self.timeout}s") from exc

        return self._unwrap(completed.stdout)

    @staticmethod
    def _unwrap(stdout: str) -> dict:
        """Pull the model's JSON out of Claude Code's response envelope.

        ``--output-format json`` returns the CLI's own envelope (session id, cost,
        and the model text under ``result``), not the bare object. Returning stdout
        directly hands callers the envelope instead of the commentary.
        """
        envelope = json.loads(stdout)
        payload = envelope.get("result", envelope) if isinstance(envelope, dict) else envelope
        if isinstance(payload, str):
            payload = json.loads(payload)
        return payload


def _money(value: float) -> str:
    """Render a dollar figure at a sensible scale — $2.4B reads better than $2,419.3M."""
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.2f}B"
    return f"${value / 1e6:,.1f}M"


class FixtureProvider:
    """Deterministic offline commentary composed from the facts payload.

    Not a canned string. A hard-coded fixture would cite numbers from whenever it was
    written, which by definition are absent from the current payload — so the
    groundedness check would reject it and offline mode would never work. This
    composes its figures out of the payload it was handed, so it is grounded by
    construction and reproducible.
    """

    name = "fixture"

    def generate(self, facts: dict) -> dict:
        variance = facts.get("variance", {}) or {}
        worst = (variance.get("largest_unfavourable") or [{}])[0]
        total = variance.get("total_variance")
        pct = variance.get("total_variance_pct")

        direction = "above" if (total or 0) > 0 else "below"
        headline = (
            f"Operating spend finished {direction} plan by {_money(abs(total or 0))} "
            f"({abs(pct or 0) * 100:.1f}% of budget) across {variance.get('n_periods', 0)} periods."
        )

        drivers = []
        for row in (variance.get("largest_unfavourable") or [])[:2]:
            spend = row.get("spend_effect", 0.0)
            mix = row.get("mix_effect", 0.0)
            lead = "a shift in budget mix toward it" if abs(mix) > abs(spend) else "overall spend growth"
            drivers.append(
                {
                    "cost_center": row.get("cost_center", "unknown"),
                    "comment": (
                        f"Came in {_money(row.get('variance', 0))} over plan "
                        f"({row.get('variance_pct', 0) * 100:.1f}%), driven mainly by {lead} "
                        f"(mix effect {_money(mix)} vs spend effect {_money(spend)})."
                    ),
                }
            )

        draft = {"headline": headline, "drivers": drivers or [{"cost_center": "n/a", "comment": "No variance data."}]}
        if worst:
            draft["watch_item"] = (
                f"{worst.get('cost_center')} carries the largest unfavourable variance "
                f"and should be reviewed before the next planning cycle."
            )
        return draft


class AnthropicProvider:
    """Direct Anthropic API. The documented seam for moving off the CLI.

    Intentionally unimplemented rather than half-built: the interface is what matters,
    and everything around it — the facts payload, the schema, the groundedness check,
    the approval log — is provider-agnostic already. Wiring this up is one class, and
    the guardrails do not change.
    """

    name = "anthropic"

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-5") -> None:
        self.api_key = api_key
        self.model = model

    def generate(self, facts: dict) -> dict:
        raise NotImplementedError(
            "AnthropicProvider is the documented API seam and is not implemented. "
            "Use FPA_NARRATIVE_PROVIDER=claudecode (headless CLI, no API key) or "
            "=fixture (offline)."
        )


def get_provider(name: str, *, model: str = "sonnet") -> NarrativeProvider:
    """Resolve a provider by configured name."""
    providers = {
        "claudecode": lambda: ClaudeCodeProvider(model=model),
        "claude-code": lambda: ClaudeCodeProvider(model=model),
        "fixture": FixtureProvider,
        "anthropic": AnthropicProvider,
    }
    if name not in providers:
        raise ValueError(f"unknown narrative provider {name!r}; expected one of {sorted(providers)}")
    return providers[name]()
