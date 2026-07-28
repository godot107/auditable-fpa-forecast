"""Append-only approval log — the human-in-the-loop record.

Records who decided what, when, on exactly which draft, and which model and prompt
version produced it. Append-only JSONL: entries are never rewritten, because an audit
trail that can be edited is not one.

The vocabulary is lowercase ``approved`` / ``rejected`` / ``edited``, matching what
``fpa.kpi.process`` reads to compute the commentary approval rate. ``edited`` is a
distinct outcome on purpose — a reviewer who rewrites every draft before approving it
is telling you the drafting does not work, and collapsing that into "approved" hides
exactly the signal worth having.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ACTIONS = ("approved", "rejected", "edited", "auto_rejected")

LOG_NAME = "approval_log.jsonl"


def get_log_file(audit_dir: Path) -> Path:
    audit_dir = Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    return audit_dir / LOG_NAME


def log_decision(
    audit_dir: Path,
    *,
    user: str,
    action: str,
    period: str,
    draft: dict | None,
    provider: str,
    prompt_version: str,
    grounded: bool,
    note: str | None = None,
) -> dict:
    """Append one decision. Returns the entry written."""
    if action not in ACTIONS:
        raise ValueError(f"action must be one of {ACTIONS}, got {action!r}")

    entry = {
        # UTC and explicit: an audit trail spanning timezones is not an audit trail.
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user": user,
        "action": action,
        "period": period,
        "provider": provider,
        "prompt_version": prompt_version,
        "grounded": bool(grounded),
        "draft": draft,
        "note": note,
    }

    with get_log_file(audit_dir).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, default=str) + "\n")
    return entry


def read_log(audit_dir: Path) -> pd.DataFrame:
    """Read the log. Returns an empty frame with the right columns if absent."""
    path = Path(audit_dir) / LOG_NAME
    columns = ["timestamp", "user", "action", "period", "provider", "prompt_version", "grounded"]
    if not path.exists():
        return pd.DataFrame(columns=columns)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], format="mixed", utc=True)
    return frame.sort_values("timestamp", ascending=False).reset_index(drop=True)
