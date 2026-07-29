"""Reader for the recorded ERP balance-constraint refusal.

Deliberately separate from :mod:`fpa.ledger.odoo_load`, and deliberately thin: ``json``
and a path, nothing else. The seeder that *produces* this record needs an XML-RPC client,
a live Odoo and pandas; the app that *reads* it needs none of those and runs on a host
where no ERP exists.

Coupling the two cost a broken page on the deployed app — importing the loader dragged in
the whole seeder, and an ImportError there took down the entire Controls page rather than
one section of it. The rule this follows is the one already applied to NumPyro in
``fpa.forecast.bayes``: reading an artifact must not depend on the machinery that wrote
it.
"""

from __future__ import annotations

import json
from pathlib import Path

from fpa.config import Settings


def proof_path(settings: Settings) -> Path:
    """Where the recorded refusal lives.

    JSON rather than Parquet on purpose: one small record whose value is that a human can
    read it and a diff can show it changing. An audit artifact that needs a library to
    open is a worse audit artifact.
    """
    return settings.data_dir / f"erp_balance_proof.{settings.data_vintage}.json"


def load_proof(settings: Settings) -> dict | None:
    """Read the recorded refusal, or ``None`` if it has never been run.

    Returns ``None`` rather than raising on unreadable content: a missing or corrupt
    proof should collapse one section of a page, not the page.
    """
    path = proof_path(settings)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
