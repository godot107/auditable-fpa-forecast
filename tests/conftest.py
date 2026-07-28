"""Shared fixtures.

The pipeline is module-scoped: it reads a pinned parquet vintage, so it is
deterministic, but re-running ingest and two backtests per test is wasteful.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fpa.config import get_settings  # noqa: E402
from fpa.pipeline import run  # noqa: E402


@pytest.fixture(scope="module")
def settings():
    return get_settings()


@pytest.fixture(scope="module")
def result(settings):
    """One full pipeline run, shared across a module's tests."""
    return run(settings)
