"""Pinned-vintage parquet cache.

A thin wrapper: the first call fetches live and writes ``<name>.<vintage>.parquet``;
every later call for the same vintage reads the snapshot. Freezing the data vintage is
what makes a run reproducible offline — re-running months later yields byte-identical
inputs, and a demo never depends on a live API being up.

Lives in this package rather than the workspace-level ``shared/`` on purpose. A portfolio
repo has to be clonable and runnable on its own: a hosted deploy (Streamlit Community
Cloud) clones exactly one repository, and an import reaching outside it fails at startup
before a single page renders. Forty-nine lines is a cheap price for a self-contained repo.

Usage::

    from fpa.cache import cached_parquet
    frame = cached_parquet(settings.vintage_path("edgar_nflx"), fetch_fn)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)


def cached_parquet(
    path: Path,
    fetch: Callable[[], pd.DataFrame | pd.Series],
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return ``fetch()`` output, reading/writing a parquet snapshot at ``path``.

    ``fetch`` is a zero-arg callable returning a DataFrame or Series. Pass
    ``refresh=True`` to force a re-fetch and overwrite the snapshot.
    """
    path = Path(path)
    if path.exists() and not refresh:
        logger.info("cache hit: %s", path.name)
        return pd.read_parquet(path)

    logger.info("cache miss: fetching %s", path.name)
    obj = fetch()
    frame = obj.to_frame() if isinstance(obj, pd.Series) else obj
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return frame
