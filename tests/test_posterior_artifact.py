"""Tests for the pinned posterior — the cache, and the stamp that keeps it honest.

Caching model output introduces exactly the defect this codebase keeps finding: an
input moves, the cached number does not, and the result is plausible rather than wrong.
So the stamp is tested harder than the cache. ``test_a_changed_series_is_detected``
is the one that matters — if it ever stops failing on perturbed data, the artifact can
serve a fan fitted to numbers that no longer exist.

The simulation itself is tested for *agreement* rather than for values: a cached
posterior and a live fit must run the same code, or the artifact silently becomes a
second implementation free to drift from the one the calibration report measured.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpa.forecast import posterior as P


@pytest.fixture
def fake_samples():
    """A posterior-shaped dict, without paying for a NUTS fit.

    Shapes match what ``fit`` returns: latent paths for level and trend, one seasonal
    offset per month, and scalar scales per draw.
    """
    rng = np.random.default_rng(0)
    draws, months = 200, 78
    return {
        "level": np.cumsum(rng.normal(0, 0.01, (draws, months)), axis=1) + 20.0,
        "trend": np.cumsum(rng.normal(0, 0.001, (draws, months)), axis=1),
        "seasonal": rng.normal(0, 0.05, (draws, 12)),
        "sigma_trend": np.abs(rng.normal(0.005, 0.001, draws)),
        "sigma_obs": np.abs(rng.normal(0.05, 0.005, draws)),
    }


# ---------------------------------------------------------------------------
# One simulation, two entry points
# ---------------------------------------------------------------------------
def test_cached_and_live_simulation_are_the_same_code(fake_samples):
    """The refactor's whole justification.

    ``simulate`` takes a fit; ``simulate_from_state`` takes terminal state read off
    disk. If these ever disagree, the cached artifact has become a second model and the
    calibration report no longer describes what the app shows.
    """
    from fpa.forecast.bayes import simulate, simulate_from_state

    live = simulate(fake_samples, 12, 5, seed=7)
    cached = simulate_from_state(
        fake_samples["level"][:, -1],
        fake_samples["trend"][:, -1],
        fake_samples["seasonal"],
        fake_samples["sigma_trend"],
        fake_samples["sigma_obs"],
        12,
        5,
        seed=7,
    )
    np.testing.assert_array_equal(live, cached)


def test_only_terminal_state_is_needed(fake_samples):
    """Why the artifact is ~1 MB rather than ~30 MB.

    Corrupting the latent history must not change the forecast. If it does, the stored
    columns are insufficient and the cache is lossy.
    """
    from fpa.forecast.bayes import simulate

    before = simulate(fake_samples, 12, 5, seed=7)
    fake_samples["level"][:, :-1] = 999.0
    fake_samples["trend"][:, :-1] = -999.0
    after = simulate(fake_samples, 12, 5, seed=7)

    np.testing.assert_array_equal(before, after)


# ---------------------------------------------------------------------------
# The stamp
# ---------------------------------------------------------------------------
def _series(values, start="2020-01-31"):
    index = pd.date_range(start, periods=len(values), freq="ME")
    return pd.Series(values, index=index)


def test_digest_is_stable_for_identical_data():
    values = np.linspace(100.0, 200.0, 24)
    assert P.series_digest(_series(values)) == P.series_digest(_series(values.copy()))


def test_a_changed_value_changes_the_digest():
    values = np.linspace(100.0, 200.0, 24)
    moved = values.copy()
    moved[7] += 0.01  # a cent on a hundred dollars
    assert P.series_digest(_series(values)) != P.series_digest(_series(moved))


def test_a_shifted_series_changes_the_digest():
    """Values alone are not enough.

    The same numbers starting a month later imply different seasonal offsets, so a
    digest over values only would call a genuinely stale posterior fresh.
    """
    values = np.linspace(100.0, 200.0, 24)
    assert P.series_digest(_series(values, "2020-01-31")) != P.series_digest(
        _series(values, "2020-02-29")
    )


def _posterior_for(leaves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, series in leaves.items():
        rows.append(
            {
                "function": key[0],
                "sub_center": key[1],
                "digest": P.series_digest(series.dropna()),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def leaves():
    index = pd.date_range("2020-01-31", periods=24, freq="ME")
    return pd.DataFrame(
        {
            ("Content", "Licensed Content"): np.linspace(100.0, 200.0, 24),
            ("G&A", "Facilities"): np.linspace(10.0, 20.0, 24),
        },
        index=index,
    )


def test_a_matching_posterior_is_not_stale(leaves):
    assert P.stale_series(_posterior_for(leaves), leaves) == []


def test_a_changed_series_is_detected(leaves):
    """The test that must never stop failing.

    Without it, `--refresh` moves the filings, the draws stay put, and the app serves
    an interval fitted to numbers that are no longer on disk — indistinguishable, on
    screen, from a valid one.
    """
    posterior = _posterior_for(leaves)
    moved = leaves.copy()
    moved[("G&A", "Facilities")] = moved[("G&A", "Facilities")] * 1.0001

    stale = P.stale_series(posterior, moved)
    assert stale == [("G&A", "Facilities")]


def test_a_series_absent_from_the_artifact_counts_as_stale(leaves):
    """Silence is the wrong failure. A missing fan must be reported, not omitted."""
    posterior = _posterior_for(leaves)
    posterior = posterior[posterior["sub_center"] != "Facilities"]

    assert ("G&A", "Facilities") in P.stale_series(posterior, leaves)


def test_an_empty_artifact_marks_everything_stale(leaves):
    empty = pd.DataFrame(columns=["function", "sub_center", "digest"])
    assert set(P.stale_series(empty, leaves)) == set(leaves.columns)


# ---------------------------------------------------------------------------
# The page must not sample
# ---------------------------------------------------------------------------
def test_the_forecast_page_never_fits_a_model():
    """The fix for the traceback on the hosted app, asserted structurally.

    The page previously guarded ``from fpa.forecast.bayes import forecast_intervals``
    with ``except ImportError`` — dead code, because numpyro is imported lazily inside
    ``fit`` and that import always succeeds. The error surfaced later, at call time,
    outside the try. It now reads a pinned artifact and never calls the sampler at all,
    which removes the failure mode rather than handling it.
    """
    from pathlib import Path

    page = (Path(__file__).resolve().parents[1] / "app" / "pages" / "2_Forecast.py").read_text()

    assert "load_posterior" in page
    assert "forecast_intervals" not in page, "the page must not fit; it loads pinned draws"
    assert "except ImportError" not in page
    assert "stale_series" in page, "a cached posterior must be checked against the data"
