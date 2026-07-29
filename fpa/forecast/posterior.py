"""A pinned posterior, so serving an interval needs no sampler.

Fitting inside the Streamlit app was wrong twice over. It cannot run on the hosted
deploy at all — NumPyro and JAX are deliberately out of ``requirements.txt`` — and even
locally a four-chain NUTS fit takes minutes behind a button that claimed fifteen
seconds. Neither is a property of the model; both are a property of doing expensive
work at read time.

So the posterior is fitted once, offline, and committed the same way the data vintage
is committed. ``python -m fpa.forecast.posterior`` writes
``data/posterior_<ticker>.<vintage>.parquet``; the app reads it and forward-simulates
in pure NumPy, in milliseconds.

**What is stored is not a fitted path.** Five arrays per draw — terminal level,
terminal trend, twelve seasonal offsets, and the two scale parameters — which is
everything :func:`fpa.forecast.bayes.simulate_from_state` consumes. The latent history
is discarded because the forward simulation never reads it. That is why nine cost
centers fit in roughly a megabyte rather than thirty.

**Every row carries a stamp, and the stamp is load-bearing.** A cached posterior is a
number computed from inputs that may since have moved: run ``--refresh``, and the
filings change while the draws do not. Nothing about the resulting fan would look
wrong. So each series stores a digest of the exact values it was fitted to, and
:func:`stale_series` re-derives that digest from the live ledger on every read. A
mismatch is reported, not absorbed — this project has found the same defect three
times already (a missing or shifted input that yields a plausible number instead of an
error), and caching model output is an obvious fourth opportunity.

The diagnostics travel with the draws for the same reason. An interval whose R-hat was
never inspected is not evidence, and eight of nine backtest fits in this project do not
converge. ``converged`` is stored per series so a reader can be refused a fan the
sampler never earned.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd

from fpa.config import Settings, get_settings

logger = logging.getLogger(__name__)

# 6000 draws are collected (4 chains x 1500) and 2000 are kept. The quantiles reported
# are 10/50/90; the Monte Carlo error on a decile from 2000 draws is far below the
# width of the interval itself, so thinning costs nothing a reader could see and cuts
# the artifact to a third.
THIN_DRAWS = 2000

SEASONAL_COLUMNS = tuple(f"seasonal_{i}" for i in range(12))
STATE_COLUMNS = ("level", "trend", "sigma_trend", "sigma_obs") + SEASONAL_COLUMNS
STAMP_COLUMNS = ("vintage", "last_period", "n_obs", "digest")
DIAGNOSTIC_COLUMNS = ("worst_rhat", "min_ess", "converged", "divergences")


def artifact_path(settings: Settings):
    return settings.vintage_path(f"posterior_{settings.ticker.lower()}")


def series_digest(values: pd.Series) -> str:
    """Fingerprint the exact numbers a fit consumed.

    Both the values and their period index, because a series can be shifted in time
    without any value changing and the resulting seasonal offsets would be wrong.
    Truncated to 16 hex characters — this detects drift, it is not a security control.
    """
    clean = values.dropna()
    payload = (
        np.asarray(clean.to_numpy(), dtype=np.float64).tobytes()
        + ",".join(str(p) for p in clean.index).encode()
    )
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_posterior(
    leaves: pd.DataFrame, settings: Settings, *, seed: int = 42, draws: int = THIN_DRAWS
) -> pd.DataFrame:
    """Fit every leaf series and return the thinned draws as one long frame.

    Imports the heavy stack lazily, like the rest of ``fpa.forecast.bayes`` — this
    module is importable, and its loader usable, on a host with no JAX.
    """
    from fpa.forecast.bayes import fit

    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []

    for (function, sub_center), series in leaves.items():
        history = series.dropna()
        logger.info("fitting %s / %s (%d obs)", function, sub_center, len(history))
        samples = fit(history, seed=seed)

        total = samples["level"].shape[0]
        keep = np.sort(rng.choice(total, size=min(draws, total), replace=False))

        block = pd.DataFrame(
            {
                "level": samples["level"][keep, -1],
                "trend": samples["trend"][keep, -1],
                "sigma_trend": samples["sigma_trend"][keep],
                "sigma_obs": samples["sigma_obs"][keep],
            }
        )
        for i, column in enumerate(SEASONAL_COLUMNS):
            block[column] = samples["seasonal"][keep, i]

        diagnostics = samples["_diagnostics"]
        block.insert(0, "function", function)
        block.insert(1, "sub_center", sub_center)
        block["vintage"] = settings.data_vintage
        block["last_period"] = pd.Timestamp(history.index[-1])
        block["n_obs"] = len(history)
        block["digest"] = series_digest(history)
        block["worst_rhat"] = diagnostics["worst_rhat"]
        block["min_ess"] = diagnostics["min_ess"]
        block["converged"] = bool(diagnostics["converged"])
        block["divergences"] = int(np.asarray(samples["_divergences"]).sum())

        if not diagnostics["converged"]:
            logger.warning(
                "%s / %s did not converge (R-hat %.3f, ESS %.0f) — stored and flagged",
                function,
                sub_center,
                diagnostics["worst_rhat"],
                diagnostics["min_ess"],
            )
        frames.append(block)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Load and serve
# ---------------------------------------------------------------------------
def load_posterior(settings: Settings) -> pd.DataFrame | None:
    """Read the pinned posterior, or ``None`` when it has not been built."""
    path = artifact_path(settings)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def stale_series(posterior: pd.DataFrame, leaves: pd.DataFrame) -> list[tuple[str, str]]:
    """Series whose stored draws no longer match the data on disk.

    Recomputes the digest from the live ledger rather than trusting the stamp to
    describe itself. A series absent from the artifact counts as stale: a fan that
    silently does not appear is a worse failure than one reported missing.
    """
    stored = (
        posterior.groupby(["function", "sub_center"])["digest"].first().to_dict()
        if not posterior.empty
        else {}
    )
    stale: list[tuple[str, str]] = []
    for key, series in leaves.items():
        if stored.get(key) != series_digest(series.dropna()):
            stale.append(key)
    return stale


def series_state(posterior: pd.DataFrame, key: tuple[str, str]) -> dict:
    """The five arrays plus the stamp for one series."""
    rows = posterior[
        (posterior["function"] == key[0]) & (posterior["sub_center"] == key[1])
    ]
    if rows.empty:
        raise KeyError(f"no stored posterior for {key}")

    first = rows.iloc[0]
    return {
        "level": rows["level"].to_numpy(),
        "trend": rows["trend"].to_numpy(),
        "sigma_trend": rows["sigma_trend"].to_numpy(),
        "sigma_obs": rows["sigma_obs"].to_numpy(),
        "seasonal": rows[list(SEASONAL_COLUMNS)].to_numpy(),
        "last_period": pd.Timestamp(first["last_period"]),
        "worst_rhat": float(first["worst_rhat"]),
        "min_ess": float(first["min_ess"]),
        "converged": bool(first["converged"]),
        "divergences": int(first["divergences"]),
        "draws": len(rows),
    }


def posterior_intervals(
    posterior: pd.DataFrame,
    key: tuple[str, str],
    horizon: int,
    *,
    seed: int = 42,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> pd.DataFrame:
    """Forward-simulate from stored draws. No JAX, no sampler, milliseconds.

    Runs the identical simulation the live path runs — see
    :func:`fpa.forecast.bayes.simulate_from_state`.
    """
    from fpa.forecast.bayes import simulate_from_state

    state = series_state(posterior, key)
    last_period = state["last_period"]

    paths = simulate_from_state(
        state["level"],
        state["trend"],
        state["seasonal"],
        state["sigma_trend"],
        state["sigma_obs"],
        horizon,
        int(last_period.month - 1),
        seed=seed,
    )

    future = pd.date_range(last_period + pd.offsets.MonthEnd(1), periods=horizon, freq="ME")
    frame = pd.DataFrame(
        {f"p{int(q * 100)}": np.quantile(paths, q, axis=0) for q in quantiles}, index=future
    )
    frame.index.name = "period"
    frame.attrs.update(
        worst_rhat=state["worst_rhat"],
        min_ess=state["min_ess"],
        converged=state["converged"],
        divergences=state["divergences"],
        draws=state["draws"],
    )
    return frame


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(settings: Settings | None = None, *, seed: int = 42) -> int:
    from fpa.forecast.models import leaf_series
    from fpa.pipeline import build_ledger

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = settings or get_settings()

    leaves = leaf_series(build_ledger(settings).ledger)
    posterior = build_posterior(leaves, settings, seed=seed)

    path = artifact_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    posterior.to_parquet(path, index=False)

    per_series = posterior.groupby(["function", "sub_center"])[
        ["worst_rhat", "min_ess", "converged"]
    ].first()
    converged = int(per_series["converged"].sum())
    size_kb = path.stat().st_size / 1024

    print(f"\nWrote {path.name} — {len(posterior):,} rows, {size_kb:,.0f} KB")
    print(f"Converged: {converged}/{len(per_series)} series (R-hat <= 1.01 and ESS >= 400)")
    print(per_series.to_string())
    if converged < len(per_series):
        print("\nUnconverged series are stored and flagged, never silently served.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
