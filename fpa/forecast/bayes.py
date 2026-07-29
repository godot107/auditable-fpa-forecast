"""Posterior-predictive forecast intervals, with actual inference behind them.

The scaffold this project replaced imported Pyro to run ``pyro.sample`` in a plain
Python loop — no model, no guide, no conditioning, no inference. It was a seeded
``np.random.normal`` wearing a probabilistic label. The rule that came out of
deleting it stands: **never ship a probabilistic label without inference behind
it.** So this module runs NUTS over a state-space model and reports what the
resulting intervals actually achieve.

The model is a **local linear trend with monthly seasonality**, on the log scale
because expenses are positive and grow multiplicatively:

    log y_t = level_t + seasonal[month_t] + noise
    level_t = level_{t-1} + trend_{t-1}
    trend_t = trend_{t-1} + trend_shock

Written with ``cumsum`` rather than a ``scan``: the two are the same model, and the
cumulative form is a plain vectorized expression that NUTS handles directly and a
reader can check by eye. The latent shocks are sampled non-centred, which is what
keeps the geometry tractable when the state variances are small.

Why a local linear trend rather than a fixed linear one: a fitted straight line
treats the slope as known forever, so its intervals grow like sqrt(h) and understate
long-horizon risk. An integrated random walk lets the slope drift, and the interval
widens accordingly. For a twelve-month FP&A forecast that difference is the whole
point of quoting an interval at all.

**Evaluation is on coverage *and* sharpness, never coverage alone.** A model that
predicts +/-$10B every month is perfectly calibrated and perfectly useless —
McElreath's forecaster who says a 40% chance of rain every day (*Statistical
Rethinking* 2e, p.223). Pinball loss is the proper scoring rule that penalizes width,
so it catches what coverage misses. And per this project's day-1 invariant, the
intervals are scored against a benchmark: empirical quantiles of seasonal-naive
residuals. An interval layer that cannot beat that has not earned its dependency.

JAX and NumPyro are lazy-imported. The rest of the test suite and the whole demo
path run without them installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fpa.forecast.backtest import rolling_origin_splits
from fpa.forecast.models import seasonal_naive

logger = logging.getLogger(__name__)

# Quantiles reported. 10/90 gives a nominal 80% interval — the band FP&A actually
# plans against, and wide enough that coverage is measurable on the number of test
# points a rolling-origin backtest of this length provides.
QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)
NOMINAL_COVERAGE = QUANTILES[-1] - QUANTILES[0]

# Small by the standards of a research model, and deliberately so: 78 monthly
# observations per leaf and nine leaves. Enough for a stable posterior, cheap enough
# that the rolling-origin evaluation below is minutes rather than an afternoon.
NUM_WARMUP = 1000
# 1500 per chain. Chosen against the ESS floor rather than by feel: at 500 the
# smaller cost centers landed around 134-138 effective draws, and raising it to
# 1500 takes them to 470-507 for only ~1.5x the wall clock. Low ESS is a compute
# problem, not a model defect, so leaving it failing would have been laziness
# dressed as honesty.
NUM_SAMPLES = 1500

# Four chains, not one — and this is not a throughput decision. R-hat compares
# variance *between* chains to variance *within* them (Gelman-Rubin), so with a
# single chain it cannot be computed at all. Reporting divergences while running
# one chain is reporting the diagnostic that happens to be available rather than
# the ones that matter: McElreath 2e p.309 and Martin et al. p.66 both treat
# R-hat, effective sample size and divergences as a set.
#
# ``vectorized`` runs them under a single vmap, so four chains cost roughly what
# one did on CPU. 4 x 250 keeps the total draw count where it was.
NUM_CHAINS = 4

# Thresholds. R-hat above 1.01 and ESS below 400 are the conventional warning
# lines (Vehtari et al. 2021, the standard McElreath 2e p.287 and Martin et al.
# section 2.4.2 both point at).
RHAT_CEILING = 1.01
ESS_FLOOR = 400.0

# Parameters worth diagnosing. The latent shock vectors are hundreds of weakly
# identified nuisance dimensions whose R-hat says little; these are the ones that
# drive the forecast.
DIAGNOSTIC_SITES = ("sigma_trend", "sigma_obs", "sigma_seasonal", "level0", "trend0")


def _model(month, y=None):
    """Local linear trend + monthly seasonality on the log scale."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    n = month.shape[0]

    # Half-normal scales. Weakly informative on the log scale, where 0.01 is a ~1%
    # month-on-month drift in the slope. ``prior_predictive`` exists to check what
    # these actually imply rather than leaving the claim in this comment.
    sigma_trend = numpyro.sample("sigma_trend", dist.HalfNormal(0.01))
    sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(0.10))
    sigma_seasonal = numpyro.sample("sigma_seasonal", dist.HalfNormal(0.10))

    level0 = numpyro.sample("level0", dist.Normal(0.0, 5.0))
    # 0.02, not 0.10, and the prior predictive is what settled it. A 0.10 log-slope
    # is 10% *per month*; compounded over a year the prior implied a 90% band on
    # annual growth of 0.16x to 6.4x, with draws reaching 127x. That is not weakly
    # informative for a cost center, it is nearly uninformative on the scale that
    # matters. At 0.02 the band is 0.58x to 1.68x — wide enough not to drive the
    # answer, narrow enough to exclude absurdity.
    trend0 = numpyro.sample("trend0", dist.Normal(0.0, 0.02))

    # Sum-to-zero seasonality, so the seasonal effects cannot absorb the level and
    # leave it unidentified.
    seasonal_raw = numpyro.sample(
        "seasonal_raw", dist.Normal(0.0, 1.0).expand([12]).to_event(1)
    )
    seasonal = numpyro.deterministic(
        "seasonal", sigma_seasonal * (seasonal_raw - seasonal_raw.mean())
    )

    # Non-centred latent shocks: sampled as standard normals and scaled, which keeps
    # the posterior geometry well conditioned when the state variance is small.
    #
    # There is deliberately **no** separate level shock. An earlier version had one
    # alongside the observation noise, and the two are not jointly identified — both
    # explain month-to-month variation and the data cannot separate them. Four chains
    # showed it immediately: R-hat 1.93, ESS 3, and posterior means for ``sigma_obs``
    # spanning 0.012 to 0.025 across chains, a 2x spread in the parameter that sets
    # the width of every interval. A single chain reported none of that.
    eps_trend = numpyro.sample("eps_trend", dist.Normal(0.0, 1.0).expand([n]).to_event(1))

    trend = numpyro.deterministic("trend", trend0 + sigma_trend * jnp.cumsum(eps_trend))
    level = numpyro.deterministic("level", level0 + jnp.cumsum(trend))

    numpyro.sample("obs", dist.Normal(level + seasonal[month], sigma_obs), obs=y)


def fit(series: pd.Series, *, seed: int = 42, progress: bool = False) -> dict[str, np.ndarray]:
    """Run NUTS and return posterior samples as plain NumPy arrays.

    Returning NumPy rather than JAX arrays confines the heavy stack to this one
    function — the forward simulation, the quantile maths and every test downstream
    stay free of a JAX import.
    """
    import jax
    import numpyro
    from numpyro.infer import MCMC, NUTS

    values = series.dropna()
    if (values <= 0).any():
        raise ValueError("the log-scale model requires strictly positive values")

    y = np.log(values.to_numpy(dtype=float))
    month = np.asarray(values.index.month - 1, dtype=np.int32)

    # 0.90, and this replaces an earlier 0.99 that was actively harmful. Tuning
    # ``target_accept_prob`` against divergences alone drives it upward, because a
    # small enough step size produces no divergences — the integrator stops taking
    # steps large enough to fail. Measured across three cost centers:
    #
    #     target_accept   divergences        R-hat        ESS
    #            0.80        85 - 142   1.010-1.028   141 - 551
    #            0.90        32 - 189   1.007-1.037   121 - 387
    #            0.95         9 -  87   1.006-1.069    85 - 452
    #            0.99         0 -  38    999 - 1804           2
    #
    # At 0.99 divergences reach *zero* while R-hat explodes past 1000 and ESS
    # collapses to 2: the chains have frozen. Nothing diverges because nothing moves.
    # This is the whole argument for reporting all three diagnostics — optimizing the
    # one you happen to be watching can destroy the two you are not.
    kernel = NUTS(_model, target_accept_prob=0.90)
    mcmc = MCMC(
        kernel,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=NUM_CHAINS,
        chain_method="vectorized",
        progress_bar=progress,
    )
    mcmc.run(jax.random.PRNGKey(seed), month=month, y=y, extra_fields=("diverging",))

    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    # Divergences are the sampler telling you it could not explore the posterior.
    # Necessary but not sufficient — see :func:`diagnostics`.
    extra = mcmc.get_extra_fields()
    samples["_divergences"] = np.asarray(
        extra.get("diverging", np.zeros(NUM_SAMPLES * NUM_CHAINS, dtype=bool))
    )
    samples["_diagnostics"] = diagnostics(mcmc.get_samples(group_by_chain=True))
    return samples


def diagnostics(by_chain: dict) -> dict[str, float]:
    """Worst R-hat and smallest effective sample size across the key parameters.

    Three diagnostics, and they fail in different ways, which is why reporting one
    is not reporting convergence:

    * **Divergences** — the sampler hit curvature it could not integrate through.
      Says the geometry defeated it; says nothing about whether the chains agree.
    * **R-hat** — between-chain variance over within-chain variance. Near 1.0 means
      the chains found the same distribution. **Uncomputable with one chain**, which
      is the reason ``NUM_CHAINS`` is 4.
    * **ESS** — how many independent draws the correlated ones are worth. A chain
      can be perfectly converged and still carry almost no information.

    Reported as the *worst* value across parameters rather than an average: a single
    unconverged parameter invalidates the fit, and averaging hides exactly that.
    """
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

    worst_rhat, min_ess = 1.0, float("inf")
    for site in DIAGNOSTIC_SITES:
        if site not in by_chain:
            continue
        draws = np.asarray(by_chain[site])
        worst_rhat = max(worst_rhat, float(np.max(split_gelman_rubin(draws))))
        min_ess = min(min_ess, float(np.min(effective_sample_size(draws))))

    return {
        "worst_rhat": worst_rhat,
        "min_ess": min_ess if np.isfinite(min_ess) else float("nan"),
        "converged": bool(worst_rhat <= RHAT_CEILING and min_ess >= ESS_FLOOR),
    }


def prior_predictive(
    n_months: int = 78, *, draws: int = 1000, seed: int = 42
) -> "np.ndarray":
    """Trajectories implied by the priors alone, before seeing any data.

    McElreath 2e p.114, on why a comment claiming a prior is weakly informative is
    not enough:

        To figure out what this prior implies, we have to simulate the prior
        predictive distribution. There is no other reliable way to understand.

    That applies with some force here, because the priors sit on the *scale of
    latent shocks in an integrated random walk* — where a value that reads as small
    ("1% a month") compounds over the series length in a way nobody estimates
    correctly by inspection.

    Returns ``(draws, n_months)`` multiplicative trajectories normalized to 1.0 at
    ``t=0``, so the output reads directly as cumulative growth.
    """
    rng = np.random.default_rng(seed)

    # Same priors as ``_model``, sampled directly. Half-normal via |Normal|.
    sigma_trend = np.abs(rng.normal(0.0, 0.01, draws))[:, None]
    sigma_seasonal = np.abs(rng.normal(0.0, 0.10, draws))[:, None]
    sigma_obs = np.abs(rng.normal(0.0, 0.10, draws))[:, None]
    trend0 = rng.normal(0.0, 0.02, draws)[:, None]

    eps = rng.standard_normal((draws, n_months))
    trend = trend0 + sigma_trend * np.cumsum(eps, axis=1)
    level = np.cumsum(trend, axis=1)

    seasonal_raw = rng.standard_normal((draws, 12))
    seasonal_raw -= seasonal_raw.mean(axis=1, keepdims=True)
    seasonal = sigma_seasonal * seasonal_raw
    months = np.arange(n_months) % 12

    log_path = level + seasonal[:, months] + sigma_obs * rng.standard_normal((draws, n_months))
    return np.exp(log_path)


def prior_predictive_summary(n_months: int = 78, **kwargs) -> dict[str, float]:
    """Twelve-month growth multiples implied by the priors.

    A cost center that the priors say could plausibly 100x in a year has priors that
    are driving the answer, whatever the comment above them claims.
    """
    paths = prior_predictive(n_months, **kwargs)
    annual = paths[:, min(11, n_months - 1)] / paths[:, 0]
    return {
        "p05_annual_growth": float(np.quantile(annual, 0.05)),
        "p50_annual_growth": float(np.quantile(annual, 0.50)),
        "p95_annual_growth": float(np.quantile(annual, 0.95)),
        "max_annual_growth": float(annual.max()),
    }


def simulate_from_state(
    level: np.ndarray,
    trend: np.ndarray,
    seasonal: np.ndarray,
    sigma_trend: np.ndarray,
    sigma_obs: np.ndarray,
    horizon: int,
    last_month: int,
    *,
    seed: int = 42,
) -> np.ndarray:
    """The forward simulation itself, taking terminal state rather than a fit.

    Split out so a cached posterior and a live fit run the *same* code. Two copies
    of this loop would be free to drift, and a stored artifact that disagreed with a
    fresh fit is precisely the kind of silent divergence this project exists to
    refuse. ``fpa.forecast.posterior`` calls it with draws read off disk.

    Pure NumPy on purpose: the whole point of caching is that serving an interval
    needs no JAX.
    """
    rng = np.random.default_rng(seed)

    level = np.asarray(level, dtype=float).copy()
    trend = np.asarray(trend, dtype=float).copy()
    seasonal = np.asarray(seasonal, dtype=float)
    sigma_trend = np.asarray(sigma_trend, dtype=float)
    sigma_obs = np.asarray(sigma_obs, dtype=float)

    draws = level.shape[0]
    out = np.empty((draws, horizon), dtype=float)

    for step in range(horizon):
        trend = trend + sigma_trend * rng.standard_normal(draws)
        level = level + trend
        month = (last_month + step + 1) % 12
        out[:, step] = level + seasonal[:, month] + sigma_obs * rng.standard_normal(draws)

    return np.exp(out)


def simulate(
    samples: dict[str, np.ndarray], horizon: int, last_month: int, *, seed: int = 42
) -> np.ndarray:
    """Forward-simulate the posterior predictive. Returns ``(draws, horizon)`` in levels.

    Each posterior draw continues the random walk from *its own* final level and
    trend, with *its own* variance parameters. That is what makes the fan carry
    parameter uncertainty, rather than plotting one fitted path with a fixed error
    band bolted on — precisely the distinction the deleted scaffold got wrong.

    Only the *terminal* level and trend are needed, not the latent path, which is
    what makes the cached artifact small.
    """
    return simulate_from_state(
        samples["level"][:, -1],
        samples["trend"][:, -1],
        samples["seasonal"],
        samples["sigma_trend"],
        samples["sigma_obs"],
        horizon,
        last_month,
        seed=seed,
    )


def forecast_intervals(
    series: pd.Series,
    horizon: int,
    *,
    seed: int = 42,
    quantiles: tuple[float, ...] = QUANTILES,
) -> pd.DataFrame:
    """Fit, simulate, and return one row per future month with quantile columns."""
    values = series.dropna()
    samples = fit(values, seed=seed)
    paths = simulate(samples, horizon, int(values.index[-1].month - 1), seed=seed)

    future = pd.date_range(values.index[-1] + pd.offsets.MonthEnd(1), periods=horizon, freq="ME")
    frame = pd.DataFrame(
        {f"p{int(q * 100)}": np.quantile(paths, q, axis=0) for q in quantiles}, index=future
    )
    frame.index.name = "period"
    # Set attrs last: pandas does not carry them through construction reliably.
    frame.attrs["divergences"] = int(samples["_divergences"].sum())
    return frame


# ---------------------------------------------------------------------------
# Evaluation — coverage AND sharpness, against a benchmark
# ---------------------------------------------------------------------------
def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of actuals falling inside ``[lower, upper]``."""
    actual = np.asarray(actual, float)
    inside = (actual >= np.asarray(lower, float)) & (actual <= np.asarray(upper, float))
    return float(np.mean(inside)) if inside.size else float("nan")


def pinball_loss(actual: np.ndarray, predicted: np.ndarray, q: float) -> float:
    """Mean pinball loss of a ``q``-quantile forecast.

    The proper scoring rule for quantiles: under-prediction is penalized by ``q``,
    over-prediction by ``1 - q``, and its minimizer is the true quantile. Unlike
    coverage it is sensitive to *width*, which is why a uselessly wide interval
    cannot score well on it.

    The same function as ``financial-forecasting-engine``'s
    ``fce/backtest/coverage.py``, restated here rather than imported: this project
    must not take a dependency on that one.
    """
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    error = np.asarray(actual, float) - np.asarray(predicted, float)
    return float(np.mean(np.maximum(q * error, (q - 1.0) * error)))


def naive_intervals(
    train: pd.Series, horizon: int, quantiles: tuple[float, ...] = QUANTILES
) -> np.ndarray:
    """Benchmark intervals: seasonal-naive centre plus empirical residual quantiles.

    The interval-layer equivalent of MASE's denominator. A Bayesian model that
    cannot beat "how wrong was the same month last year, historically" has not
    earned its dependency — and this project does not publish a metric without its
    benchmark.
    """
    values = train.to_numpy(dtype=float)
    period = 12
    if len(values) > period:
        residuals = values[period:] - values[:-period]
    elif len(values) > 1:
        residuals = np.diff(values)
    else:
        residuals = np.zeros(1)

    centre = seasonal_naive(train, horizon, period=period)
    offsets = np.quantile(residuals, quantiles)
    return centre[:, None] + offsets[None, :]


@dataclass
class IntervalReport:
    """Interval quality for one series, model versus benchmark."""

    series: str
    n_points: int
    model: dict[str, float]
    benchmark: dict[str, float]
    divergences: int = 0
    worst_rhat: float = 1.0
    min_ess: float = float("nan")
    detail: dict = field(default_factory=dict)

    @property
    def converged(self) -> bool:
        """Both diagnostics inside their conventional thresholds."""
        return self.worst_rhat <= RHAT_CEILING and self.min_ess >= ESS_FLOOR

    @property
    def beats_benchmark(self) -> bool:
        """Lower mean pinball loss is better."""
        return self.model["pinball"] < self.benchmark["pinball"]


def score_intervals(
    actual: np.ndarray, quantile_values: np.ndarray, quantiles: tuple[float, ...] = QUANTILES
) -> dict[str, float]:
    """Coverage, sharpness and pinball loss for one set of quantile forecasts."""
    lower, upper = quantile_values[:, 0], quantile_values[:, -1]
    losses = [pinball_loss(actual, quantile_values[:, i], q) for i, q in enumerate(quantiles)]
    scale = float(np.mean(np.abs(actual))) or 1.0
    return {
        "coverage": coverage(actual, lower, upper),
        "pinball": float(np.mean(losses)),
        # Sharpness as a share of the level, so it is comparable across cost centers
        # of very different sizes.
        "relative_width": float(np.mean(upper - lower) / scale),
    }


def backtest_intervals(
    series: pd.Series,
    *,
    horizon: int = 12,
    folds: int = 3,
    seed: int = 42,
    quantiles: tuple[float, ...] = QUANTILES,
) -> IntervalReport:
    """Rolling-origin evaluation of the interval layer against the naive benchmark.

    Same origin discipline as the point-forecast backtest: the test window follows
    the training window and the model sees no observation inside it. Fewer folds by
    default, because each one is a full NUTS run.
    """
    values = series.dropna()
    splits = rolling_origin_splits(len(values), horizon, folds=folds)

    actuals: list[np.ndarray] = []
    model_q: list[np.ndarray] = []
    naive_q: list[np.ndarray] = []
    divergences = 0
    worst_rhat, min_ess = 1.0, float("inf")

    for fold, (train_idx, test_idx) in enumerate(splits):
        train, test = values.iloc[train_idx], values.iloc[test_idx]
        samples = fit(train, seed=seed + fold)
        divergences += int(samples["_divergences"].sum())
        fold_diagnostics = samples["_diagnostics"]
        worst_rhat = max(worst_rhat, fold_diagnostics["worst_rhat"])
        min_ess = min(min_ess, fold_diagnostics["min_ess"])

        paths = simulate(samples, len(test), int(train.index[-1].month - 1), seed=seed + fold)
        model_q.append(np.quantile(paths, quantiles, axis=0).T)
        naive_q.append(naive_intervals(train, len(test), quantiles))
        actuals.append(test.to_numpy(dtype=float))

    actual = np.concatenate(actuals)
    return IntervalReport(
        series=str(series.name),
        n_points=int(actual.size),
        model=score_intervals(actual, np.vstack(model_q), quantiles),
        benchmark=score_intervals(actual, np.vstack(naive_q), quantiles),
        divergences=divergences,
        worst_rhat=worst_rhat,
        min_ess=min_ess,
        detail={"folds": len(splits), "horizon": horizon, "nominal_coverage": NOMINAL_COVERAGE},
    )


def evaluate_hierarchy(
    ledger: pd.DataFrame, *, horizon: int = 12, folds: int = 3, seed: int = 42
) -> list[IntervalReport]:
    """Backtest intervals for every leaf cost center.

    Deliberately **not** part of the default pipeline run. Each fold is a full NUTS
    fit, so this is ~6 minutes against ~10 seconds for the rest of the pipeline, and
    a demo path that takes six minutes is a demo nobody runs. Invoked with
    ``python -m fpa --intervals``, which writes the report to ``reports/``.
    """
    from fpa.forecast.models import leaf_series

    leaves = leaf_series(ledger)
    reports: list[IntervalReport] = []
    for column in leaves.columns:
        series = leaves[column].dropna()
        series.name = f"{column[0]} / {column[1]}"
        logger.info("fitting %s", series.name)
        reports.append(backtest_intervals(series, horizon=horizon, folds=folds, seed=seed))
    return reports


def interval_report_markdown(reports: list[IntervalReport]) -> str:
    """Render the interval evaluation, leading with what it does not achieve."""
    if not reports:
        return "_No interval evaluation in this run._"

    lines = [
        f"## Interval calibration — nominal {NOMINAL_COVERAGE:.0%}",
        "",
        "Coverage **and** sharpness. A model that always predicts a huge interval is "
        "perfectly calibrated and perfectly useless, so pinball loss — which penalizes "
        "width — is the scoring rule, and the benchmark is the empirical spread of "
        "seasonal-naive residuals.",
        "",
        "| series | coverage | rel. width | pinball | naive pinball | beats naive? | R-hat | ESS |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for report in sorted(reports, key=lambda r: r.series):
        flag = "" if report.converged else " ⚠"
        lines.append(
            f"| `{report.series}`{flag} | {report.model['coverage']:.0%} | "
            f"{report.model['relative_width']:.2f} | {report.model['pinball']:,.0f} | "
            f"{report.benchmark['pinball']:,.0f} | "
            f"{'**yes**' if report.beats_benchmark else 'no'} | "
            f"{report.worst_rhat:.3f} | {report.min_ess:,.0f} |"
        )

    wins = sum(r.beats_benchmark for r in reports)
    mean_coverage = float(np.mean([r.model["coverage"] for r in reports]))
    total_divergences = sum(r.divergences for r in reports)

    lines += [
        "",
        f"Beats the naive interval on **{wins} of {len(reports)}** series. Mean coverage "
        f"**{mean_coverage:.0%}** against a nominal {NOMINAL_COVERAGE:.0%}.",
    ]

    # The diagnosis worth naming, and the reason both metrics are reported rather
    # than coverage alone: a series can be over-covered *and* worse than the
    # benchmark, which is precisely what buying calibration with width looks like.
    # Coverage on its own would score those series as the best in the table.
    bought = [
        r for r in reports
        if r.model["coverage"] > NOMINAL_COVERAGE + 0.05 and not r.beats_benchmark
    ]
    if bought:
        names = ", ".join(f"`{r.series}`" for r in bought)
        lines += [
            "",
            f"**Buying calibration with width:** {names}. Above-nominal coverage while "
            "losing to the benchmark on pinball loss means the band is wide rather than "
            "well-placed. Coverage alone would rank these the best rows in the table — "
            "which is exactly McElreath's forecaster predicting a 40% chance of rain "
            "every day (*Statistical Rethinking* 2e, p.223).",
        ]

    too_narrow = [r for r in reports if r.model["coverage"] < NOMINAL_COVERAGE - 0.05]
    if too_narrow:
        names = ", ".join(f"`{r.series}`" for r in too_narrow)
        lines += [
            "",
            f"**Too narrow:** {names}. Realized values fall outside the band more often "
            f"than {1 - NOMINAL_COVERAGE:.0%} of the time, which understates risk — the "
            "failure direction that matters in a plan. Reported rather than tuned away.",
        ]
    # Convergence, reported before the calibration conclusions rather than after.
    # A coverage figure computed from a posterior the sampler never explored is not
    # a weaker result, it is a different one — so the reader is told which rows to
    # discount before reading them.
    unconverged = [r for r in reports if not r.converged]
    if unconverged:
        names = ", ".join(f"`{r.series}`" for r in unconverged)
        lines += [
            "",
            f"**⚠ Not converged: {names}.** R-hat above {RHAT_CEILING} or ESS below "
            f"{ESS_FLOOR:.0f}. Their calibration figures are computed from a posterior "
            "the sampler did not fully explore and should be read as provisional.",
        ]
    else:
        lines += [
            "",
            f"All {len(reports)} fits converged: R-hat ≤ {RHAT_CEILING} and ESS ≥ "
            f"{ESS_FLOOR:.0f} on every scale parameter, across every fold.",
        ]

    lines += [
        "",
        f"_{total_divergences} divergent transition(s) across all fits._ Divergences are "
        "reported **with** R-hat and ESS, never alone: tuning `target_accept_prob` against "
        "divergences by itself drives it upward until the step size collapses, at which "
        "point nothing diverges because nothing moves. Measured here at 0.99 — zero "
        "divergences, R-hat past 1000, ESS of 2.",
    ]
    return "\n".join(lines)
