"""Tests for the posterior-predictive interval layer.

Split deliberately in two:

* The **scoring** tests run always. They are pure NumPy and they are what stops the
  headline claim — "coverage *and* sharpness" — from being decoration.
* The **inference** tests are skipped unless NumPyro is installed, and they exist to
  assert the one property the deleted Pyro scaffold did not have: that the posterior
  actually depends on the data.

That second point is the whole reason this module was rewritten. A seeded RNG
produces a plausible fan too. The difference is whether it moves when the data does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpa.forecast.bayes import (
    ESS_FLOOR,
    NOMINAL_COVERAGE,
    QUANTILES,
    RHAT_CEILING,
    IntervalReport,
    coverage,
    interval_report_markdown,
    naive_intervals,
    pinball_loss,
    prior_predictive,
    prior_predictive_summary,
    score_intervals,
    simulate,
)

def _months(n: int, start: str = "2020-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


# ---------------------------------------------------------------------------
# Scoring — always runs
# ---------------------------------------------------------------------------
def test_coverage_counts_points_inside_the_band():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    assert coverage(actual, np.zeros(4), np.full(4, 5.0)) == 1.0
    assert coverage(actual, np.full(4, 2.5), np.full(4, 5.0)) == 0.5


def test_pinball_loss_is_minimised_at_the_true_quantile():
    """The property that makes it a *proper* scoring rule.

    If this failed, the metric would not be measuring quantile quality at all and
    every conclusion drawn from the calibration table would be void.
    """
    rng = np.random.default_rng(0)
    sample = rng.normal(size=20_000)
    for q in (0.1, 0.5, 0.9):
        truth = float(np.quantile(sample, q))
        at_truth = pinball_loss(sample, np.full_like(sample, truth), q)
        for offset in (-0.5, -0.1, 0.1, 0.5):
            assert pinball_loss(sample, np.full_like(sample, truth + offset), q) > at_truth


def test_pinball_loss_rejects_a_quantile_outside_the_unit_interval():
    with pytest.raises(ValueError):
        pinball_loss(np.zeros(3), np.zeros(3), 1.0)


def test_pinball_loss_penalises_a_uselessly_wide_interval():
    """The claim the whole evaluation section rests on.

    A band of ±10x achieves 100% coverage. Coverage alone would rank it best; pinball
    loss must rank it worse than a tight, well-placed one. Without this, "coverage
    and sharpness" is a sentence in a README rather than a property of the code.
    """
    actual = np.array([100.0, 102.0, 98.0, 101.0])
    tight = np.column_stack([actual - 5, actual, actual + 5])
    absurd = np.column_stack([actual - 1000, actual, actual + 1000])

    tight_score = score_intervals(actual, tight, QUANTILES)
    absurd_score = score_intervals(actual, absurd, QUANTILES)

    assert absurd_score["coverage"] == 1.0
    assert tight_score["coverage"] == 1.0
    assert absurd_score["pinball"] > tight_score["pinball"]
    assert absurd_score["relative_width"] > tight_score["relative_width"]


def test_relative_width_is_comparable_across_scales():
    """Sharpness is expressed as a share of the level, so a $7B and a $250M cost
    center can appear in the same column without the larger one always looking worse."""
    small = np.full(4, 100.0)
    large = small * 1000
    band = lambda a: np.column_stack([a * 0.9, a, a * 1.1])  # noqa: E731
    assert score_intervals(small, band(small))["relative_width"] == pytest.approx(
        score_intervals(large, band(large))["relative_width"]
    )


def test_naive_benchmark_centres_on_seasonal_naive():
    """The benchmark must be the seasonal-naive point forecast plus its own residual
    spread — the interval-layer analogue of MASE's denominator."""
    series = pd.Series(np.arange(1, 37, dtype=float), index=_months(36))
    intervals = naive_intervals(series, 6, QUANTILES)

    assert intervals.shape == (6, 3)
    # Monotone quantiles: p10 <= p50 <= p90 at every horizon.
    assert (np.diff(intervals, axis=1) >= 0).all()
    # Residuals of a pure +1/month series over a 12-month lag are all exactly 12.
    assert intervals[:, 1] == pytest.approx(series.to_numpy()[-12:][:6] + 12.0)


def test_report_names_the_series_that_buy_calibration_with_width():
    """A series can be over-covered *and* worse than the benchmark. That combination
    is the diagnosis the report exists to make, so it must appear in the text."""
    over = IntervalReport(
        series="Wide One",
        n_points=36,
        model={"coverage": 0.97, "pinball": 900.0, "relative_width": 0.9},
        benchmark={"coverage": 0.80, "pinball": 500.0, "relative_width": 0.3},
    )
    narrow = IntervalReport(
        series="Narrow One",
        n_points=36,
        model={"coverage": 0.50, "pinball": 400.0, "relative_width": 0.1},
        benchmark={"coverage": 0.80, "pinball": 500.0, "relative_width": 0.3},
    )
    markdown = interval_report_markdown([over, narrow])

    assert "Buying calibration with width" in markdown
    assert "`Wide One`" in markdown
    assert "Too narrow" in markdown
    assert "`Narrow One`" in markdown
    assert not over.beats_benchmark
    assert narrow.beats_benchmark  # lower pinball, despite terrible coverage


def test_empty_report_says_so_rather_than_rendering_an_empty_table():
    assert "No interval evaluation" in interval_report_markdown([])


def test_nominal_coverage_matches_the_reported_quantiles():
    assert NOMINAL_COVERAGE == pytest.approx(0.8)
    assert QUANTILES[0] + (1 - QUANTILES[-1]) == pytest.approx(1 - NOMINAL_COVERAGE)


def test_simulation_fans_out_with_horizon():
    """An integrated random walk must widen faster than a fixed-trend model would.

    Runs on synthetic posterior samples, so it needs no sampler: the property is in
    :func:`simulate`, not in the fit.
    """
    draws = 2_000
    rng = np.random.default_rng(1)
    samples = {
        "level": np.full((draws, 1), np.log(100.0)),
        "trend": np.zeros((draws, 1)),
        "seasonal": np.zeros((draws, 12)),
        "sigma_trend": np.full(draws, 0.005),
        "sigma_obs": np.full(draws, 0.01),
    }
    paths = simulate(samples, 12, last_month=0, seed=7)
    width = np.quantile(paths, 0.9, axis=0) - np.quantile(paths, 0.1, axis=0)

    assert paths.shape == (draws, 12)
    assert (paths > 0).all(), "levels are exponentiated and must stay positive"
    assert width[-1] > 2.5 * width[0], "intervals must widen materially over the horizon"


def test_simulation_carries_parameter_uncertainty():
    """Draws with different variance parameters must produce a wider fan.

    This is exactly what the deleted Pyro scaffold lacked: it applied one error band
    to one fitted path. Here the spread of the posterior itself has to show up in the
    predictive distribution.
    """
    draws = 4_000
    base = {
        "level": np.full((draws, 1), np.log(100.0)),
        "trend": np.zeros((draws, 1)),
        "seasonal": np.zeros((draws, 12)),
        "sigma_obs": np.full(draws, 0.01),
    }
    certain = simulate({**base, "sigma_trend": np.full(draws, 0.005)}, 6, 0, seed=3)
    # Same mean trend volatility, but spread across draws rather than fixed.
    uncertain = simulate(
        {**base, "sigma_trend": np.linspace(0.001, 0.009, draws)}, 6, 0, seed=3
    )

    spread = lambda p: np.quantile(p, 0.9, axis=0) - np.quantile(p, 0.1, axis=0)  # noqa: E731
    assert spread(uncertain)[-1] > spread(certain)[-1]


# ---------------------------------------------------------------------------
# Inference — needs the optional stack (requirements-bayes.txt)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_posterior_depends_on_the_data():
    """The one property a seeded RNG cannot fake.

    Two series with different growth rates must produce different posterior trends.
    If this passed for the deleted scaffold it would only be by coincidence, because
    nothing in it was conditioned on anything.
    """
    pytest.importorskip("numpyro")
    from fpa.forecast.bayes import fit

    index = _months(48)
    flat = pd.Series(100.0 * np.ones(48) * (1 + 0.001 * np.arange(48)), index=index)
    steep = pd.Series(100.0 * (1.02 ** np.arange(48)), index=index)

    flat_trend = float(np.mean(fit(flat, seed=1)["trend"][:, -1]))
    steep_trend = float(np.mean(fit(steep, seed=1)["trend"][:, -1]))

    assert steep_trend > flat_trend
    # ~2%/month on the log scale is ~0.0198; the posterior should be in that region.
    assert 0.010 < steep_trend < 0.030


@pytest.mark.slow
def test_fit_rejects_non_positive_values():
    pytest.importorskip("numpyro")
    from fpa.forecast.bayes import fit

    series = pd.Series([1.0, 2.0, -3.0, 4.0], index=_months(4))
    with pytest.raises(ValueError, match="strictly positive"):
        fit(series)


# ---------------------------------------------------------------------------
# Prior predictive — the check McElreath says there is no substitute for
# ---------------------------------------------------------------------------
def test_priors_imply_plausible_annual_growth_for_a_cost_center():
    """The priors must be weakly informative *on the scale that matters*.

    Asserting that in a comment is not enough, and this project of all projects
    should not do it — McElreath 2e p.114: "To figure out what this prior implies,
    we have to simulate the prior predictive distribution. There is no other
    reliable way to understand."

    Simulating found a real defect. ``trend0 ~ Normal(0, 0.10)`` is a 10% log-slope
    *per month*; compounded it implied a 90% band on annual growth of 0.16x to 6.4x,
    with draws reaching 127x. A prior that says a cost center might grow 127-fold in
    a year is driving the answer, whatever it is called.
    """
    summary = prior_predictive_summary()

    # A cost center that could plausibly halve or nearly double in a year: wide, but
    # recognisably a cost center rather than a lottery ticket.
    assert 0.4 < summary["p05_annual_growth"] < 0.8
    assert 1.3 < summary["p95_annual_growth"] < 2.5
    assert summary["max_annual_growth"] < 10.0
    assert 0.9 < summary["p50_annual_growth"] < 1.1, "the prior must be centred on no growth"


def test_prior_predictive_paths_are_positive_and_shaped_correctly():
    paths = prior_predictive(24, draws=500, seed=1)
    assert paths.shape == (500, 24)
    assert (paths > 0).all(), "a multiplicative model cannot produce non-positive spend"


def test_report_flags_a_series_whose_fit_did_not_converge():
    """A coverage figure from an unexplored posterior is a different kind of number.

    The reader has to be told which rows to discount *before* reading the
    calibration conclusions, not in a footnote after them.
    """
    bad = IntervalReport(
        series="Stuck",
        n_points=36,
        model={"coverage": 0.80, "pinball": 400.0, "relative_width": 0.3},
        benchmark={"coverage": 0.80, "pinball": 500.0, "relative_width": 0.3},
        worst_rhat=1.93,
        min_ess=3.0,
    )
    good = IntervalReport(
        series="Fine",
        n_points=36,
        model={"coverage": 0.80, "pinball": 400.0, "relative_width": 0.3},
        benchmark={"coverage": 0.80, "pinball": 500.0, "relative_width": 0.3},
        worst_rhat=1.004,
        min_ess=650.0,
    )
    assert not bad.converged
    assert good.converged
    assert not bad.marginal, "R-hat 1.93 with ESS 3 is a failed sampler, not a threshold miss"

    markdown = interval_report_markdown([bad, good])
    assert "Genuinely unconverged" in markdown
    assert "`Stuck`" in markdown


def test_a_threshold_miss_is_reported_apart_from_a_failed_sampler():
    """R-hat 1.011 with ESS 1,106 is not the same finding as R-hat 1.99 with ESS 3.

    Filing them together produced a published claim — "8 of 9 do not converge" — that was
    true by the letter and wrong in what it conveyed. Four of the ten real failures were
    threshold misses with healthy mixing, and one of them had the *best* ESS in the table.
    """
    marginal = IntervalReport(
        series="Marginal",
        n_points=36,
        model={"coverage": 0.80, "pinball": 400.0, "relative_width": 0.3},
        benchmark={"coverage": 0.80, "pinball": 500.0, "relative_width": 0.3},
        worst_rhat=1.011,
        min_ess=1106.0,
    )
    assert not marginal.converged
    assert marginal.marginal, "just over the ceiling with strong ESS is a threshold artifact"

    markdown = interval_report_markdown([marginal])
    assert "Marginal, and filed separately" in markdown
    assert "threshold artifact" in markdown


def test_the_report_states_that_it_is_not_a_short_window_effect():
    """The corrected diagnosis has to reach the page, not just the commit message.

    The per-fold measurement is unambiguous: the shortest window has the best pass rate.
    The old explanation survived for weeks because only the worst fold was ever reported.
    """
    bad = IntervalReport(
        series="Stuck",
        n_points=36,
        model={"coverage": 0.80, "pinball": 400.0, "relative_width": 0.3},
        benchmark={"coverage": 0.80, "pinball": 500.0, "relative_width": 0.3},
        worst_rhat=1.93,
        min_ess=3.0,
    )
    markdown = interval_report_markdown([bad])
    assert "not a short-window effect" in markdown
    assert "7 of 9" in markdown


def test_folds_are_recorded_individually():
    """An aggregate that cannot express "one bad fold out of three" is not a diagnostic."""
    report = IntervalReport(
        series="Mixed",
        n_points=36,
        model={"coverage": 0.80, "pinball": 400.0, "relative_width": 0.3},
        benchmark={"coverage": 0.80, "pinball": 500.0, "relative_width": 0.3},
        worst_rhat=1.072,
        min_ess=60.0,
        folds=(
            {"fold": 0, "train_months": 42, "worst_rhat": 1.004, "min_ess": 812.0, "converged": True},
            {"fold": 1, "train_months": 54, "worst_rhat": 1.072, "min_ess": 60.0, "converged": False},
            {"fold": 2, "train_months": 66, "worst_rhat": 1.002, "min_ess": 1825.0, "converged": True},
        ),
    )
    assert not report.converged, "the series is flagged when any fold fails"
    assert report.folds_converged == 2, "but two of its three folds are clean"


def test_convergence_thresholds_are_the_conventional_ones():
    """1.01 and 400 are the standard warning lines, not numbers chosen to pass."""
    assert RHAT_CEILING == 1.01
    assert ESS_FLOOR == 400.0
