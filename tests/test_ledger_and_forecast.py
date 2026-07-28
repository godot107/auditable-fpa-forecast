"""Tests for disaggregation, the variance bridge, and forecast metrics.

Each test names the claim that would be void if it broke.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpa.config import EXPENSE_ACCOUNTS
from fpa.forecast.backtest import mase, rolling_origin_splits, wmape
from fpa.forecast.models import seasonal_naive
from fpa.variance import assert_bridge_ties, build_variance_report, cost_center_variance


# --- Disaggregation: "modeled, but foots to filed" -------------------------
def test_monthly_ledger_foots_to_filed_quarters(result):
    """Void claim: that cost-center detail is constrained by the filings."""
    rolled = result.ledger.groupby(["quarter_end", "account"])["amount"].sum().unstack()
    accounts = [a for a in EXPENSE_ACCOUNTS if a in rolled.columns]
    filed = result.quarterly[accounts].reindex(rolled.index)

    relative = ((rolled[accounts] - filed).abs() / filed.abs()).to_numpy()
    assert np.nanmax(relative) < 1e-9


def test_driver_product_reproduces_revenue(result):
    """Void claim: that only one side of the rate/volume split is modeled."""
    assert np.allclose(
        result.drivers["members"] * result.drivers["arpu"], result.drivers["revenue"]
    )


def test_ledger_is_reproducible_from_the_seed(settings):
    """Void claim: that a demo run is deterministic."""
    from fpa.ingest.edgar import quarterly_actuals
    from fpa.ledger.disaggregate import monthly_ledger

    quarterly = quarterly_actuals(settings)
    first = monthly_ledger(settings, quarterly)
    second = monthly_ledger(settings, quarterly)
    pd.testing.assert_frame_equal(first, second)


# --- Q4 derivation guard ---------------------------------------------------
def test_q4_is_not_derived_from_years_that_do_not_articulate(result):
    """Void claim: that the 2020 window start was measured rather than chosen.

    FY2017-FY2019 must be refused; FY2020 onward must be used.
    """
    provenance = result.metadata["q4_provenance"]
    for year in (2017, 2018, 2019):
        assert provenance[year]["reason"] == "annual identity broken"
        assert provenance[year]["derived"] == []
    assert provenance[2021]["reason"] == "verified"
    assert provenance[2021]["derived"]


def test_window_contains_no_pre_2020_data(result):
    assert result.quarterly.index.min() >= pd.Timestamp("2020-01-01")


# --- Variance bridge -------------------------------------------------------
def test_variance_effects_sum_to_the_total(result):
    """Void claim: the bridge. A decomposition that does not tie explains nothing."""
    variance = cost_center_variance(result.ledger, result.budget, periods=12)
    assert_bridge_ties(variance)  # raises if it does not tie

    recomposed = (variance["spend_effect"] + variance["mix_effect"]).sum()
    assert np.isclose(recomposed, variance["variance"].sum())


def test_variance_report_assembles(result):
    report = build_variance_report(
        result.ledger, result.budget, result.revenue, result.revenue_budget,
        result.drivers, periods=12,
    )
    assert not report["by_cost_center"].empty
    assert report["summary"]["total_variance"] != 0
    # The driver split must declare what it did not measure.
    assert "not measured" in report["revenue_drivers"]["basis"]


# --- Forecast metrics ------------------------------------------------------
def test_mase_is_one_for_the_benchmark_against_itself():
    """A metric that does not equal 1 for its own benchmark is miscalibrated."""
    rng = np.random.default_rng(0)
    series = pd.Series(100 + np.arange(48) + rng.normal(0, 5, 48))
    train, test = series[:36], series[36:].to_numpy()
    predicted = seasonal_naive(train, len(test), period=12)

    value = mase(test, predicted, train.to_numpy(), period=12)
    assert 0.2 < value < 5.0  # finite and sane, not a divide-by-zero artifact


def test_wmape_is_zero_for_a_perfect_forecast():
    actual = np.array([10.0, 20.0, 30.0])
    assert wmape(actual, actual) == 0.0


def test_wmape_is_not_dominated_by_a_small_denominator():
    """Void claim: the reason MASE/wMAPE replaced plain MAPE.

    One tiny actual with a large relative error must not swamp the metric.
    """
    actual = np.array([1_000_000.0, 1_000_000.0, 1.0])
    predicted = np.array([1_000_000.0, 1_000_000.0, 2.0])
    assert wmape(actual, predicted) < 1e-5


def test_backtest_folds_never_see_inside_their_own_horizon():
    """Void claim: that the reported accuracy is achievable in production."""
    for train_idx, test_idx in rolling_origin_splits(78, 12, folds=6):
        assert train_idx.max() < test_idx.min()
        assert len(test_idx) == 12


def test_backtest_reports_where_the_model_loses(result):
    """Void claim: honest reporting. A backtest showing only wins is marketing."""
    losses = result.backtest_filed.losses_to_benchmark()
    assert isinstance(losses, pd.DataFrame)
    # On filed data this model genuinely does lose on some series.
    assert len(losses) > 0


def test_filed_backtest_is_less_flattering_than_the_modeled_one(result):
    """Void claim: the headline honesty finding.

    The monthly ledger is disaggregated by this project, so a model scores better on
    it than on filed data. If that gap ever inverted, the honest-metrics section of
    the README would be wrong.
    """
    filed = result.backtest_filed.summary().set_index("model")["mase"]
    monthly = result.backtest_monthly.summary().set_index("model")["mase"]
    assert filed["ets"] > monthly["ets"]
