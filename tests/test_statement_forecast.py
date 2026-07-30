"""Tests for the three-statement forecast.

The claim under test is structural: only revenue is forecast, and everything else follows.
Two tests matter more than the rest.

``test_a_missing_derived_line_raises`` pins the bug this module was born with. The first
version read ``quarterly`` instead of the balance-sheet frame and guarded each line with
``if column in history.columns``, so ``treasury_stock`` — a derived residual worth **−$28B**
— was silently skipped. Equity was projected with no contra-equity, and the cash plug absorbed
the error and grew to $84B against an actual $9B. Fourth instance of this codebase's signature
defect, and the only reason it was visible is that the plug produced an absurd number rather
than a merely wrong one.

``test_the_forecast_articulates`` is the property the whole structure exists for: a forecast
that broke ``A = L + E`` while a blocking control proves it on actuals would be the worst
inconsistency in the project.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpa.config import EXPENSE_ACCOUNTS, get_settings
from fpa.forecast import statements as S
from fpa.pipeline import build_ledger


@pytest.fixture(scope="module")
def context():
    return build_ledger(get_settings())


@pytest.fixture(scope="module")
def forecasts(context):
    income = S.forecast_income_statement(context.quarterly, 4)
    balance = S.forecast_balance_sheet(context.balance_sheet, context.quarterly, income)
    return income, balance


# ---------------------------------------------------------------------------
# Income statement: one series forecast, the rest driven
# ---------------------------------------------------------------------------
def test_operating_income_is_the_residual_not_a_forecast(forecasts):
    """The claim the README made for weeks without implementing it."""
    income, _ = forecasts
    expected = income["revenue"] - income[list(EXPENSE_ACCOUNTS)].sum(axis=1)
    pd.testing.assert_series_equal(
        income["operating_income"], expected, check_names=False
    )


def test_every_expense_is_a_fixed_share_of_forecast_revenue(forecasts):
    income, _ = forecasts
    for account, ratio in income.attrs["cost_ratios"].items():
        implied = (income[account] / income["revenue"]).round(10).unique()
        assert len(implied) == 1, f"{account} ratio drifts across the horizon"
        assert implied[0] == pytest.approx(ratio, rel=1e-9)


def test_deriving_operating_income_requires_every_expense_line(context):
    """A residual computed from a partial set is wrong and must not be produced silently."""
    trimmed = context.quarterly.drop(columns=["marketing"])
    with pytest.raises(ValueError, match="cannot derive operating income"):
        S.forecast_income_statement(trimmed, 4)


def test_ratios_use_the_trailing_window_not_all_history(context):
    """Measured, not stylistic: cost of revenue averages ~56.6% across 26 quarters and
    ~51.0% over the last four, because the mix shifted. A full-history ratio would forecast
    a margin the business left behind."""
    trailing = S.cost_ratios(context.quarterly, window=4)["cost_of_revenue"]
    full = S.cost_ratios(context.quarterly, window=len(context.quarterly))["cost_of_revenue"]
    assert trailing < full - 0.02


# ---------------------------------------------------------------------------
# Balance sheet: derived, and it has to balance
# ---------------------------------------------------------------------------
def test_the_forecast_articulates(forecasts):
    """A = L + E on every forecast quarter, to float precision."""
    _, balance = forecasts
    assert balance["balance_check"].abs().max() < 1.0  # dollars, on a ~$60B sheet


def test_a_missing_derived_line_raises(context):
    """The bug: treasury stock silently skipped, equity overstated by ~$28B.

    If this ever stops raising, a partial balance sheet can be projected again and the cash
    plug will absorb whatever is missing — producing a number that looks like a forecast.
    """
    income = S.forecast_income_statement(context.quarterly, 4)
    crippled = context.balance_sheet.drop(columns=["treasury_stock"])

    with pytest.raises(ValueError, match="missing required lines"):
        S.forecast_balance_sheet(crippled, context.quarterly, income)


def test_retained_earnings_roll_matches_net_income(forecasts):
    income, balance = forecasts
    rolled = balance["retained_earnings"].diff().dropna()
    # check_freq=False: .diff() drops the index frequency while .iloc[1:] keeps it, and a
    # freq attribute is not the claim under test.
    pd.testing.assert_series_equal(
        rolled, income["net_income"].iloc[1:], check_names=False, check_freq=False
    )


def test_treasury_stock_becomes_more_negative_as_buybacks_fund(forecasts):
    """Contra-equity must grow with repurchases, or equity compounds unopposed —
    which is exactly how the $84B cash forecast happened."""
    _, balance = forecasts
    assert (balance["treasury_stock"].diff().dropna() < 0).all()
    assert balance.attrs["buybacks_per_quarter"] > 0


def test_the_forecast_stays_in_the_same_order_of_magnitude_as_the_actuals(context, forecasts):
    """The smell test the plug failed loudly the first time.

    Not a tight bound — a forecast should be allowed to move — but cash tripling in four
    quarters means an assumption is wrong, not that the business changed.
    """
    _, balance = forecasts
    last = context.balance_sheet.dropna(subset=["assets"]).iloc[-1]
    for line in ("assets", "liabilities", "equity", "cash"):
        ratio = balance[line].iloc[-1] / last[line]
        assert 0.5 < ratio < 2.0, f"{line} moves {ratio:.1f}x in four quarters"


def test_the_plug_is_checked_against_an_independent_roll_forward(context, forecasts):
    """Cash closing the identity makes the identity untestable, so the plug needs its own
    diagnostic. It caught the original defect; it has to keep working."""
    income, balance = forecasts
    plug = S.plug_plausibility(balance, context.quarterly, income)

    assert {"plug_change_in_cash", "roll_forward_change", "gap"} <= set(plug.columns)
    # The gap is real and reported, not asserted away: the simple roll-forward omits
    # working-capital timing and content cash spend. It must stay bounded, though.
    assert plug["gap_pct_of_revenue"].abs().max() < 0.30


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------
def test_derived_and_direct_are_scored_on_identical_folds(context):
    comparison = S.compare_derived_vs_direct(context.quarterly, horizon=4, folds=4)
    counts = comparison.groupby("approach")["fold"].apply(set)
    assert counts["derived"] == counts["direct"], "the two approaches must share folds"
    assert comparison["mase"].notna().all()


def test_the_report_states_the_verdict_either_way(context, forecasts):
    """A comparison that only prints a number lets the reader miss the finding."""
    income, balance = forecasts
    comparison = S.compare_derived_vs_direct(context.quarterly, horizon=4, folds=4)
    markdown = S.statement_report_markdown(income, balance, comparison)

    assert "Derivation" in markdown
    assert "cash as the plug" in markdown
    # The assumptions have to be on the page, not just in the code.
    assert "as % of revenue" in markdown and "days" in markdown
    # And the thing deliberately not forecast.
    assert "7.571" in markdown and "not forecast" in markdown
