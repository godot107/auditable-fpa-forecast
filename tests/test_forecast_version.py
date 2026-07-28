"""Tests for publishing the forecast to the ERP as a budget version.

The claim under test is narrow and worth stating: the ERP must hold *the same*
forecast Python published. Everything between the two — an account split, a sign
flip, a rounding to cents — is a place the number can quietly change.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpa.ledger.budget import build_budget
from fpa.ledger.forecast_version import (
    account_mix,
    assert_split_ties,
    forecast_by_account,
)


def test_account_mix_sums_to_one_within_each_cost_center(result):
    """A split that does not sum to 1.0 loses or invents spend."""
    mix = account_mix(result.ledger)
    totals = mix.groupby(["function", "sub_center"])["share"].sum()
    assert (totals - 1.0).abs().max() < 1e-12


def test_only_technology_and_product_splits_across_accounts(result):
    """Most cost centers draw on one account; the split exists for the one that does not.

    If this ever changed silently, the split would be doing work nobody reviewed.
    """
    mix = account_mix(result.ledger)
    counts = mix.groupby(["function", "sub_center"])["account"].nunique()
    multi = {function for (function, _sub), n in counts.items() if n > 1}
    assert multi == {"Technology & Product"}


def test_account_split_reproduces_the_leaf_forecast_exactly(result):
    """The account detail must sum back to the forecast it was split from.

    Void this and the ERP holds a forecast that no longer matches the one the
    backtest was run against — while every individual line still looks reasonable.
    """
    split = forecast_by_account(result.forecast, result.ledger)
    assert_split_ties(split, result.forecast)

    rolled = split.groupby(["period", "function", "sub_center"])["amount"].sum()
    original = result.forecast.set_index(["period", "function", "sub_center"])["forecast"]
    assert (rolled - original.reindex(rolled.index)).abs().max() < 1e-6


def test_split_covers_every_forecast_row(result):
    """No cost center may be dropped on the way into the ERP."""
    split = forecast_by_account(result.forecast, result.ledger)
    assert set(map(tuple, split[["function", "sub_center"]].drop_duplicates().to_numpy())) == set(
        map(tuple, result.forecast[["function", "sub_center"]].drop_duplicates().to_numpy())
    )
    assert split["amount"].gt(0).all(), "a forecast expense line must be positive"


def test_split_rejects_an_unmapped_cost_center(result):
    """A cost center with no account mix must raise, not silently vanish."""
    forecast = result.forecast.copy()
    forecast.loc[forecast.index[0], "sub_center"] = "Nonexistent Center"

    with pytest.raises(ValueError, match="no account mix"):
        forecast_by_account(forecast, result.ledger)


def test_assert_split_ties_catches_a_corrupted_split(result):
    """The tie assertion has to actually fail when the split is wrong."""
    split = forecast_by_account(result.forecast, result.ledger)
    split.loc[split.index[0], "amount"] += 1_000.0

    with pytest.raises(AssertionError, match="does not tie"):
        assert_split_ties(split, result.forecast)


def test_untrimmed_budget_covers_the_full_fiscal_year(settings, result):
    """The plan published to the ERP must span twelve months, not stop at today.

    A plan is committed in advance for the year. Truncating it at the last actual
    would leave the rolling forecast with nothing to be compared against for the
    months that have not happened — which is the only reason both versions are
    loaded into the ERP at all.
    """
    trimmed = build_budget(settings, result.ledger, trim_to_actuals=True)
    full = build_budget(settings, result.ledger, trim_to_actuals=False)

    assert len(full) > len(trimmed)

    # Every fiscal year that has a plan at all must have all twelve months of it.
    months_per_year = full.groupby(full["period"].dt.year)["period"].nunique()
    assert (months_per_year == 12).all()

    # The trimmed version is a strict subset — same numbers, fewer rows.
    key = ["period", "account", "function", "sub_center"]
    merged = trimmed.merge(full, on=key, suffixes=("_trim", "_full"))
    assert len(merged) == len(trimmed)
    assert (merged["budget_trim"] - merged["budget_full"]).abs().max() < 1e-9
