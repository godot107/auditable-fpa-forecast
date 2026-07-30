"""A three-statement forecast: one series is forecast, the rest are derived.

The README has claimed all along that you should *forecast the drivers and let the margin
fall out*, and nothing in the code did that — `operating_income` was extrapolated directly
like any other line, and it was the worst row in the backtest at MASE 1.160. This module
implements the claim so it can be **tested** rather than asserted.

Only **revenue** is forecast. Everything else follows:

    revenue            ETS on filed quarterly data — MASE 0.343, the one series that works
      └─ expenses      trailing cost ratios x forecast revenue
           └─ EBIT     DERIVED as revenue - expenses, never forecast
                └─ net income   effective tax rate on derived pretax
                     └─ balance sheet   retained earnings roll + working-capital days,
                                        with cash as the plug that forces A = L + E

Three properties make this worth having over a pile of independent extrapolations:

* **Coherence.** Forecast seventeen balance-sheet lines separately and they will not balance
  at any future date. `A = L + E` holds to $0.00 across 26 *actual* quarters and a blocking
  control proves it; a forecast that violated the identity the actuals maintain would be the
  worst inconsistency in the project. Here cash absorbs the residual by construction, exactly
  as it does in every corporate three-statement model.
* **Every input is filed.** No cost center, no invented granularity. The ratios come from
  filed history and the driver is a filed line.
* **It is falsifiable.** :func:`compare_derived_vs_direct` scores derived EBIT against
  directly forecasting EBIT on the same rolling-origin folds. If derivation does not win, the
  project's own advice was wrong and the report says so.

**Free cash flow is deliberately absent.** It backtests at MASE 7.571 — seven times worse
than a naive forecast — because it is a difference of two lumpy series. Reporting that it
cannot be forecast is more useful than reporting a number.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fpa.config import EXPENSE_ACCOUNTS
from fpa.forecast.backtest import mase, rolling_origin_splits
from fpa.forecast.models import MODELS

logger = logging.getLogger(__name__)

# Trailing quarters used for every ratio. Four, not the full history, and that is a measured
# choice: cost of revenue averages 56.6% of revenue across all 26 quarters but 51.0% over the
# last four, because the mix has shifted structurally. A full-history ratio would forecast a
# margin the business left behind years ago.
RATIO_WINDOW = 4

DAYS_PER_QUARTER = 365.25 / 4

# Working-capital lines, and the flow each one scales with. Payables track cost, not revenue —
# attaching them to revenue would make a margin change look like a payment-terms change.
WORKING_CAPITAL: dict[str, str] = {
    "deferred_revenue": "revenue",
    "accounts_payable": "cost_of_revenue",
    "accrued_liabilities": "revenue",
}

# Balance-sheet lines held at their last filed value, stated rather than buried. Each is either
# a financing decision this model does not attempt to predict (debt, buybacks) or a slow
# accrual whose quarter-to-quarter movement is small relative to the lines above.
HELD_FLAT = (
    "short_term_investments",
    "ppe_net",
    "other_assets_noncurrent",
    "long_term_debt",
    "other_liabilities_noncurrent",
    "common_stock",
    "aoci",
)


def cost_ratios(quarterly: pd.DataFrame, *, window: int = RATIO_WINDOW) -> pd.Series:
    """Each expense line as a share of revenue, averaged over the trailing window."""
    revenue = quarterly["revenue"]
    return pd.Series(
        {
            account: float((quarterly[account] / revenue).tail(window).mean())
            for account in EXPENSE_ACCOUNTS
            if account in quarterly.columns
        }
    )


def working_capital_days(quarterly: pd.DataFrame, *, window: int = RATIO_WINDOW) -> pd.Series:
    """Working-capital balances expressed as days of the flow they attach to.

    Days rather than a percentage because days are the unit an FP&A team actually negotiates
    and reports — "we are carrying 13 days of deferred revenue" is a sentence a CFO acts on.
    """
    out = {}
    for balance, flow in WORKING_CAPITAL.items():
        if balance in quarterly.columns and flow in quarterly.columns:
            ratio = (quarterly[balance] / quarterly[flow]).tail(window).mean()
            out[balance] = float(ratio * DAYS_PER_QUARTER)
    return pd.Series(out)


def _future_quarters(last: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(last + pd.offsets.QuarterEnd(1), periods=horizon, freq="QE")


def forecast_income_statement(
    quarterly: pd.DataFrame,
    horizon: int = 4,
    *,
    model: str = "ets",
    window: int = RATIO_WINDOW,
) -> pd.DataFrame:
    """Forecast revenue, drive the expenses off it, derive EBIT and net income.

    ``operating_income`` is **not** in the output because it was forecast — it is the
    residual ``revenue - sum(expenses)``. That is the entire point of the module.
    """
    history = quarterly.dropna(subset=["revenue"])
    revenue_path = MODELS[model](history["revenue"], horizon, period=4)
    index = _future_quarters(history.index[-1], horizon)

    out = pd.DataFrame({"revenue": revenue_path}, index=index)
    ratios = cost_ratios(history, window=window)
    for account, ratio in ratios.items():
        out[account] = out["revenue"] * ratio

    # Derived, not forecast. If a cost line is missing from the ratios the residual would be
    # silently wrong, so require the full set rather than summing what happens to be present.
    missing = [a for a in EXPENSE_ACCOUNTS if a not in out.columns]
    if missing:
        raise ValueError(f"cannot derive operating income without {missing}")
    out["operating_income"] = out["revenue"] - out[list(EXPENSE_ACCOUNTS)].sum(axis=1)

    # Below the line: the gap between EBIT and pretax income, and the effective tax rate, both
    # from trailing actuals. Interest expense stopped being tagged after 2024-09-30, so the gap
    # is carried as one figure rather than pretending its components are still observable.
    below_line = (history["pretax_income"] - history["operating_income"]).tail(window).mean()
    tax_rate = (history["income_tax"] / history["pretax_income"]).tail(window).mean()
    out["pretax_income"] = out["operating_income"] + float(below_line)
    out["income_tax"] = out["pretax_income"] * float(tax_rate)
    out["net_income"] = out["pretax_income"] - out["income_tax"]

    out.index.name = "period"
    out.attrs["model"] = model
    out.attrs["cost_ratios"] = ratios.to_dict()
    out.attrs["below_line"] = float(below_line)
    out.attrs["tax_rate"] = float(tax_rate)
    return out


EQUITY_LINES = ("common_stock", "retained_earnings", "treasury_stock", "aoci")
LIABILITY_LINES = (
    "accounts_payable", "accrued_liabilities", "deferred_revenue", "long_term_debt",
    "other_liabilities_noncurrent", "content_liabilities_current",
    "content_liabilities_noncurrent",
)
NON_CASH_ASSETS = (
    "short_term_investments", "ppe_net", "other_assets_noncurrent", "content_assets",
    "other_current_assets",
)
# Scale with content spend, which lands in cost of revenue. Holding them flat would shrink the
# largest asset on the sheet relative to a growing business.
CONTENT_SCALED = ("content_assets", "content_liabilities_current", "content_liabilities_noncurrent")


def forecast_balance_sheet(
    balance_sheet: pd.DataFrame,
    quarterly: pd.DataFrame,
    income: pd.DataFrame,
    *,
    window: int = RATIO_WINDOW,
) -> pd.DataFrame:
    """Derive the balance sheet from the forecast P&L, with cash as the plug.

    Takes the **balance-sheet frame**, not ``quarterly``. Three of the largest lines —
    ``content_assets``, ``treasury_stock`` and the content liabilities — are *derived*
    residuals produced by ``fpa.ingest.statements`` and are absent from ``quarterly``
    entirely. An earlier version read ``quarterly`` and guarded each line with
    ``if column in history.columns``, so treasury stock was silently skipped: equity was
    projected with **no contra-equity at all**, overstating it by roughly $28B, and the cash
    plug absorbed the error and grew to $84B against an actual $9B.

    That is this codebase's signature defect for the fourth time — a missing input producing a
    plausible number instead of an error — so the lines are now **required**, and a missing one
    raises.

    Nothing here is extrapolated:

    * ``retained_earnings`` rolls on net income. Verified against actuals: ``RE_t - RE_t-1``
      equals net income to within **$0.4M** across 26 quarters, because this filer books
      repurchases to treasury stock rather than to retained earnings.
    * ``treasury_stock`` rolls on forecast buybacks. Also verified: the quarterly movement
      tracks the filed ``PaymentsForRepurchaseOfCommonStock`` to within ~$40M.
    * Working capital comes from **days** of the flow it attaches to.
    * Content lines scale with cost of revenue.
    * Everything else is held flat and named in :data:`HELD_FLAT`.

    Cash then closes ``A = L + E``. A plug makes the identity true by construction, which
    makes it useless as a test — so :func:`plug_plausibility` exists to check the plug is not
    quietly absorbing a modelling error, which is exactly how the $84B was caught.
    """
    history = balance_sheet.dropna(subset=["assets"])
    last = history.iloc[-1]
    days = working_capital_days(quarterly, window=window)

    required = set(EQUITY_LINES) | set(LIABILITY_LINES) | set(NON_CASH_ASSETS)
    missing = sorted(required - set(history.columns))
    if missing:
        raise ValueError(
            f"balance sheet is missing required lines {missing} — refusing to project a "
            "partial sheet, because the cash plug would silently absorb the gap"
        )

    # Buybacks are filed. Trailing average rather than a forecast: a repurchase programme is a
    # capital-allocation decision, and the point is to fund it, not to predict it.
    buybacks = float(quarterly["buybacks"].tail(window).mean())

    rows = []
    retained = float(last["retained_earnings"])
    treasury = float(last["treasury_stock"])

    for period, quarter in income.iterrows():
        retained += float(quarter["net_income"])
        treasury -= buybacks
        row: dict[str, float] = {"retained_earnings": retained, "treasury_stock": treasury}

        for balance, flow in WORKING_CAPITAL.items():
            row[balance] = float(quarter[flow]) * days[balance] / DAYS_PER_QUARTER

        for column in HELD_FLAT:
            row[column] = float(last[column])

        for column in CONTENT_SCALED:
            ratio = (history[column] / quarterly["cost_of_revenue"]).tail(window).mean()
            row[column] = float(quarter["cost_of_revenue"]) * float(ratio)

        row["other_current_assets"] = float(last["other_current_assets"])
        rows.append(pd.Series(row, name=period))

    frame = pd.DataFrame(rows)
    non_cash = frame[list(NON_CASH_ASSETS)].sum(axis=1)

    frame["equity"] = frame[list(EQUITY_LINES)].sum(axis=1)
    frame["liabilities"] = frame[list(LIABILITY_LINES)].sum(axis=1)
    frame["cash"] = frame["equity"] + frame["liabilities"] - non_cash
    frame["assets"] = frame["cash"] + non_cash
    frame["balance_check"] = frame["assets"] - frame["liabilities"] - frame["equity"]

    frame.index.name = "period"
    frame.attrs["working_capital_days"] = days.to_dict()
    frame.attrs["held_flat"] = HELD_FLAT
    frame.attrs["plug"] = "cash"
    frame.attrs["buybacks_per_quarter"] = buybacks
    frame.attrs["opening_cash"] = float(last["cash"])
    return frame


def plug_plausibility(
    balance: pd.DataFrame, quarterly: pd.DataFrame, income: pd.DataFrame, *, window: int = RATIO_WINDOW
) -> pd.DataFrame:
    """Check the cash plug against an independent cash roll-forward.

    Cash closing ``A = L + E`` makes the identity untestable, so it has to be tested another
    way: build the change in cash from the cash-flow drivers — net income, non-cash charges,
    capex, buybacks — and compare it to what the plug did. A large divergence means the plug is
    absorbing a modelling error rather than reflecting one.

    This is not a control; it is a diagnostic, because a genuine working-capital swing will
    also open a gap. It exists because it is what caught an $84B cash forecast produced by a
    silently omitted contra-equity line.
    """
    non_cash_charges = float(
        (quarterly["depreciation_amortization"] + quarterly["stock_compensation"]).tail(window).mean()
    )
    capex_ratio = float((quarterly["capex"] / quarterly["revenue"]).tail(window).mean())
    buybacks = balance.attrs["buybacks_per_quarter"]

    plug_delta = balance["cash"].diff()
    plug_delta.iloc[0] = balance["cash"].iloc[0] - balance.attrs["opening_cash"]

    implied = (
        income["net_income"] + non_cash_charges - income["revenue"] * capex_ratio - buybacks
    )
    return pd.DataFrame(
        {
            "plug_change_in_cash": plug_delta,
            "roll_forward_change": implied,
            "gap": plug_delta - implied,
            "gap_pct_of_revenue": (plug_delta - implied) / income["revenue"],
        }
    )


def compare_derived_vs_direct(
    quarterly: pd.DataFrame,
    *,
    horizon: int = 4,
    folds: int = 4,
    model: str = "ets",
    window: int = RATIO_WINDOW,
) -> pd.DataFrame:
    """Score derived EBIT against directly forecasting EBIT, on identical folds.

    The experiment the README's advice implies and never ran. Same rolling origin, same
    horizon, same model for the driver — the only difference is whether operating income is
    extrapolated from its own history or computed as revenue minus driven costs.

    A result either way is worth having. If derivation loses, the claim was wrong.
    """
    frame = quarterly.dropna(subset=["revenue", "operating_income", "pretax_income", "income_tax"])
    values = frame["operating_income"].to_numpy(float)

    records = []
    for fold, (train_idx, test_idx) in enumerate(
        rolling_origin_splits(len(frame), horizon, folds=folds), start=1
    ):
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
        actual = test["operating_income"].to_numpy(float)

        derived = forecast_income_statement(
            train, horizon, model=model, window=window
        )["operating_income"].to_numpy(float)
        direct = MODELS[model](train["operating_income"], horizon, period=4)

        for label, predicted in (("derived", derived), ("direct", direct)):
            records.append(
                {
                    "fold": fold,
                    "approach": label,
                    "mase": mase(actual, predicted, values[train_idx], period=4),
                    "train_quarters": len(train_idx),
                }
            )
    return pd.DataFrame(records)


def statement_report_markdown(
    income: pd.DataFrame, balance: pd.DataFrame, comparison: pd.DataFrame
) -> str:
    """Report the forecast and the experiment that justifies its structure."""
    worst = float(balance["balance_check"].abs().max())
    by_approach = comparison.groupby("approach")["mase"].mean()
    derived, direct = by_approach.get("derived", float("nan")), by_approach.get("direct", float("nan"))

    if derived < direct:
        verdict = f"**Derivation wins** — {(1 - derived / direct):.1%} lower error."
    else:
        verdict = (
            f"**Derivation loses, and that is the finding.** The advice this repo has been "
            f"giving was wrong for this series."
        )
    # Both approaches can be worse than doing nothing, and that has to be said in the same
    # breath. A relative win between two losing methods is not a result to quote on its own.
    if min(derived, direct) >= 1.0:
        verdict += (
            "\n\n**But both lose to seasonal-naive.** Every value here is above 1.0, so on "
            "operating income neither approach beats *"
            "same quarter last year*. Deriving it is the better of two methods that should not "
            "be used for guidance — which is the same conclusion the per-series table reaches, "
            "arrived at a second way."
        )

    lines = [
        "## Three-statement forecast — one series forecast, the rest derived",
        "",
        f"Only `revenue` is forecast ({income.attrs['model']}). Expenses are driven off it by "
        f"trailing {RATIO_WINDOW}-quarter cost ratios, `operating_income` is the residual, and "
        "the balance sheet is derived with **cash as the plug**.",
        "",
        f"**Articulation on the forecast:** worst `|A − (L+E)|` across "
        f"{len(balance)} forecast quarters is **${worst:,.2f}**. The identity that a blocking "
        "control proves on actuals is not broken by the projection — cash closes it by "
        "construction.",
        "",
        "### Does deriving EBIT beat forecasting it?",
        "",
        "| approach | mean MASE |",
        "|---|---|",
        f"| revenue-driven, EBIT derived | **{derived:.3f}** |",
        f"| EBIT extrapolated directly | {direct:.3f} |",
        "",
        verdict,
        "",
        "Same rolling-origin folds, same horizon, same model for the driver. The only "
        "difference is whether operating income comes from its own history or from revenue "
        "minus driven costs.",
        "",
        "### Assumptions, stated",
        "",
        "| driver | value |",
        "|---|---|",
    ]
    for account, ratio in income.attrs["cost_ratios"].items():
        lines.append(f"| `{account}` as % of revenue | {ratio:.1%} |")
    for balance_line, days in balance.attrs["working_capital_days"].items():
        lines.append(f"| `{balance_line}` | {days:.1f} days |")
    lines += [
        f"| effective tax rate | {income.attrs['tax_rate']:.1%} |",
        f"| below-the-line gap (EBIT → pretax) | ${income.attrs['below_line'] / 1e6:,.1f}M |",
        "",
        f"Held flat and named rather than hidden: `{'`, `'.join(HELD_FLAT)}`. Each is either a "
        "financing decision this model does not predict or a slow accrual whose movement is "
        "small next to the lines above.",
        "",
        f"`treasury_stock` is **not** held flat — it rolls on "
        f"${balance.attrs['buybacks_per_quarter'] / 1e9:,.2f}B of buybacks per quarter, the "
        "trailing filed average. It has to: retained earnings compounding on net income with "
        "no contra-equity to fund the repurchases overstated equity by ~$28B in an earlier "
        "version, and the cash plug absorbed it and reached $84B against an actual $9B.",
        "",
        "**Free cash flow is not forecast.** It backtests at MASE 7.571, seven times worse than "
        "a naive forecast, because it is a difference of two lumpy series. That measurement is "
        "the deliverable.",
    ]
    return "\n".join(lines)
