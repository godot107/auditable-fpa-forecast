"""Layer 1 — P&L and balance-sheet KPIs, computed from filed facts only.

Nothing here is modeled. Every input is a us-gaap tag Netflix filed, so every
output carries the same traceability as its inputs. Two are derived rather than
tagged, and both are derivations by definition rather than estimates:

* **Gross profit** — Netflix stopped tagging ``GrossProfit`` after 2020-12-31.
  Revenue minus cost of revenue *is* its definition.
* **Free cash flow** — never a single tag anywhere; cash from operations minus
  capital expenditure is the standard construction.
"""

from __future__ import annotations

import pandas as pd


def pnl_kpis(quarterly: pd.DataFrame) -> pd.DataFrame:
    """Return per-quarter P&L KPIs computed from filed line items."""
    q = quarterly
    out = pd.DataFrame(index=q.index)

    out["revenue"] = q["revenue"]
    out["gross_profit"] = q["revenue"] - q["cost_of_revenue"]
    out["gross_margin"] = out["gross_profit"] / q["revenue"]

    # EBIT is filed directly as OperatingIncomeLoss — not derived.
    out["ebit"] = q["operating_income"]
    out["ebit_margin"] = q["operating_income"] / q["revenue"]

    out["opex"] = q[["research_development", "marketing", "general_administrative"]].sum(axis=1)
    out["opex_ratio"] = out["opex"] / q["revenue"]
    out["rd_ratio"] = q["research_development"] / q["revenue"]
    out["marketing_ratio"] = q["marketing"] / q["revenue"]
    out["ga_ratio"] = q["general_administrative"] / q["revenue"]

    if "net_income" in q:
        out["net_income"] = q["net_income"]
        out["net_margin"] = q["net_income"] / q["revenue"]
    if {"income_tax", "net_income"} <= set(q.columns):
        pretax = q["net_income"] + q["income_tax"]
        out["effective_tax_rate"] = (q["income_tax"] / pretax).where(pretax != 0)

    if {"cash_from_operations", "capex"} <= set(q.columns):
        out["free_cash_flow"] = q["cash_from_operations"] - q["capex"]
        out["fcf_margin"] = out["free_cash_flow"] / q["revenue"]

    return out


def balance_sheet_kpis(quarterly: pd.DataFrame) -> pd.DataFrame:
    """Return balance-sheet and leverage KPIs where the instant facts are present.

    Balance-sheet tags are *instants* (a point-in-time balance), not durations, so
    they attach to the quarter-end date rather than to a period.
    """
    q = quarterly
    out = pd.DataFrame(index=q.index)

    if not {"assets", "liabilities", "equity"} <= set(q.columns):
        return out

    out["assets"] = q["assets"]
    out["liabilities"] = q["liabilities"]
    out["equity"] = q["equity"]
    out["leverage"] = q["liabilities"] / q["equity"].where(q["equity"] != 0)

    if "cash" in q:
        out["cash"] = q["cash"]
        out["cash_pct_assets"] = q["cash"] / q["assets"].where(q["assets"] != 0)

    # Balance-sheet articulation, as a displayed figure rather than a hidden one:
    # Assets - (Liabilities + Equity) should be zero on a filed balance sheet.
    out["balance_check"] = q["assets"] - (q["liabilities"] + q["equity"])

    return out
