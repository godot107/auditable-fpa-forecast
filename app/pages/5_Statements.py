"""Three statements — and the identities that make them statements rather than tags.

A pile of XBRL concepts is not a set of financial statements. What makes them
statements is that they articulate: the balance sheet balances, the income
statement bridges to net income, the cash-flow statement explains the movement in
cash. Every one of those is computed here and gated by a blocking control.

The page leads with the articulation checks rather than the numbers, because that
is the order the numbers become trustworthy in.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _shared import badge, gate_banner, pipeline, setup
from fpa.config import (
    BALANCE_SHEET_ASSETS,
    BALANCE_SHEET_EQUITY,
    BALANCE_SHEET_LIABILITIES,
    DERIVED_BALANCE_SHEET,
)

setup("Statements")

result = pipeline()
st.title("Three statements, and proof they articulate")

if not gate_banner(result):
    st.stop()

bs, cf, is_ = result.balance_sheet, result.cash_flow, result.income_statement

# --- The articulation checks, first ----------------------------------------
st.subheader("Articulation")
st.caption(
    "These four identities are what separate a set of financial statements from a "
    "set of tags. Each is a **blocking** control: if one fails, this page does not "
    "render."
)

checks = {
    "balance_sheet_balances": "Assets = Liabilities + Equity",
    "balance_sheet_partition": "Detail lines partition the filed totals",
    "cash_flow_articulates": "CFO + CFI + CFF + FX = ΔCash",
    "net_income_bridge": "Pre-tax − tax = net income",
}
outcomes = {r.name: r for r in result.controls.results}
columns = st.columns(len(checks))
for column, (name, label) in zip(columns, checks.items()):
    outcome = outcomes.get(name)
    worst = (outcome.detail or {}).get("worst_absolute") if outcome else None
    column.metric(
        label,
        "PASS" if outcome and outcome.passed else "FAIL",
        f"worst ${worst:,.2f}" if worst is not None else None,
        delta_color="off",
    )

st.info(
    "**Why this matters for the ERP.** Because `A = L + E` holds to **$0.00**, the "
    "balance sheet posts to Odoo as a self-balancing journal entry with *no clearing "
    "account* — the debits and credits are the filed statement. Odoo would reject the "
    "entry outright if the statement did not balance, which makes the ERP an "
    "independent check on the ingest rather than just a destination for it."
)

# --- Balance sheet ----------------------------------------------------------
st.divider()
st.subheader("Balance sheet")

period = st.select_slider(
    "Quarter end",
    options=list(bs.index),
    value=bs.index[-1],
    format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
)
row = bs.loc[period]

derived_names = set(DERIVED_BALANCE_SHEET)


def _section(title: str, lines) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Code": line.code,
                "Line": line.name,
                "$M": round(row[line.account] / 1e6, 1),
                "Provenance": "IMPLIED" if line.account in derived_names else "REAL",
            }
            for line in lines
            if line.account in bs.columns
        ]
    )


left, right = st.columns(2)
with left:
    st.markdown("**Assets**")
    assets = _section("Assets", BALANCE_SHEET_ASSETS)
    st.dataframe(assets, width="stretch", hide_index=True)
    st.metric("Total assets", f"${row['assets'] / 1e9:,.2f}B")
with right:
    st.markdown("**Liabilities and equity**")
    credits = pd.concat(
        [_section("L", BALANCE_SHEET_LIABILITIES), _section("E", BALANCE_SHEET_EQUITY)],
        ignore_index=True,
    )
    st.dataframe(credits, width="stretch", hide_index=True)
    st.metric(
        "Total liabilities and equity",
        f"${(row['liabilities'] + row['equity']) / 1e9:,.2f}B",
    )

st.markdown(
    f"{badge('REAL')} filed line &nbsp; {badge('IMPLIED')} derived as a residual of "
    "filed subtotals against filed detail",
    unsafe_allow_html=True,
)
st.caption(
    "Three of Netflix's largest lines are not obtainable as tags. **Content assets** "
    "sit under a company extension that `companyfacts` does not expose; **treasury "
    "stock** stopped being tagged in 2022-03-31 while the buyback programme continued; "
    "**content liabilities** are presented but not tagged. Each is derived as a "
    "residual of filed figures — and a residual is only honest if it is named as the "
    "line it represents and its sign is checked, which `derived_balance_sheet_lines` "
    "does on every run."
)

with st.expander("Filed disclosure tags — reported, never posted"):
    st.caption(
        "The lease tags are ASC 842 *disclosures* nested **inside** the "
        "'other non-current' captions, not siblings of them. Posting them alongside "
        "double-counts, which showed up as a non-current liability of **−\\$571M** — a "
        "negative liability, and the only reason the error was visible at all. "
        "Settled by the adoption step in 2019-Q1: `OtherAssetsNoncurrent` jumped "
        "\\$816M in the quarter \\$812M of right-of-use assets were first recognised. A "
        "caption that jumps by the amount of the thing being adopted contains it."
    )
    disclosed = [c for c in bs.columns if c.endswith("_disclosed")]
    if disclosed:
        table = (bs[disclosed] / 1e6).round(1).tail(8)
        table.index = table.index.strftime("%Y-%m-%d")
        st.dataframe(table, width="stretch")

# --- Cash flow --------------------------------------------------------------
st.divider()
st.subheader("Cash flow")
st.caption(
    "Built in Python and **not** posted to the ERP, deliberately: no general ledger "
    "journalizes a cash-flow statement. It is derived from the movement in balance "
    "sheet accounts, which is why a consolidation tool computes it and a "
    "transactional system does not."
)

recent = cf.tail(12)
fig = go.Figure()
for section, color in [
    ("cash_from_operations", "#2e7d32"),
    ("cash_from_investing", "#1565c0"),
    ("cash_from_financing", "#ef6c00"),
]:
    if section in recent.columns:
        fig.add_trace(
            go.Bar(
                x=recent.index,
                y=recent[section] / 1e6,
                name=section.replace("cash_from_", "").title(),
                marker_color=color,
            )
        )
fig.add_trace(
    go.Scatter(
        x=recent.index,
        y=recent["net_change"] / 1e6,
        name="Net change in cash",
        mode="lines+markers",
        line=dict(color="#e0e0e0", width=2),
    )
)
fig.update_layout(
    barmode="relative",
    template="plotly_dark",
    height=380,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=10, b=10, l=10, r=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    yaxis_title="$M",
)
st.plotly_chart(fig, width="stretch")

worst_residual = cf["roll_forward_residual"].abs().max()
st.success(
    f"Roll-forward explains the movement in cash across "
    f"{int(cf['roll_forward_residual'].notna().sum())} quarters — worst residual "
    f"**${worst_residual:,.2f}**. Reconciled against cash *including restricted* "
    "cash, which is a different concept from the balance-sheet cash line under "
    "ASU 2016-18; substituting one for the other breaks the reconciliation by "
    "exactly the restricted balance."
)

if "free_cash_flow" in cf.columns:
    st.metric(
        "Trailing four-quarter free cash flow",
        f"${cf['free_cash_flow'].tail(4).sum() / 1e9:,.2f}B",
        help="Cash from operations less capital expenditure, both filed.",
    )

# --- Income statement -------------------------------------------------------
st.divider()
st.subheader("Income statement")

display = (is_.tail(8) / 1e6).round(1)
for per_share in ("eps_basic", "eps_diluted"):
    if per_share in is_.columns:
        display[per_share] = is_[per_share].tail(8).round(2)
display.index = display.index.strftime("%Y-%m-%d")
st.dataframe(display.T, width="stretch")
st.caption(
    "\\$M, except EPS (\\$/share). `gross_profit` is computed rather than read — Netflix "
    "stopped tagging `GrossProfit` after 2020-12-31, and Revenue − CostOfRevenue is "
    "its definition. `non_operating` is the residual between operating and pre-tax "
    "income, which is the only route to a complete bridge since `InterestExpense` was "
    "discontinued after 2024-09-30."
)

eps_outcome = outcomes.get("eps_consistency")
if eps_outcome:
    st.caption(
        f"**Unit handling:** {eps_outcome.message} — the ingest reads three different "
        "XBRL units here (`USD`, `shares`, `USD/shares`). A regression to hardcoded "
        "`USD` would not raise; the per-share tags would silently vanish. This is the "
        "control that would notice."
    )
