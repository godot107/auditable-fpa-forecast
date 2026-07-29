"""Landing page: what this is, the KPI header, and the data lineage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _shared import badge, gate_banner, metric_card, money, pipeline, setup
from fpa.kpi.pnl import balance_sheet_kpis, pnl_kpis
from fpa.kpi.process import process_kpis

setup("Overview")

result = pipeline()
settings = result.settings

st.title("Auditable FP&A Rolling Forecast")
st.markdown(
    f"**{settings.ticker}** · actuals from SEC EDGAR XBRL · "
    f"{result.quarterly.index.min():%b %Y} – {result.quarterly.index.max():%b %Y}"
)
st.caption(
    "Every actual traces to the accession number of the filing it was tagged in. "
    "Cost-center detail below the filed line items is modeled and asserted to foot "
    "back to the filed total. An LLM writes the variance commentary and never "
    "produces a number."
)

gate_open = gate_banner(result)
st.divider()

# --- KPI header -----------------------------------------------------------
pnl = pnl_kpis(result.quarterly)

# A period selector rather than a hardcoded tail. Twenty-six quarters of filed data
# were unreachable from the UI, which is a strange property for a page whose argument
# is that the history is auditable. Everything below follows the selection — KPIs,
# drivers, balance sheet, accession — because a page where half the cards move and
# half stay pinned to the latest quarter is worse than one that never moved at all.
quarters = list(pnl.index)
period = st.selectbox(
    "Filed quarter",
    options=quarters[::-1],  # newest first, so the default selection is the latest
    format_func=lambda p: f"FY{p.year} Q{p.quarter} — quarter ending {p:%b %Y}",
    key="home_period",
)
latest = pnl.loc[period]
is_latest = period == quarters[-1]

accn = result.accessions.query("account == 'revenue' and end == @period")
accn_note = f"accn {accn['accn'].iloc[0]}" if len(accn) else ""

st.subheader(
    f"Filed quarter — {period:%b %Y}" + ("  (latest)" if is_latest else "")
)
# Year over year against the same quarter a year earlier, not the prior quarter.
# Streaming revenue is seasonal, so a sequential comparison reads as performance when
# it is calendar.
prior_year = period - pd.offsets.DateOffset(years=1)
prior = pnl.loc[prior_year] if prior_year in pnl.index else None


def yoy(field: str) -> str:
    if prior is None or not np.isfinite(prior.get(field, np.nan)) or prior[field] == 0:
        return "no prior-year quarter filed"
    return f"{latest[field] / prior[field] - 1:+.1%} YoY"


cols = st.columns(4)
cards = [
    ("Revenue", money(latest["revenue"], "B"), "REAL", f"{yoy('revenue')} · {accn_note}"),
    ("EBIT", money(latest["ebit"], "B"), "REAL", yoy("ebit")),
    ("EBIT margin", f"{latest['ebit_margin']:.1%}", "REAL",
     "—" if prior is None else f"{(latest['ebit_margin'] - prior['ebit_margin']) * 100:+.1f} pts YoY"),
    ("Gross margin", f"{latest['gross_margin']:.1%}", "REAL",
     "—" if prior is None else f"{(latest['gross_margin'] - prior['gross_margin']) * 100:+.1f} pts YoY"),
]
for col, (label, value, kind, note) in zip(cols, cards):
    col.markdown(metric_card(label, value, kind, note), unsafe_allow_html=True)

cols = st.columns(4)
# Drivers are monthly and carry the quarter they belong to, so they can be matched to
# the selection exactly rather than by position. They move with the selector; a page
# where the KPI cards travel and the drivers stay pinned to the latest month would be
# quietly comparing two different periods.
driver_rows = result.drivers[result.drivers["quarter_end"] == period]
drivers = driver_rows.iloc[-1] if len(driver_rows) else result.drivers.iloc[-1]
process = process_kpis(result.controls, backtest_filed=result.backtest_filed)
cards = [
    ("Free cash flow", money(latest.get("free_cash_flow", float("nan")), "B"), "REAL", "CFO − capex"),
    ("Members", f"{drivers['members'] / 1e6:,.0f}M", "MODELED", "not disclosed in XBRL"),
    ("ARPU / month", f"${drivers['arpu']:,.2f}", "IMPLIED", "revenue ÷ members"),
    ("Control pass rate", f"{process['control_pass_rate']:.0%}", "REAL",
     f"{process['controls_blocking_failed']} blocking failures"),
]
for col, (label, value, kind, note) in zip(cols, cards):
    col.markdown(metric_card(label, value, kind, note), unsafe_allow_html=True)

st.divider()

# --- Revenue & EBIT -------------------------------------------------------
left, right = st.columns([3, 2])

with left:
    st.subheader("Filed revenue and EBIT")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=pnl.index, y=pnl["revenue"] / 1e9, name="Revenue ($B)",
               marker_color="#1565c0", opacity=0.85)
    )
    fig.add_trace(
        go.Scatter(x=pnl.index, y=pnl["ebit"] / 1e9, name="EBIT ($B)",
                   line=dict(color="#66bb6a", width=2.5))
    )
    fig.update_layout(
        template="plotly_dark", height=380, hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, width='stretch')
    st.caption(f"{badge('REAL')} every bar is a filed us-gaap fact.", unsafe_allow_html=True)

with right:
    st.subheader("Data lineage")
    st.markdown(
        """
```
SEC EDGAR XBRL          filed 10-Q / 10-K, accession per fact
   ↓ disaggregation     seeded; asserted to foot to the filed total
Odoo 17 (Postgres)      CoA · journal entries · analytic cost centers
   ↓ SQL extract
Parquet vintage         pinned; the demo never calls the network
   ↓
controls (blocking) → forecast → variance → narrative → approval
```
"""
    )
    st.markdown(
        f"{badge('REAL')} filed &nbsp; {badge('MODELED')} allocated here &nbsp; "
        f"{badge('IMPLIED')} forced by an identity &nbsp; {badge('FORECAST')} model output",
        unsafe_allow_html=True,
    )
    st.caption(
        "Odoo stands in for the client ERP (SAP / Oracle / NetSuite) that would feed "
        "OneStream in a real engagement. It is open-source, so it can actually be "
        "stood up and shown."
    )

# --- Balance sheet --------------------------------------------------------
bs = balance_sheet_kpis(result.quarterly)
if not bs.empty and "balance_check" in bs:
    st.divider()
    st.subheader(f"Balance sheet — {period:%b %Y}")
    cols = st.columns(4)
    filed_bs = bs.dropna(subset=["assets"])
    # Follows the selector; falls back to the latest filed position if the selected
    # quarter has no balance sheet, rather than showing a blank card.
    at_period = filed_bs[filed_bs.index <= period]
    last_bs = at_period.iloc[-1] if len(at_period) else filed_bs.iloc[-1]
    for col, (label, value, kind) in zip(
        cols,
        [
            ("Assets", money(last_bs["assets"], "B"), "REAL"),
            ("Liabilities", money(last_bs["liabilities"], "B"), "REAL"),
            ("Equity", money(last_bs["equity"], "B"), "REAL"),
            ("Leverage (L/E)", f"{last_bs['leverage']:.2f}x", "REAL"),
        ],
    ):
        col.markdown(metric_card(label, value, kind), unsafe_allow_html=True)

    worst = bs["balance_check"].abs().max()
    st.caption(
        f"Balance-sheet articulation on filed figures — worst |A − (L+E)| across "
        f"{bs['assets'].notna().sum()} quarters: **${worst:,.0f}**."
    )
