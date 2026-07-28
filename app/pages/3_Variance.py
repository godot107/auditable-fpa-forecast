"""Variance page — budget vs actual, decomposed into spend and mix effects."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from _shared import badge, gate_banner, pipeline, setup
from fpa.variance import build_variance_report

setup("Variance")

result = pipeline()
st.title("Budget vs actual")

if not gate_banner(result):
    st.stop()

months = st.slider("Trailing months", 3, 36, 12, step=3)
report = build_variance_report(
    result.ledger, result.budget, result.revenue, result.revenue_budget,
    result.drivers, periods=months,
)
variance = report["by_cost_center"]
summary = report["summary"]

col1, col2, col3 = st.columns(3)
col1.metric("Actual opex", f"${summary['total_actual'] / 1e9:,.2f}B")
col2.metric("Budget", f"${summary['total_budget'] / 1e9:,.2f}B")
col3.metric(
    "Variance",
    f"${summary['total_variance'] / 1e9:,.2f}B",
    f"{summary['total_variance_pct'] * 100:,.1f}% vs plan",
    delta_color="inverse",  # over-spend is unfavourable
)

st.caption(
    "Expenses: a positive variance means actual came in **above** plan, which is "
    "unfavourable. Revenue: positive is favourable."
)

# --- The bridge -----------------------------------------------------------
st.subheader("Variance bridge by cost center")
st.caption(
    "**Spend effect** is what each center would have overrun by had every center grown "
    "proportionally. **Mix effect** is the rest — that center's share of the budget "
    "shifting. The two always sum to the variance, by construction."
)

fig = go.Figure()
labels = variance["function"] + " / " + variance["sub_center"]
fig.add_trace(go.Bar(x=labels, y=variance["spend_effect"] / 1e6, name="Spend effect ($M)",
                     marker_color="#1565c0"))
fig.add_trace(go.Bar(x=labels, y=variance["mix_effect"] / 1e6, name="Mix effect ($M)",
                     marker_color="#ef6c00"))
fig.update_layout(
    barmode="relative", template="plotly_dark", height=420,
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=10, b=10, l=10, r=10), xaxis_tickangle=-30,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
st.plotly_chart(fig, width="stretch")

# Drop attrs before display: they carry Timestamps, which Arrow cannot serialize.
display = variance.copy()
display.attrs = {}
for column in ("actual", "budget", "variance", "spend_effect", "mix_effect"):
    display[column] = (display[column] / 1e6).round(1)
display["variance_pct"] = (display["variance_pct"] * 100).round(1)
st.dataframe(display, width="stretch", hide_index=True)
st.caption("$M, except variance_pct (%).")

# --- Revenue drivers ------------------------------------------------------
st.subheader("Revenue: rate vs volume")
drivers = report["revenue_drivers"]
if drivers:
    cols = st.columns(4)
    cols[0].metric("Actual revenue", f"${drivers['actual_revenue'] / 1e9:,.2f}B")
    cols[1].metric("Plan", f"${drivers['planned_revenue'] / 1e9:,.2f}B")
    cols[2].metric("Actual ARPU", f"${drivers['actual_arpu']:,.2f}")
    cols[3].metric("Planned ARPU", f"${drivers['planned_arpu']:,.2f}")
    st.info(f"**What was measured:** {drivers['basis']}")
    st.markdown(
        f"{badge('MODELED')} members &nbsp; {badge('IMPLIED')} ARPU — "
        "their product reproduces filed revenue exactly.",
        unsafe_allow_html=True,
    )
