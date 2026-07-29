"""Variance page — budget vs actual, decomposed into spend and mix effects."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from _shared import badge, cost_center_disclaimer, gate_banner, pipeline, setup
from fpa.variance import build_variance_report

setup("Variance")

result = pipeline()
st.title("Budget vs actual")

if not gate_banner(result):
    st.stop()

cost_center_disclaimer("Every variance below is decomposed by cost center.")

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

# The worked example is computed from the frame on screen rather than written out, so
# it cannot go stale when the trailing-months slider moves. The mix effect is the part
# readers reliably misread — a centre can be over its own budget and still be the
# largest *favourable* contributor — so the example picks that row deliberately.
_most_favourable = variance.loc[variance["mix_effect"].idxmin()]
_most_unfavourable = variance.loc[variance["mix_effect"].idxmax()]
_total_variance = variance.attrs["total_variance"]
_share = _most_favourable["budget"] / variance["budget"].sum()

with st.expander("How to read spend effect and mix effect"):
    st.markdown(
        f"""
Total spend missed plan by **${_total_variance / 1e6:,.0f}M**. The bridge answers a
question the single number cannot: *was this everyone drifting together, or did the
shape of spending change?*

**Spend effect — your share if nothing changed shape.**

```
spend_effect = (this centre's budget / total budget) × total variance
```

Hold the planned mix fixed and hand every centre its proportional slice of the
company-wide miss. It sums to **${variance['spend_effect'].sum() / 1e6:,.0f}M** — the
whole variance — because the planned shares sum to one.

**Mix effect — how far you moved against the pack.**

```
mix_effect = this centre's variance − its spend effect
```

Whatever the proportional story cannot explain. It sums to
**${variance['mix_effect'].sum():,.2f}** — exactly zero, always. Mix is a
*redistribution*, not new money: for one centre to consume more of the budget than
planned, another must consume less. A mix column that did not net to zero would be
arithmetic, not analysis.

---

**The part that trips people up.**
`{_most_favourable['function']} / {_most_favourable['sub_center']}` is
**${_most_favourable['variance'] / 1e6:,.0f}M over its own budget** — and it is the
largest **favourable** row in the mix column at
**${_most_favourable['mix_effect'] / 1e6:,.0f}M**.

Both are true. It holds **{_share:.0%}** of planned spend, so a proportional share of a
${_total_variance / 1e6:,.0f}M overrun would have been
${_most_favourable['spend_effect'] / 1e6:,.0f}M. It came in well under that. It grew
more slowly than the company, so its share of total spend *fell* — even while it
overran in absolute terms.

The mirror image is `{_most_unfavourable['function']} / {_most_unfavourable['sub_center']}`:
a small centre by plan, absorbing **${_most_unfavourable['mix_effect'] / 1e6:,.0f}M** of
budget that the plan had pointed elsewhere.

**Why an FP&A team wants both.** Spend effect asks *"are we spending more than we said?"*
Mix effect asks *"are we spending it on what we said?"* A business can pass the first and
fail the second, and only the second shows a strategy quietly changing without a
re-plan — which is the thing a variance review is for.

**A caveat this decomposition earns.** There is no rate/volume split here. That needs a
*planned* quantity to vary actual against — planned headcount, planned cloud instance
hours — and no such plan exists in filed data. Rather than invent one, the split stops
at spend and mix, which are computable from budget and actual alone.
"""
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
