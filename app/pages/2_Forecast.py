"""Forecast page — the bottom-up hierarchy forecast and the two honest backtests."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _shared import badge, cost_center_disclaimer, gate_banner, pipeline, setup
from fpa.forecast.models import leaf_series
from fpa.forecast.posterior import load_posterior, posterior_intervals, stale_series

setup("Forecast")

result = pipeline()

st.title("Forecast & validation")

if not gate_banner(result):
    st.stop()  # the gate: nothing below is computed when controls fail

cost_center_disclaimer("Everything forecast on this page is a cost center.")

st.caption(
    "Forecasts are produced bottom-up: each leaf cost center is forecast "
    "independently and parents are formed by aggregation, so the hierarchy is "
    "coherent by construction — cost-center forecasts always sum to the total."
)

# --- Forecast chart -------------------------------------------------------
ledger = result.ledger
forecast = result.forecast

actual_total = ledger.groupby("period")["amount"].sum()
forecast_total = forecast.groupby("period")["forecast"].sum()

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=actual_total.index, y=actual_total / 1e6, name="Actual opex ($M)",
               line=dict(color="#29b6f6", width=2))
)
fig.add_trace(
    go.Scatter(x=forecast_total.index, y=forecast_total / 1e6, name="Forecast ($M)",
               line=dict(color="#ab47bc", width=2, dash="dot"))
)
budget_total = result.budget.groupby("period")["budget"].sum()
fig.add_trace(
    go.Scatter(x=budget_total.index, y=budget_total / 1e6, name="Budget ($M)",
               line=dict(color="#ffa726", width=1.5, dash="dash"))
)
fig.update_layout(
    template="plotly_dark", height=420, hovermode="x unified",
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=10, b=10, l=10, r=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
st.plotly_chart(fig, width='stretch')
st.markdown(
    f"{badge('MODELED')} monthly actuals are filed quarters disaggregated here. "
    f"{badge('FORECAST')} {result.settings.horizon_months}-month bottom-up forecast.",
    unsafe_allow_html=True,
)

st.divider()

# --- The honest validation ------------------------------------------------
st.markdown(result.validation_markdown())

st.divider()

# --- Per-series detail ----------------------------------------------------
st.subheader("Accuracy by series (filed quarterly data)")
by_model = result.backtest_filed.by_model.pivot(
    index="series", columns="model", values="mase"
).round(3)
# An explicit verdict column rather than a colour gradient: it states the finding
# instead of leaving the reader to decode a shade, and avoids pulling in
# matplotlib purely to tint a table.
best = by_model.drop(columns=["seasonal_naive"], errors="ignore").min(axis=1)
by_model.insert(0, "beats naive?", ["yes" if v < 1.0 else "NO" for v in best])
st.dataframe(by_model, width='stretch')
st.caption(
    "MASE below 1.0 beats a seasonal-naive benchmark; above 1.0 loses to it. "
    "Read `operating_income` — it is the hardest row, because a margin is a small "
    "difference between two large numbers, so modest errors either side compound "
    "into a large error in the residual."
)

st.subheader("Error growth across the forecast horizon")
by_horizon = result.backtest_monthly.by_horizon() / 1e6
fig2 = go.Figure()
for column in by_horizon.columns:
    fig2.add_trace(go.Scatter(x=by_horizon.index, y=by_horizon[column], name=column, mode="lines+markers"))
fig2.update_layout(
    template="plotly_dark", height=320,
    xaxis_title="months ahead", yaxis_title="mean absolute error ($M)",
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=10, b=10, l=10, r=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
st.plotly_chart(fig2, width='stretch')
st.caption(
    "Error should grow with horizon. A model whose 12-month error matches its "
    "1-month error is usually leaking future information."
)

# --- Cost-center forecast table -------------------------------------------
st.subheader("Forecast by cost center")
pivot = forecast.pivot_table(
    index="period", columns=["function", "sub_center"], values="forecast", aggfunc="sum"
)
st.dataframe((pivot / 1e6).round(1), width='stretch')
st.caption("$M. Columns sum to the total forecast above — bottom-up reconciliation.")

# --- Posterior-predictive intervals ---------------------------------------
st.divider()
st.subheader("Posterior-predictive interval")
st.caption(
    "A NumPyro **local linear trend** with monthly seasonality, fitted on the log scale "
    "with NUTS. Every posterior draw continues the random walk from *its own* level, "
    "trend and variances, so the fan carries parameter uncertainty rather than being one "
    "fitted path with an error band attached. The draws are **pinned like the data "
    "vintage** — fitted once offline and committed — so this page forward-simulates them "
    "in NumPy and never runs a sampler."
)

with st.expander("What the fan actually is — and why it is the honest version of a scenario"):
    st.markdown(
        """
**Start with what it is not.** It is not one forecast with error bars bolted on.

The model has seen 78 months of this cost center. Instead of fitting a single "best"
line through them, the sampler explores **every combination of starting level, trend and
seasonal shape that is consistent with what actually happened** — thousands of them.
Each of those is then run forward twelve months, carrying *its own* trend and *its own*
volatility. The shaded band is where 80% of those futures land.

So the width is not a confidence dressing. It is the answer to: *given everything this
line has ever done, how much disagreement is there about where it goes next?*

---

### Why this is a scenario set

FP&A already plans against **downside / base / upside**. Those cases are normally
*chosen* — "call the downside base minus 10%." It is a round number somebody picked, and
nobody can say how likely it is.

`p10`, `p50`, `p90` are those same three cases, **measured instead of chosen**:

| Planning case | Here | Meaning |
|---|---|---|
| Downside | `p10` | 1 year in 10 comes in below this |
| Base | `p50` | as likely to be over as under |
| Upside | `p90` | 1 year in 10 comes in above this |

The downside is not a haircut someone agreed to in a meeting. It is the tenth percentile
of the futures this cost center's own history supports. **That is the whole value: a
planning range with a stated probability attached, derived from measured variability
rather than negotiated.**

And note the band *widens* with horizon. Month 12 is genuinely less knowable than month
1, and the chart says so. A forecast quoted as a single number hides exactly that.

---

### What it cannot tell you, and this matters

It answers **"what range should I plan against if nothing changes?"**

It does **not** answer **"what happens if we cut content spend 15%?"**

That second question is an *intervention* — a different future, not an uncertain one.
Answering it needs to know how the business responds to a decision it has never made,
and nothing in filed data identifies that. This project declines to guess (see the
scenario note in the README).

> The fan is uncertainty about an **unchanged** future.
> A what-if is a **different** future.

Reading the first as the second is how a forecast quietly becomes a promise.
"""
    )

leaves = leaf_series(result.ledger)
choice = st.selectbox(
    "Cost center",
    options=list(leaves.columns),
    format_func=lambda c: f"{c[0]} / {c[1]}",
    key="bayes_center",
)

# Read a pinned posterior rather than fitting one. Sampling at read time was wrong
# twice: it cannot run on the hosted deploy at all, and locally it took minutes behind
# a button that claimed fifteen seconds. The draws are fitted once, offline, and
# committed the same way the data vintage is — so serving a fan is a NumPy forward
# simulation in milliseconds, with no JAX anywhere on this page.
posterior = load_posterior(result.settings)
history = leaves[choice].dropna()
bands = None

if posterior is None:
    st.info(
        "**No pinned posterior in this vintage.** Build it with "
        "`python -m fpa.forecast.posterior` — nine NUTS fits, a few minutes, offline. "
        "It writes ~1 MB of draws that this page reads directly. Measured calibration "
        "lives in `reports/interval_calibration.md` regardless."
    )
elif choice in stale_series(posterior, leaves):
    # The stamp is the point. A cached posterior is a number computed from inputs that
    # may since have moved, and nothing about the resulting fan would look wrong.
    st.error(
        f"**The stored posterior does not match the current data for "
        f"`{choice[0]} / {choice[1]}`.** The digest of the series on disk differs from "
        "the one these draws were fitted to — the ledger has changed since the fit. "
        "Rebuild with `python -m fpa.forecast.posterior`. No interval is shown, because "
        "a fan drawn from a stale posterior looks exactly like a valid one."
    )
else:
    bands = posterior_intervals(posterior, choice, result.settings.horizon_months)
    if not bands.attrs["converged"]:
        st.warning(
            f"**This fit did not converge** — R-hat {bands.attrs['worst_rhat']:.3f} "
            f"(ceiling 1.01), ESS {bands.attrs['min_ess']:.0f} (floor 400). The interval "
            "below is computed from a posterior the sampler did not fully explore. It is "
            "shown because hiding it would be worse than labelling it, but it is not "
            "evidence of anything."
        )

if bands is not None:

    fan = go.Figure()
    fan.add_trace(go.Scatter(
        x=history.index[-24:], y=history.to_numpy()[-24:] / 1e6,
        name="Actual", mode="lines", line=dict(color="#e0e0e0", width=2),
    ))
    fan.add_trace(go.Scatter(
        x=bands.index, y=bands["p90"] / 1e6, name="p90",
        mode="lines", line=dict(width=0), showlegend=False,
    ))
    fan.add_trace(go.Scatter(
        x=bands.index, y=bands["p10"] / 1e6, name="80% interval",
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(21,101,192,0.30)",
    ))
    fan.add_trace(go.Scatter(
        x=bands.index, y=bands["p50"] / 1e6, name="p50",
        mode="lines", line=dict(color="#1565c0", width=2, dash="dash"),
    ))
    fan.update_layout(
        template="plotly_dark", height=380, yaxis_title="$M",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fan, width="stretch")

    width = bands["p90"] - bands["p10"]
    columns = st.columns(5)
    columns[0].metric("Interval width, month 1", f"${width.iloc[0] / 1e6:,.0f}M")
    columns[1].metric(
        "Month 12", f"${width.iloc[-1] / 1e6:,.0f}M",
        f"{width.iloc[-1] / width.iloc[0]:.1f}x wider", delta_color="off",
    )
    # All three diagnostics on screen, not just the one that looks reassuring. This is
    # the page's own thesis applied to itself: divergences alone read healthy on a fit
    # whose chains had frozen.
    columns[2].metric("Divergent transitions", f"{bands.attrs.get('divergences', 0)}")
    columns[3].metric(
        "R-hat (worst)", f"{bands.attrs['worst_rhat']:.3f}",
        "ok" if bands.attrs["worst_rhat"] <= 1.01 else "above 1.01", delta_color="off",
    )
    columns[4].metric(
        "ESS (min)", f"{bands.attrs['min_ess']:,.0f}",
        "ok" if bands.attrs["min_ess"] >= 400 else "below 400", delta_color="off",
    )
    st.caption(
        "The band **must** widen with horizon — a fitted straight line would grow like "
        "√h and understate long-horizon risk, which is the reason for an integrated "
        "random walk. Divergent transitions are the sampler reporting where it could not "
        "explore the posterior, and they are **never read alone**: `target_accept_prob` "
        "sits at 0.90, not the 0.99 an earlier version used. Tuning against divergences "
        "by itself drives it upward until the step size collapses, at which point nothing "
        "diverges because nothing moves — measured here at 0.99 as zero divergences with "
        "R-hat past 1000 and an ESS of 2."
    )

st.info(
    "**Calibration is measured, and currently provisional.** `python -m fpa --intervals` "
    "runs a rolling-origin evaluation of all nine cost centers on coverage **and** "
    "sharpness, and writes `reports/interval_calibration.md` with R-hat and ESS per "
    "series. **8 of 9 backtest fits do not converge** — rolling origin trains on 42-66 "
    "months instead of 78, and short series identify the scales poorly. Two of the three "
    "series that beat the naive benchmark are the worst-converged rows in the table, and "
    "the one clean fit loses to it. The layer is not yet ready to carry a headline."
)
