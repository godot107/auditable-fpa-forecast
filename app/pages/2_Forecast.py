"""Forecast page — the bottom-up hierarchy forecast and the two honest backtests."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _shared import badge, gate_banner, pipeline, setup
from fpa.forecast.models import leaf_series

setup("Forecast")

result = pipeline()

st.title("Forecast & validation")

if not gate_banner(result):
    st.stop()  # the gate: nothing below is computed when controls fail

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
# Opt-in behind a button, because each fit is a full NUTS run. The page must stay
# usable in seconds; nobody clicks through a demo that blocks for minutes.
st.divider()
st.subheader("Posterior-predictive interval")
st.caption(
    "A NumPyro **local linear trend** with monthly seasonality, fitted on the log scale "
    "with NUTS. Every posterior draw continues the random walk from *its own* level, "
    "trend and variances, so the fan carries parameter uncertainty rather than being one "
    "fitted path with an error band attached."
)

leaves = leaf_series(result.ledger)
choice = st.selectbox(
    "Cost center",
    options=list(leaves.columns),
    format_func=lambda c: f"{c[0]} / {c[1]}",
    key="bayes_center",
)

if st.button("Fit posterior (NUTS, ~15s)", key="fit_bayes"):
    try:
        from fpa.forecast.bayes import forecast_intervals
    except ImportError:
        st.error("NumPyro/JAX not installed — `pip install -r requirements-bayes.txt`.")
    else:
        history = leaves[choice].dropna()
        with st.spinner("Running NUTS…"):
            st.session_state["intervals"] = (
                choice,
                history,
                forecast_intervals(history, result.settings.horizon_months),
            )

stored = st.session_state.get("intervals")
if stored and stored[0] == choice:
    _, history, bands = stored

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
    columns = st.columns(3)
    columns[0].metric("Interval width, month 1", f"${width.iloc[0] / 1e6:,.0f}M")
    columns[1].metric(
        "Month 12", f"${width.iloc[-1] / 1e6:,.0f}M",
        f"{width.iloc[-1] / width.iloc[0]:.1f}x wider", delta_color="off",
    )
    columns[2].metric("Divergent transitions", f"{bands.attrs.get('divergences', 0)}")
    st.caption(
        "The band **must** widen with horizon — a fitted straight line would grow like "
        "√h and understate long-horizon risk, which is the reason for an integrated "
        "random walk. Divergent transitions are the sampler reporting where it could not "
        "explore the posterior; `target_accept_prob` is set to 0.99 to keep them near zero, "
        "because a divergent fit is a biased fit rather than a rough one."
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
