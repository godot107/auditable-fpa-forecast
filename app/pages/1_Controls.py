"""Controls page — the integrity gate, and the audit trail behind every actual."""

from __future__ import annotations

import streamlit as st

from _shared import gate_banner, pipeline, setup
from fpa.kpi.process import control_detail

setup("Controls")

result = pipeline()

st.title("Controls")
st.caption(
    "These run inside the pipeline on every run, not only in CI. A control that "
    "only runs in a test suite protects the developer; a control that runs in the "
    "pipeline protects the number."
)

gate_banner(result)

st.subheader("Control register")
detail = control_detail(result.controls)
st.dataframe(
    detail,
    width='stretch',
    hide_index=True,
    column_config={
        "control": st.column_config.TextColumn("Control", width="medium"),
        "severity": st.column_config.TextColumn("Severity", width="small"),
        "status": st.column_config.TextColumn("Status", width="small"),
        "message": st.column_config.TextColumn("Detail", width="large"),
    },
)

st.markdown(
    "**BLOCKING** stops the pipeline — no forecast, no commentary. "
    "**WARN** is surfaced for a human but does not halt the run."
)

# --- The two expected warnings, explained ---------------------------------
warnings = result.controls.warnings
if warnings:
    st.subheader("Open warnings")
    for w in warnings:
        with st.expander(f"`{w.name}` — {w.message}"):
            if w.name == "budget_coverage":
                st.markdown(
                    "Expected and structural. The plan for fiscal year *Y* is built "
                    "from year *Y−1* actuals, so the first year in the window has no "
                    "plan to be measured against. Reported rather than papered over "
                    "with a zero budget, which would silently show a 100% favourable "
                    "variance for twelve months."
                )
            elif w.name == "restatements":
                st.markdown(
                    "These are facts whose **value changed** between filings — not "
                    "merely periods that were re-reported. Every 10-K repeats the "
                    "prior year's quarters as comparatives; counting those would flag "
                    "roughly three quarters of the fact table and train everyone to "
                    "ignore the control. We keep the most recently filed value."
                )

st.divider()

# --- Q4 provenance --------------------------------------------------------
st.subheader("Derived-quarter provenance")
st.caption(
    "Q4 is never filed as a 10-Q — companies roll it into the 10-K as a full year — "
    "so it must be derived as FY − (Q1+Q2+Q3). The pipeline refuses to derive from a "
    "fiscal year whose annual income statement does not articulate, because the whole "
    "discrepancy would land in that one quarter."
)

provenance = result.metadata.get("q4_provenance", {})
rows = [
    {
        "fiscal year": year,
        "status": info["reason"],
        "accounts derived": len(info["derived"]),
        "in window": year >= int(result.settings.window_start[:4]),
    }
    for year, info in sorted(provenance.items())
]
if rows:
    st.dataframe(rows, width='stretch', hide_index=True)
    st.markdown(
        f"This is why the series starts at **{result.settings.window_start[:4]}** rather "
        "than at the earliest available filing: FY2017–FY2019 do not articulate against "
        "this chart of accounts (off by −$27.3M, +$3.8M and −$127.9M), and FY2020 onward "
        "tie to the dollar. The boundary is re-proved every run by "
        "`annual_identity_in_window` rather than asserted in a comment."
    )

st.divider()

# --- Audit trail ----------------------------------------------------------
st.subheader("Audit trail — every filed fact and its filing")
accessions = result.accessions
col1, col2 = st.columns([1, 3])
with col1:
    account = st.selectbox("Account", sorted(accessions["account"].unique()))
with col2:
    st.caption(
        f"{len(accessions):,} facts, {accessions['accn'].nunique():,} distinct filings. "
        "`n_distinct_values` above 1 means the reported figure actually moved."
    )

st.dataframe(
    accessions[accessions["account"] == account].sort_values("end", ascending=False),
    width='stretch',
    hide_index=True,
)
