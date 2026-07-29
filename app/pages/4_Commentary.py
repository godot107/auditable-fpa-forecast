"""Commentary page — AI-drafted variance narrative under a groundedness gate.

The rule this page demonstrates: **the model writes the commentary, never the number.**
Figures are computed in Python and handed over as a facts payload; every numeral the
model returns is checked back against it. A draft citing an ungrounded figure never
reaches the reviewer.
"""

from __future__ import annotations

import json
import shutil

import streamlit as st

from _shared import gate_banner, pipeline, setup
from fpa.audit import log_decision, read_log
from fpa.narrative import from_pipeline, generate_draft, get_provider, to_json
from fpa.narrative.groundedness import check
from fpa.variance import build_variance_report

setup("Commentary")

result = pipeline()
st.title("AI-drafted variance commentary")

if not gate_banner(result):
    st.stop()

st.caption(
    "The model receives only computed figures and may describe them. Every numeral it "
    "returns is checked against that payload; a draft citing anything else is rejected "
    "before a human sees it. The result is a **draft** — a reviewer approves or rejects, "
    "and the decision is written to an append-only audit log."
)

report = build_variance_report(
    result.ledger, result.budget, result.revenue, result.revenue_budget,
    result.drivers, periods=12,
)
facts = from_pipeline(result, report)

# `claudecode` shells out to the `claude` CLI, which exists on a developer machine and
# not on a hosted deploy. Offering an option that can only fail is worse than not
# offering it — and the seam is the point being demonstrated, not the binary.
_has_claude_cli = shutil.which("claude") is not None
_providers = ["fixture", "claudecode"] if _has_claude_cli else ["fixture"]

col1, col2 = st.columns([1, 2])
with col1:
    provider_name = st.selectbox(
        "Provider",
        _providers,
        help="fixture = deterministic, offline. claudecode = headless `claude -p`, no API key.",
    )
with col2:
    st.caption(
        "`anthropic` is a documented seam, deliberately unimplemented — the facts payload, "
        "schema, groundedness check and approval log are all provider-agnostic already."
    )
    if not _has_claude_cli:
        st.caption(
            ":grey[`claudecode` is hidden here because the `claude` CLI is not on this "
            "host. That the page still works is the argument for the seam: the "
            "groundedness gate is what guards the number, and it does not care which "
            "model produced the prose.]"
        )

if st.button("Generate draft", type="primary"):
    with st.spinner("Generating and checking groundedness…"):
        st.session_state["draft"] = generate_draft(
            facts, get_provider(provider_name, model=result.settings.narrative_model)
        )

draft_result = st.session_state.get("draft")

if draft_result is not None:
    if draft_result.publishable:
        st.success(f"Groundedness check **passed** — {draft_result.grounded.message}")
    else:
        st.error(
            f"Draft **rejected** — {draft_result.error or draft_result.grounded.message}. "
            "Nothing is published."
        )

    if draft_result.draft:
        draft = draft_result.draft
        st.subheader(draft.get("headline", ""))
        for item in draft.get("drivers", []):
            st.markdown(f"**{item['cost_center']}** — {item['comment']}")
        if draft.get("watch_item"):
            st.info(f"**Watch:** {draft['watch_item']}")

        st.caption(
            f"provider `{draft_result.provider}` · prompt `{draft_result.prompt_version}` · "
            f"attempt {draft_result.attempts} · "
            f"{draft_result.grounded.checked} figures verified"
        )

        # --- Human in the loop --------------------------------------------
        st.divider()
        st.subheader("Review")
        reviewer = st.text_input("Reviewer", value="analyst")
        note = st.text_input("Note (optional)")
        approve, reject = st.columns(2)

        period = facts.get("variance", {}).get("period_end", "unknown")
        if approve.button("Approve", width="stretch"):
            log_decision(
                result.settings.audit_dir, user=reviewer, action="approved",
                period=period, draft=draft, provider=draft_result.provider,
                prompt_version=draft_result.prompt_version, grounded=True, note=note or None,
            )
            st.success("Approved and written to the audit log.")
        if reject.button("Reject", width="stretch"):
            log_decision(
                result.settings.audit_dir, user=reviewer, action="rejected",
                period=period, draft=draft, provider=draft_result.provider,
                prompt_version=draft_result.prompt_version, grounded=True, note=note or None,
            )
            st.warning("Rejected and written to the audit log.")

    for rejected in draft_result.rejected_drafts:
        with st.expander("A rejected draft (kept for the audit trail)"):
            st.json(rejected)

# --- Prove the check bites ------------------------------------------------
st.divider()
st.subheader("Try to get a fabricated number past the check")
st.caption(
    "Edit the sentence below. Figures drawn from the payload pass; anything else is "
    "rejected. This is the guarantee, live."
)
probe = st.text_area(
    "Candidate commentary",
    value="Cloud Infrastructure overran plan by $604.4M, a 22.9% miss.",
    height=80,
)
if probe.strip():
    outcome = check(probe, facts)
    (st.success if outcome.passed else st.error)(outcome.message)

with st.expander("The facts payload the model is allowed to see"):
    st.code(to_json(facts), language="json")

# --- Audit log ------------------------------------------------------------
st.divider()
st.subheader("Approval log")
log = read_log(result.settings.audit_dir)
if log.empty:
    st.caption("No decisions recorded yet.")
else:
    approval_rate = (log["action"] == "approved").mean()
    st.metric("Approved without edit", f"{approval_rate:.0%}", f"{len(log)} decisions")
    st.caption(
        "Human-in-the-loop as a measured number rather than a claim. If reviewers edit "
        "most drafts, the drafting is not working — and that should be visible."
    )
    st.dataframe(
        log[["timestamp", "user", "action", "period", "provider", "grounded"]],
        width="stretch", hide_index=True,
    )
