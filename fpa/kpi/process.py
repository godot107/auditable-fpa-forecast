"""Layer 4 — process and trust KPIs: metrics about the pipeline, not the business.

These are the numbers a finance leader needs before trusting the other numbers,
and they are close to free because the pipeline already emits the underlying data.

``commentary_approval_rate`` is the one worth putting on screen: it turns
"human-in-the-loop" from a claim into a measurement. If reviewers edit most
AI-drafted commentary, the drafting is not working, and that should be visible
rather than asserted.
"""

from __future__ import annotations

import pandas as pd

from fpa.controls import ControlReport, Severity


def process_kpis(
    controls: ControlReport,
    *,
    backtest_filed=None,
    audit_log: pd.DataFrame | None = None,
    variance: pd.DataFrame | None = None,
) -> dict:
    """Return the process/trust KPI block for one run."""
    kpis: dict[str, object] = {
        "control_pass_rate": controls.pass_rate,
        "controls_total": len(controls.results),
        "controls_failed": sum(not r.passed for r in controls.results),
        "controls_blocking_failed": len(controls.blocking_failures),
        "gate_passed": controls.passed,
    }

    if backtest_filed is not None:
        summary = backtest_filed.summary()
        best = summary.iloc[0]
        kpis["best_model"] = best["model"]
        kpis["best_model_mase"] = float(best["mase"])
        # How much the model actually buys over doing nothing clever.
        kpis["lift_over_naive"] = float(
            1.0 - best["mase"] / summary.set_index("model")["mase"].get("seasonal_naive", float("nan"))
        )
        kpis["series_losing_to_naive"] = int(len(backtest_filed.losses_to_benchmark()))

    if variance is not None and not variance.empty and "explanation" in variance.columns:
        kpis["variance_lines_explained"] = float(variance["explanation"].notna().mean())

    if audit_log is not None and not audit_log.empty:
        decisions = audit_log[audit_log["action"].isin(["approved", "rejected", "edited"])]
        if not decisions.empty:
            kpis["commentary_reviewed"] = int(len(decisions))
            kpis["commentary_approval_rate"] = float(
                (decisions["action"] == "approved").mean()
            )
            kpis["commentary_edit_rate"] = float((decisions["action"] == "edited").mean())

    return kpis


def control_detail(controls: ControlReport) -> pd.DataFrame:
    """Per-control table for display, ordered so failures surface first."""
    frame = controls.to_frame()
    severity_order = {Severity.BLOCKING.value: 0, Severity.WARN.value: 1, Severity.INFO.value: 2}
    frame["_status"] = (frame["status"] == "PASS").astype(int)
    frame["_sev"] = frame["severity"].map(severity_order)
    return (
        frame.sort_values(["_status", "_sev"])
        .drop(columns=["_status", "_sev"])
        .reset_index(drop=True)
    )
