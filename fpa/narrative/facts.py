"""Compile the facts payload — the only thing the model is allowed to see.

The model receives this dictionary and nothing else. Every number in it was computed
in Python by the pipeline, and ``fpa.narrative.groundedness`` rejects any draft citing
a figure that is not here.

Deliberately excludes a generation timestamp. It would be a number in the payload that
the model could legitimately cite, it changes every run so drafts stop being
reproducible, and it belongs in the audit log — which is where it is recorded.
"""

from __future__ import annotations

import json

import pandas as pd


def build_facts_payload(
    *,
    ticker: str,
    variance_summary: dict,
    revenue_drivers: dict,
    control_pass_rate: float,
    controls_failed: int,
    blocking_failed: int,
    backtest_mase: float | None = None,
    backtest_model: str | None = None,
    series_losing_to_naive: int | None = None,
) -> dict:
    """Assemble the JSON-serializable facts the commentary may describe."""
    payload: dict = {
        "entity": {"ticker": ticker},
        "variance": variance_summary,
        "revenue": revenue_drivers,
        "pipeline_health": {
            "control_pass_rate": round(float(control_pass_rate), 4),
            "controls_failed": int(controls_failed),
            "blocking_failures": int(blocking_failed),
        },
    }

    if backtest_mase is not None:
        payload["forecast_quality"] = {
            "model": backtest_model,
            "mase_filed_quarterly": round(float(backtest_mase), 4),
            "series_losing_to_naive": series_losing_to_naive,
            "basis": (
                "MASE is scaled by the in-sample seasonal-naive error: below 1.0 beats "
                "the benchmark, above 1.0 loses to it."
            ),
        }
    return payload


def from_pipeline(result, variance_report: dict) -> dict:
    """Build the payload straight from a :class:`fpa.pipeline.PipelineResult`."""
    backtest_mase = backtest_model = losing = None
    if result.backtest_filed is not None:
        summary = result.backtest_filed.summary().iloc[0]
        backtest_model = str(summary["model"])
        backtest_mase = float(summary["mase"])
        losing = int(len(result.backtest_filed.losses_to_benchmark()))

    return build_facts_payload(
        ticker=result.settings.ticker,
        variance_summary=variance_report.get("summary", {}),
        revenue_drivers=variance_report.get("revenue_drivers", {}),
        control_pass_rate=result.controls.pass_rate,
        controls_failed=sum(not r.passed for r in result.controls.results),
        blocking_failed=len(result.controls.blocking_failures),
        backtest_mase=backtest_mase,
        backtest_model=backtest_model,
        series_losing_to_naive=losing,
    )


def to_json(facts: dict, *, indent: int = 2) -> str:
    """Serialize the payload, coercing pandas/NumPy scalars to plain Python."""

    def default(obj):
        if isinstance(obj, (pd.Timestamp,)):
            return str(obj.date())
        if hasattr(obj, "item"):
            return obj.item()
        return str(obj)

    return json.dumps(facts, indent=indent, default=default)
