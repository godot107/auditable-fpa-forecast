"""Shared helpers for the Streamlit pages: cached pipeline run, theme, provenance badges."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fpa.config import get_settings  # noqa: E402
from fpa.pipeline import run  # noqa: E402

# Provenance badges. Every figure on screen carries one, so nothing is ambiguous
# about whether it was filed, modeled, or predicted.
BADGE = {
    "REAL": ("#2e7d32", "REAL — filed with the SEC"),
    "MODELED": ("#ef6c00", "MODELED — allocated by this project"),
    "IMPLIED": ("#6a1b9a", "IMPLIED — forced by an identity"),
    "FORECAST": ("#1565c0", "FORECAST — model output"),
}

CSS = """
<style>
  .metric-card { background: rgba(255,255,255,0.04); border-radius: 10px;
                 padding: 16px 18px; margin-bottom: 14px;
                 border-left: 3px solid rgba(255,255,255,0.15); }
  .metric-card h4 { margin: 0 0 6px 0; font-size: 0.78rem; letter-spacing: .04em;
                    text-transform: uppercase; opacity: .72; font-weight: 600; }
  .metric-card h2 { margin: 0; font-size: 1.7rem; font-weight: 650; }
  .badge { display:inline-block; padding: 1px 7px; border-radius: 5px;
           font-size: 0.63rem; font-weight: 700; letter-spacing: .05em;
           color: #fff; vertical-align: middle; }
  .accn { font-family: ui-monospace, monospace; font-size: 0.72rem; opacity: .6; }
</style>
"""


def setup(title: str) -> None:
    st.set_page_config(page_title=f"{title} — FP&A Demo", layout="wide", page_icon="📐")
    st.markdown(CSS, unsafe_allow_html=True)


def badge(kind: str) -> str:
    color, tooltip = BADGE[kind]
    return f'<span class="badge" style="background:{color}" title="{tooltip}">{kind}</span>'


def metric_card(label: str, value: str, kind: str, note: str = "") -> str:
    extra = f'<div class="accn">{note}</div>' if note else ""
    return (
        f'<div class="metric-card"><h4>{label} {badge(kind)}</h4>'
        f"<h2>{value}</h2>{extra}</div>"
    )


@st.cache_resource(show_spinner="Running pipeline — ingest, controls, forecast…")
def pipeline():
    """One cached pipeline run shared by every page.

    ``cache_resource`` rather than ``cache_data`` because the result holds
    non-serializable objects (the control report, backtest results).
    """
    return run(get_settings())


def gate_banner(result) -> bool:
    """Render the control-gate state. Returns True when downstream output may show."""
    if result.gate_passed:
        st.success(
            f"Control gate **passed** — {sum(r.passed for r in result.controls.results)}"
            f"/{len(result.controls.results)} controls, "
            f"{len(result.controls.blocking_failures)} blocking failures."
        )
        return True

    names = ", ".join(f"`{r.name}`" for r in result.controls.blocking_failures)
    st.error(
        f"**Control gate FAILED** — {names}. No forecast, variance or commentary is "
        "produced for this run. This is deliberate: a figure that failed its "
        "integrity checks should not exist downstream, not even labelled."
    )
    return False


def money(value: float, unit: str = "M") -> str:
    divisor = {"M": 1e6, "B": 1e9}[unit]
    return f"${value / divisor:,.0f}{unit}"
