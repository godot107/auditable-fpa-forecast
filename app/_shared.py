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
        # Verified, not merely passed. A skipped control returns True so the pipeline
        # can run without a live ERP, so counting passes would report a run with no
        # data behind four checks identically to a fully exercised one.
        skipped = len(result.controls.results) - len(result.controls.verified)
        note = f", {skipped} skipped — a skip is not a pass" if skipped else ""
        st.success(
            f"Control gate **passed** — {len(result.controls.verified)}"
            f"/{len(result.controls.results)} controls verified, "
            f"{len(result.controls.blocking_failures)} blocking failures{note}."
        )
        return True

    names = ", ".join(f"`{r.name}`" for r in result.controls.blocking_failures)
    st.error(
        f"**Control gate FAILED** — {names}. No forecast, variance or commentary is "
        "produced for this run. This is deliberate: a figure that failed its "
        "integrity checks should not exist downstream, not even labelled."
    )
    return False


def cost_center_disclaimer(where: str = "") -> None:
    """State plainly that the cost-center hierarchy is invented.

    Shared rather than written per page, because a disclaimer that drifts between
    surfaces is worse than one that is missing: the reader learns the caveat once and
    then assumes it applies everywhere it does not appear.

    The distinction this has to carry: the *amounts* are filed and foot to the filing
    at machine precision, while the *attribution* is fabricated. "Modeled" on its own
    reads as calibrated-to-something. No filer publishes spend by department, so the
    nine cost centers below are a plausible org chart and nothing more.
    """
    st.warning(
        f"**The cost centers are invented.**{' ' + where if where else ''} No SEC filer "
        "discloses spend by department, so the nine cost centers here — and the four "
        "functions above them — are a plausible streaming-company org chart written by "
        "this project, not anything Netflix reports.\n\n"
        "Two layers are modeled: the split of each filed expense line across cost "
        "centers, and the phasing of each filed quarter across its three months. **A "
        "single cost-center line in a single month is therefore invented twice over.**\n\n"
        "What is real is the constraint: the modeled detail sums to the filed quarterly "
        "total at 1e-12 relative, re-proved by the `ledger_foots_to_filed` control on "
        "every run. Real at the total, invented at the line — the reverse of what a "
        "general ledger usually implies."
    )


def money(value: float, unit: str = "M") -> str:
    divisor = {"M": 1e6, "B": 1e9}[unit]
    return f"${value / divisor:,.0f}{unit}"
