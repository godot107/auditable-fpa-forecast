"""KPI layer.

Organized by provenance, because that is what determines how much weight a number
can carry:

* :mod:`fpa.kpi.pnl` — **Layer 1 & balance sheet.** Computed directly from filed
  facts. Every figure traces to an accession number.
* :mod:`fpa.kpi.process` — **Layer 4.** Process and trust metrics about the
  pipeline itself: control pass rate, forecast accuracy by vintage, share of
  AI-drafted commentary approved unedited.
* :mod:`fpa.kpi.finops` — **Layer 3, scaffolded only.** Technology unit economics.
  The hooks and cost-center structure exist; the metrics are not built out.
"""

from fpa.kpi.pnl import pnl_kpis, balance_sheet_kpis
from fpa.kpi.process import process_kpis

__all__ = ["pnl_kpis", "balance_sheet_kpis", "process_kpis"]
