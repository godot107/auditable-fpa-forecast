"""Ingest layer — filed actuals from SEC EDGAR XBRL."""

from fpa.ingest.edgar import load_facts, quarterly_actuals

__all__ = ["load_facts", "quarterly_actuals"]
