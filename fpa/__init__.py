"""Auditable FP&A rolling-forecast pipeline.

Actuals come from SEC EDGAR XBRL, so every reported figure traces to the accession
number of the filing it came from. Cost-center detail below the filed line items is
modeled, and asserted to foot back to the filed total. An LLM writes the variance
commentary but never produces a number.
"""

__version__ = "0.1.0"
