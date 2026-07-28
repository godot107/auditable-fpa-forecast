"""Layer 3 — technology unit economics (FinOps). **Scaffolded, not built out.**

The cost-center hierarchy in ``fpa.config`` already carries the sub-towers these
metrics attach to (Cloud Infrastructure, CDN & Delivery, Platform Engineering), and
the ledger resolves spend to them. What is missing is the *consumption* denominator
— streaming hours, GB egress, compute hours — which no public filing discloses and
which this project deliberately does not invent.

That is the honest constraint worth stating out loud: a FinOps unit-cost metric is
only as good as its usage telemetry, and a demo built on public filings has none.
In a real engagement this denominator comes from the cloud provider's billing
export (or Apptio's own consumption feed), joined on cost centre and period.

Each function below documents the metric and raises rather than returning a
plausible-looking number.
"""

from __future__ import annotations

import pandas as pd

# Cost-center sub-towers these metrics are defined over. Present in the ledger
# today; the usage denominators are not.
TECH_SUB_CENTERS = ("Cloud Infrastructure", "CDN & Delivery", "Platform Engineering")


class UsageDataUnavailable(NotImplementedError):
    """Raised when a unit-economics metric is requested without consumption data."""


def tech_spend_by_center(ledger: pd.DataFrame) -> pd.DataFrame:
    """Technology spend by sub-center and month. **This part works today.**

    The numerator of every unit-cost metric below, and useful on its own as
    showback: which tower is consuming the technology budget.
    """
    tech = ledger[ledger["sub_center"].isin(TECH_SUB_CENTERS)]
    return (
        tech.groupby(["period", "sub_center"], as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "spend"})
    )


def allocation_coverage(ledger: pd.DataFrame) -> float:
    """Share of technology spend attributed to a named cost center.

    The classic ITFM metric: what fraction of the technology bill has an owner,
    versus sitting in an unallocated shared pool. Trivially 1.0 here because the
    disaggregation model assigns every dollar by construction — which is exactly
    why it is *not* a meaningful number in this demo, and is documented as such
    rather than displayed as a perfect score.
    """
    return float(ledger["amount"].notna().mean())


def cost_per_streaming_hour(ledger: pd.DataFrame, usage: pd.DataFrame | None = None):
    """Delivery cost per streaming hour — the FinOps north-star unit metric.

    Requires an hours-streamed denominator per period. Netflix does not file one.
    """
    raise UsageDataUnavailable(
        "cost_per_streaming_hour needs a streaming-hours denominator, which is not "
        "in any public filing. Wire a usage feed (cloud billing export or Apptio "
        "consumption data) keyed on period and cost center."
    )


def cost_per_member(ledger: pd.DataFrame, drivers: pd.DataFrame | None = None):
    """Technology cost per member.

    The denominator (members) is itself modeled in ``fpa.ledger.disaggregate``, so
    this ratio would divide a modeled numerator by a modeled denominator and
    present the result as a unit economic. Deliberately not built: it would look
    like a KPI and mean nothing.
    """
    raise UsageDataUnavailable(
        "cost_per_member would divide modeled spend by modeled members. Both sides "
        "would be synthetic; the ratio would carry no information."
    )


def commitment_coverage(*_args, **_kwargs):
    """Share of compute covered by reserved instances / savings plans."""
    raise UsageDataUnavailable(
        "commitment_coverage needs cloud commitment and on-demand usage records."
    )
