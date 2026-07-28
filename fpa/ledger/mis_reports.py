"""MIS Builder report definitions, committed as data rather than clicked together.

This is the EPM reporting layer. `mis_builder` (OCA) is what turns Odoo from an ERP
into something that can render a KPI-driven income statement and balance sheet with
budget and variance columns — the half stock Odoo does not have.

**Why the definitions live here.** A report assembled through the web UI exists only
in one database. It cannot be reviewed, diffed, or rebuilt, and when someone asks
"why does gross margin say that", the answer is a screenshot. Defining the reports
in version control makes the KPI expressions reviewable artifacts, the same argument
this project makes for keeping the SQL in ``sql/`` as files.

**The expression language** is `mis_builder`'s Account Expression Parser:

    balp[500000]   variation of the balance over the period  (p = period)
    bale[1%]       balance at end of period, accounts starting with 1
    bali[...]      balance at start of period
    balu[...]      unallocated P&L — start of fiscal year to start of period

Sign convention follows the ledger: revenue is a credit, so ``balp[400000]`` is
negative and is negated to present a positive top line. Expenses are debits and
present as-is. Getting this backwards produces a plausible income statement with the
margin inverted, which is why every P&L KPI here carries an explicit sign.

Verified against ``docker/addons/mis-builder/mis_builder/models/`` at branch 18.0
rather than assumed: ``mis.report.kpi`` takes ``name``, ``description``,
``expression``, ``type`` (``num``/``pct``/``str``), ``compare_method``
(``diff``/``pct``/``none``), ``accumulation_method`` (``sum``/``avg``/``none``) and
``sequence``; ``mis.report.instance.period`` takes ``mode`` (``fix``/``relative``/
``none``), ``type`` (``d``/``w``/``m``/``y``/``date_range``), ``offset``,
``duration`` and ``source``.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

from fpa.config import (
    BALANCE_SHEET_ASSETS,
    BALANCE_SHEET_EQUITY,
    BALANCE_SHEET_LIABILITIES,
    Settings,
    get_settings,
)

logger = logging.getLogger(__name__)


class Kpi(NamedTuple):
    """One line of a MIS report."""

    name: str  # technical name; usable inside other expressions
    description: str  # the caption a reader sees
    expression: str
    kpi_type: str = "num"
    compare_method: str = "diff"
    accumulation: str = "sum"


# ---------------------------------------------------------------------------
# Income statement
# ---------------------------------------------------------------------------
# Subtotals reference earlier KPIs by name rather than repeating their account
# lists. That is not brevity for its own sake: a chart of accounts that grows a
# line has exactly one place to change, so a subtotal cannot silently stop
# including something.
PROFIT_AND_LOSS: tuple[Kpi, ...] = (
    Kpi("revenue", "Revenue", "-balp[400000]"),
    Kpi("cost_of_revenue", "Cost of revenue", "balp[500000]"),
    Kpi("gross_profit", "Gross profit", "revenue - cost_of_revenue"),
    Kpi(
        "gross_margin",
        "Gross margin %",
        "gross_profit / revenue",
        kpi_type="pct",
        compare_method="diff",
        # A margin is a ratio, so summing it across months is meaningless. Average
        # it pro-rata instead — the difference shows up the moment anyone looks at
        # a quarter column.
        accumulation="avg",
    ),
    Kpi("research_development", "Technology and development", "balp[600000]"),
    Kpi("marketing", "Marketing", "balp[610000]"),
    Kpi("general_administrative", "General and administrative", "balp[620000]"),
    Kpi(
        "total_opex",
        "Total operating expense",
        "research_development + marketing + general_administrative",
    ),
    Kpi("operating_income", "Operating income", "gross_profit - total_opex"),
    Kpi(
        "operating_margin",
        "Operating margin %",
        "operating_income / revenue",
        kpi_type="pct",
        accumulation="avg",
    ),
)

# ---------------------------------------------------------------------------
# Balance sheet
# ---------------------------------------------------------------------------
# Built from the same ``fpa.config`` tables the ERP loader posts from, so the
# report cannot drift from the chart of accounts. A hand-maintained copy of the
# account list here would be a second source of truth, and second sources of truth
# are how a report quietly stops footing.
def _balance_sheet_kpis() -> tuple[Kpi, ...]:
    kpis: list[Kpi] = []
    sequence_groups = [
        ("asset", BALANCE_SHEET_ASSETS, 1),
        ("liability", BALANCE_SHEET_LIABILITIES, -1),
        ("equity", BALANCE_SHEET_EQUITY, -1),
    ]
    subtotals: dict[str, list[str]] = {"asset": [], "liability": [], "equity": []}

    for group, lines, sign in sequence_groups:
        for line in lines:
            # ``bale`` — balance at end of period. A balance sheet is a position at
            # a date, not a movement over one; ``balp`` would report the quarter's
            # change and look like a very small balance sheet.
            prefix = "-" if sign < 0 else ""
            kpis.append(
                Kpi(
                    line.account,
                    line.name,
                    f"{prefix}bale[{line.code}]",
                    # A balance does not sum across periods — the closing balance
                    # is the balance, not the total of twelve of them.
                    accumulation="none",
                )
            )
            subtotals[group].append(line.account)

    kpis.append(Kpi("total_assets", "Total assets", " + ".join(subtotals["asset"]), accumulation="none"))
    kpis.append(
        Kpi(
            "total_liabilities",
            "Total liabilities",
            " + ".join(subtotals["liability"]),
            accumulation="none",
        )
    )
    kpis.append(
        Kpi("total_equity", "Total equity", " + ".join(subtotals["equity"]), accumulation="none")
    )
    kpis.append(
        Kpi(
            "balance_check",
            "Check: assets − liabilities − equity",
            "total_assets - total_liabilities - total_equity",
            accumulation="none",
        )
    )
    return tuple(kpis)


BALANCE_SHEET = _balance_sheet_kpis()


class ReportSpec(NamedTuple):
    """A MIS report and the period columns its default instance renders."""

    name: str
    description: str
    kpis: tuple[Kpi, ...]
    instance: str
    periods: tuple[dict, ...]


# Period columns. ``mode="relative"`` with ``offset`` counts back from the report's
# base date, so these stay correct as the data moves forward — a fixed-date column
# would be stale the moment a new quarter lands.
_QUARTERLY_COLUMNS = (
    {"name": "Current quarter", "type": "m", "offset": -3, "duration": 3, "sequence": 10},
    {"name": "Prior quarter", "type": "m", "offset": -6, "duration": 3, "sequence": 20},
    {"name": "Year to date", "type": "m", "offset": -12, "duration": 12, "sequence": 30},
)

REPORTS: tuple[ReportSpec, ...] = (
    ReportSpec(
        name="Income Statement — FP&A",
        description="Revenue through operating margin, from the cost-center allocation journal",
        kpis=PROFIT_AND_LOSS,
        instance="Income Statement — rolling quarters",
        periods=_QUARTERLY_COLUMNS,
    ),
    ReportSpec(
        name="Balance Sheet — Filed",
        description="Filed balance-sheet positions, with the articulation check as a KPI",
        kpis=BALANCE_SHEET,
        instance="Balance Sheet — current vs prior quarter",
        periods=(
            {"name": "Current quarter", "type": "m", "offset": -3, "duration": 3, "sequence": 10},
            {"name": "Prior quarter", "type": "m", "offset": -6, "duration": 3, "sequence": 20},
        ),
    ),
)


def seed_reports(client, company_id: int, specs: tuple[ReportSpec, ...] = REPORTS) -> int:
    """Create or refresh the MIS reports and their default instances.

    Idempotent by rewrite: the KPIs of an existing report are deleted and recreated
    rather than patched. A report definition is small and wholly derived from this
    file, so rewriting it is both simpler and safer than reconciling field by field
    — and it means editing a KPI here actually propagates, which a create-if-absent
    seeder would not.
    """
    installed = client.execute(
        "ir.module.module",
        "search",
        [("name", "=", "mis_builder"), ("state", "=", "installed")],
        limit=1,
    )
    if not installed:
        logger.warning("mis_builder not installed — skipping report definitions")
        return 0

    created = 0
    for spec in specs:
        report_id = client.find_or_create(
            "mis.report",
            [("name", "=", spec.name)],
            {"name": spec.name, "description": spec.description},
        )

        existing = client.execute("mis.report.kpi", "search", [("report_id", "=", report_id)])
        if existing:
            client.execute("mis.report.kpi", "unlink", existing)

        for sequence, kpi in enumerate(spec.kpis, start=1):
            client.execute(
                "mis.report.kpi",
                "create",
                {
                    "report_id": report_id,
                    "name": kpi.name,
                    "description": kpi.description,
                    "expression": kpi.expression,
                    "type": kpi.kpi_type,
                    "compare_method": kpi.compare_method,
                    "accumulation_method": kpi.accumulation,
                    "sequence": sequence * 10,
                },
            )
        created += len(spec.kpis)

        instance_id = client.find_or_create(
            "mis.report.instance",
            [("name", "=", spec.instance)],
            {
                "name": spec.instance,
                "report_id": report_id,
                "company_id": company_id,
                # Posted entries only. Including drafts would report numbers that
                # are not in the books.
                "target_move": "posted",
            },
        )

        periods = client.execute(
            "mis.report.instance.period", "search", [("report_instance_id", "=", instance_id)]
        )
        if periods:
            client.execute("mis.report.instance.period", "unlink", periods)

        for period in spec.periods:
            client.execute(
                "mis.report.instance.period",
                "create",
                {
                    "report_instance_id": instance_id,
                    "name": period["name"],
                    "mode": "relative",
                    "type": period["type"],
                    "offset": period["offset"],
                    "duration": period["duration"],
                    "sequence": period["sequence"],
                    "source": "actuals",
                },
            )

        logger.info(
            "report %r: %d KPIs, %d period columns", spec.name, len(spec.kpis), len(spec.periods)
        )

    return created


def main(settings: Settings | None = None) -> int:
    """Load the report definitions into a running Odoo."""
    from fpa.ledger.odoo_load import OdooClient, ensure_user_groups

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = settings or get_settings()

    client = OdooClient(
        settings.odoo_url, settings.odoo_db, settings.odoo_user, settings.odoo_password
    ).connect()
    ensure_user_groups(client)
    company_id = client.execute("res.company", "search", [], limit=1)[0]

    count = seed_reports(client, company_id)
    print(f"Loaded {count} KPIs across {len(REPORTS)} MIS reports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
