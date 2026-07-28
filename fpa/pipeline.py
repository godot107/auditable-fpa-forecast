"""The pipeline — the only place the stages are wired together.

Everything else (the CLI, the Streamlit app, the tests) calls into here, so there
is exactly one definition of what a run does and one place the control gate is
enforced. Same convention as ``financial-forecasting-engine/fce/pipeline.py``.

Order matters and is not negotiable: **controls run before anything is forecast or
published.** If a blocking control fails, the run stops with the report attached
and no forecast is produced — a number that failed its integrity checks should not
exist downstream, not even unlabeled.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from fpa.config import EXPENSE_ACCOUNTS, Settings, get_settings
from fpa.controls import ControlReport, LedgerContext, run_controls
from fpa.forecast.backtest import BacktestResult, honest_validation_report, run_backtest
from fpa.forecast.models import forecast_hierarchy, leaf_series
from fpa.ingest.edgar import accession_index, load_facts, quarterly_actuals
from fpa.ingest.statements import balance_sheet, cash_flow, income_statement
from fpa.ledger.budget import build_budget, build_revenue_budget
from fpa.ledger.disaggregate import monthly_drivers, monthly_ledger, monthly_revenue

logger = logging.getLogger(__name__)


class ControlGateError(RuntimeError):
    """Raised when a caller demands published output from a run that failed its controls."""


@dataclass
class PipelineResult:
    """Everything one run produced, plus the control report that gates it."""

    settings: Settings
    quarterly: pd.DataFrame
    ledger: pd.DataFrame
    revenue: pd.DataFrame
    drivers: pd.DataFrame
    budget: pd.DataFrame
    revenue_budget: pd.DataFrame
    facts: pd.DataFrame
    accessions: pd.DataFrame
    # The three statements. Built before the gate, because a control that stops the
    # run still has to say *which* statement failed to articulate.
    income_statement: pd.DataFrame
    balance_sheet: pd.DataFrame
    cash_flow: pd.DataFrame
    segments: pd.DataFrame | None
    controls: ControlReport
    forecast: pd.DataFrame | None = None
    backtest_monthly: BacktestResult | None = None
    backtest_filed: BacktestResult | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def gate_passed(self) -> bool:
        return self.controls.passed

    def require_gate(self) -> None:
        """Raise unless every blocking control passed."""
        if not self.gate_passed:
            names = ", ".join(r.name for r in self.controls.blocking_failures)
            raise ControlGateError(
                f"blocking control(s) failed: {names}. No forecast or commentary is published."
            )

    def validation_markdown(self) -> str:
        if self.backtest_monthly is None or self.backtest_filed is None:
            return "_Forecast not produced: control gate failed._"
        return honest_validation_report(
            self.backtest_monthly,
            self.backtest_filed,
            horizon_months=self.settings.horizon_months,
            horizon_quarters=self.metadata.get("horizon_quarters", 4),
        )


def build_ledger(settings: Settings, *, refresh: bool = False) -> LedgerContext:
    """Ingest filed actuals and build the modeled monthly ledger, budget and drivers."""
    quarterly = quarterly_actuals(settings, refresh=refresh)
    ledger = monthly_ledger(settings, quarterly)
    revenue = monthly_revenue(settings, quarterly)
    drivers = monthly_drivers(settings, quarterly)
    # The other two statements. The cost-center ledger is built from the income
    # statement alone, but the balance sheet and cash flow are what the ERP posts
    # and what the articulation controls test — so they are built on every run.
    balance = balance_sheet(quarterly)
    cash = cash_flow(quarterly)

    # The ERP extract is optional by design: Odoo is the system of record, not a
    # runtime dependency. If a snapshot was materialized, the reconciliation
    # control checks it; if not, that control skips and the pipeline runs offline.
    # Regional revenue, if the Financial Statement Data Set snapshot exists. A
    # separate ingest against a separate SEC product, and ~85 MB per archive, so it
    # is never fetched implicitly — the pipeline runs without it and the
    # reconciliation control skips.
    segments = None
    if settings.vintage_path(f"segment_revenue_{settings.ticker.lower()}").exists():
        try:
            from fpa.ingest.segments import regional_revenue

            segments = regional_revenue(settings)
            logger.info("loaded regional revenue (%d fiscal years)", len(segments))
        except Exception as exc:
            logger.warning("segment data unreadable, continuing without it: %s", exc)

    erp: dict[str, pd.DataFrame | None] = {
        "erp_extract": None,
        "erp_balance_sheet": None,
        "erp_trial_balance": None,
    }
    snapshots = {
        "erp_extract": ("odoo_monthly_actuals", "extract_monthly_actuals"),
        "erp_balance_sheet": ("odoo_balance_sheet", "extract_balance_sheet"),
        "erp_trial_balance": ("odoo_trial_balance", "extract_trial_balance"),
    }
    for field_name, (snapshot, loader) in snapshots.items():
        if not settings.vintage_path(snapshot).exists():
            continue
        try:
            import fpa.extract.odoo_sql as odoo_sql

            erp[field_name] = getattr(odoo_sql, loader)(settings)
            logger.info("loaded %s (%d rows)", snapshot, len(erp[field_name]))
        except Exception as exc:  # a bad snapshot must not take down the run
            logger.warning("%s unreadable, continuing without it: %s", snapshot, exc)

    return LedgerContext(
        quarterly=quarterly,
        ledger=ledger,
        revenue=revenue,
        drivers=drivers,
        budget=build_budget(settings, ledger),
        facts=load_facts(settings, refresh=False),
        balance_sheet=balance,
        cash_flow=cash,
        segments=segments,
        **erp,
    )


def backtest_frames(context: LedgerContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the two series frames the two backtests run on.

    The first is the modeled monthly ledger; the second is filed quarterly data
    only. They are scored separately and reported together because the monthly one
    flatters the model — see :func:`honest_validation_report`.
    """
    leaves = leaf_series(context.ledger)
    monthly = leaves.copy()
    monthly[("Total", "Opex")] = leaves.sum(axis=1)
    monthly[("Total", "Revenue")] = context.revenue.set_index("period")["amount"]

    filed_columns = ["revenue", *EXPENSE_ACCOUNTS, "operating_income"]
    filed = context.quarterly[[c for c in filed_columns if c in context.quarterly.columns]].dropna()
    return monthly, filed


def run(
    settings: Settings | None = None,
    *,
    refresh: bool = False,
    horizon_quarters: int = 4,
) -> PipelineResult:
    """Run the full pipeline. Forecasting happens only if the control gate passes."""
    settings = settings or get_settings()

    logger.info("ingesting filed actuals for %s", settings.ticker)
    context = build_ledger(settings, refresh=refresh)

    logger.info("running controls")
    controls = run_controls(context)

    result = PipelineResult(
        settings=settings,
        quarterly=context.quarterly,
        ledger=context.ledger,
        revenue=context.revenue,
        drivers=context.drivers,
        budget=context.budget,
        revenue_budget=build_revenue_budget(settings, context.revenue),
        facts=context.facts,
        accessions=accession_index(settings),
        income_statement=income_statement(context.quarterly),
        balance_sheet=context.balance_sheet,
        cash_flow=context.cash_flow,
        segments=context.segments,
        controls=controls,
        metadata={
            "horizon_quarters": horizon_quarters,
            "q4_provenance": context.quarterly.attrs.get("q4_provenance", {}),
        },
    )

    if not controls.passed:
        # The gate. Nothing downstream is computed, so nothing downstream can be
        # displayed, exported or narrated.
        logger.error(
            "control gate FAILED (%s) — no forecast produced",
            ", ".join(r.name for r in controls.blocking_failures),
        )
        return result

    logger.info("controls passed (%.0f%%) — forecasting", controls.pass_rate * 100)
    monthly_frame, filed_frame = backtest_frames(context)

    result.forecast = forecast_hierarchy(
        context.ledger, settings.horizon_months, model="ets"
    )
    result.backtest_monthly = run_backtest(
        monthly_frame, horizon=settings.horizon_months, folds=settings.backtest_folds
    )
    result.backtest_filed = run_backtest(
        filed_frame, horizon=horizon_quarters, folds=4, period=4
    )
    return result
