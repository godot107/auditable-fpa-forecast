"""Materialize Odoo into a pinned parquet snapshot.

The app never queries Odoo live. It reads the snapshot this module writes, which
means the demo survives the ERP being down — and, more to the point, is how
analytics against an ERP is actually built: you extract to a warehouse rather than
running planning queries against a transactional system.

Queries live in ``sql/`` as plain ``.sql`` files rather than embedded strings, so
they can be read, run and reviewed on their own.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from fpa.config import Settings
from fpa.cache import cached_parquet

logger = logging.getLogger(__name__)

SQL_DIR = Path(__file__).resolve().parent.parent.parent / "sql"


def _dsn(settings: Settings) -> str:
    """Postgres DSN for Odoo's database, from the environment."""
    host = os.getenv("ODOO_PG_HOST", "localhost")
    port = os.getenv("ODOO_PG_PORT", "5432")
    user = os.getenv("ODOO_PG_USER", "odoo")
    password = os.getenv("ODOO_PG_PASSWORD", "odoo")
    return f"postgresql://{user}:{password}@{host}:{port}/{settings.odoo_db}"


def read_sql(name: str, settings: Settings) -> pd.DataFrame:
    """Run a named query from ``sql/`` against Odoo's Postgres.

    Uses DuckDB's postgres scanner so no extra Postgres driver is needed — the
    project already depends on DuckDB for the warehouse side.
    """
    import duckdb

    query = (SQL_DIR / f"{name}.sql").read_text()
    connection = duckdb.connect()
    connection.execute("INSTALL postgres; LOAD postgres;")
    connection.execute(f"ATTACH '{_dsn(settings)}' AS odoo (TYPE postgres, READ_ONLY);")
    connection.execute("USE odoo;")
    try:
        return connection.execute(query).fetchdf()
    finally:
        connection.close()


def extract_monthly_actuals(settings: Settings, *, refresh: bool = False) -> pd.DataFrame:
    """Monthly actuals by cost center, extracted from the ERP and pinned."""
    path = settings.vintage_path("odoo_monthly_actuals")

    def fetch() -> pd.DataFrame:
        frame = read_sql("monthly_actuals_by_cost_center", settings)
        frame["period"] = pd.to_datetime(frame["period"])
        return frame

    return cached_parquet(path, fetch, refresh=refresh)


def extract_trial_balance(settings: Settings, *, refresh: bool = False) -> pd.DataFrame:
    """Trial balance by account and fiscal year."""
    return cached_parquet(
        settings.vintage_path("odoo_trial_balance"),
        lambda: read_sql("trial_balance", settings),
        refresh=refresh,
    )


def extract_balance_sheet(settings: Settings, *, refresh: bool = False) -> pd.DataFrame:
    """The balance sheet as the ERP holds it, one row per quarter end and account."""
    path = settings.vintage_path("odoo_balance_sheet")

    def fetch() -> pd.DataFrame:
        frame = read_sql("balance_sheet", settings)
        frame["period"] = pd.to_datetime(frame["period"])
        return frame

    return cached_parquet(path, fetch, refresh=refresh)


def reconcile_balance_sheet(extract: pd.DataFrame, balance_sheet: pd.DataFrame) -> pd.DataFrame:
    """Compare the ERP's balance sheet against the filed one, line by line.

    The second half of the round trip, and a harder one than the P&L: these
    balances went through a *movement* posting, so an error in any single quarter's
    entry propagates into every later balance rather than staying local. A
    cumulative sum that still lands on the filed figure at all 26 quarter ends is a
    much stronger statement than 26 independent comparisons would be.

    Odoo's stored sign is the natural ledger balance (debit − credit), so the
    credit-side lines come back negative and are flipped to presentation sign here
    using the same table the loader posted from — not a second copy of it.
    """
    from fpa.config import balance_sheet_lines

    code_to_line = {line.code: (line.account, sign) for line, sign in balance_sheet_lines()}

    work = extract.copy()
    work["account"] = work["account_code"].map(lambda c: code_to_line.get(c, (None, 0))[0])
    work["sign"] = work["account_code"].map(lambda c: code_to_line.get(c, (None, 0))[1])
    work = work[work["account"].notna()]
    # Back to presentation sign: assets stay, liabilities and equity flip.
    work["presented"] = work["balance"] * work["sign"]

    erp = work.pivot_table(index="period", columns="account", values="presented", aggfunc="sum")
    accounts = [c for c in erp.columns if c in balance_sheet.columns]
    filed = balance_sheet[accounts].reindex(erp.index)

    diff = erp[accounts] - filed
    return pd.DataFrame(
        {
            "erp_latest": erp[accounts].iloc[-1] if len(erp) else pd.Series(dtype=float),
            "filed_latest": filed.iloc[-1] if len(filed) else pd.Series(dtype=float),
            "abs_diff": diff.abs().max(),
            "rel_diff": (diff.abs() / filed.abs().replace(0, pd.NA)).max(),
        }
    )


def reconcile_to_filed(extract: pd.DataFrame, quarterly: pd.DataFrame) -> pd.DataFrame:
    """Compare the ERP extract against the filed quarterly figures, per account.

    This is the round-trip proof. The numbers went EDGAR → disaggregation → Odoo
    journal entries → SQL extract, through an ORM, a double-entry posting, an
    analytic distribution and a GROUP BY. If they still tie to the filed 10-Q,
    nothing was lost or double-counted on the way.
    """
    # Derived from the seeder's own map rather than restated here, so the two
    # cannot drift apart.
    from fpa.ledger.odoo_load import ACCOUNT_MAP

    code_to_internal = {code: internal for internal, (code, _n, _t) in ACCOUNT_MAP.items()}

    work = extract.copy()
    work["internal"] = work["account_code"].map(code_to_internal)
    work["quarter_end"] = pd.PeriodIndex(work["period"], freq="Q").to_timestamp(how="end").normalize()

    erp = work.groupby(["quarter_end", "internal"])["amount"].sum().unstack()
    accounts = [c for c in erp.columns if c in quarterly.columns]
    filed = quarterly[accounts].reindex(erp.index)

    diff = erp[accounts] - filed
    return pd.DataFrame(
        {
            "erp_total": erp[accounts].sum(),
            "filed_total": filed.sum(),
            "abs_diff": diff.abs().max(),
            "rel_diff": (diff.abs() / filed.abs()).max(),
        }
    )
