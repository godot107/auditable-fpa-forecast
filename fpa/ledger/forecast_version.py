"""Turn the Python forecast into a second budget version inside the ERP.

Odoo has no forecasting engine, and neither does ``mis_builder`` — nor should they.
What an EPM tool gives a finance team that an ERP does not is a **scenario
dimension**: the ability to put Actual, Plan and Forecast side by side against the
same chart of accounts and the same cost centers, and read the gap between them.

That dimension is the thing OneStream and Anaplan are actually bought for, and it
is reproducible here because ``crossovered.budget`` is a versioned container.
Loading the statistical forecast as ``FY2026 Forecast`` next to ``FY2026 Plan``
gives Odoo three columns it can report on:

* **Actual** — computed by Odoo itself from the analytic lines the posted journal
  entries generate. Not loaded; derived.
* **Plan** — the budget built in ``fpa.ledger.budget``.
* **Forecast** — this module.

The forecast stays authored in Python. The ERP is where it is *published*, which is
the same division of labour the rest of this project uses: planning logic in code,
system of record in the ledger.

One structural wrinkle worth naming. The forecast is produced at leaf cost-center
grain — ``(function, sub_center)`` — because that is the hierarchy the bottom-up
reconciliation is defined over. Budget lines need an **account** as well, since
``general_budget_id`` is what ties a line to the GL accounts its actuals are summed
from. Technology & Product draws on two accounts (cost of revenue and R&D), so its
forecast is split across them by recent actual mix. That split is a modeling choice
and is labelled as one; :func:`account_mix` documents the basis and the split is
exact, so the account detail always sums back to the forecast it came from.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fpa.config import FUNCTION_ACCOUNTS, Settings, get_settings

logger = logging.getLogger(__name__)

# Months of actuals used to establish how a cost center's spend divides between
# its accounts. Twelve, to span a full seasonal cycle: Technology & Product's
# split between cost of revenue and R&D moves with the production calendar, and a
# shorter window would read whatever phase the last quarter happened to be in.
MIX_WINDOW_MONTHS = 12


def account_mix(ledger: pd.DataFrame, *, window: int = MIX_WINDOW_MONTHS) -> pd.DataFrame:
    """Recent share of each cost center's spend by account.

    Returns ``function``, ``sub_center``, ``account``, ``share`` where share sums
    to 1.0 within each cost center. Most centers draw on a single account and get
    a share of 1.0; only Technology & Product genuinely splits.
    """
    cutoff = ledger["period"].max() - pd.DateOffset(months=window)
    recent = ledger[ledger["period"] > cutoff]
    if recent.empty:  # pragma: no cover - only if the ledger is shorter than the window
        recent = ledger

    totals = recent.groupby(["function", "sub_center", "account"])["amount"].sum()
    share = totals / totals.groupby(level=["function", "sub_center"]).sum()
    return share.rename("share").reset_index()


def forecast_by_account(forecast: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    """Split the leaf-level forecast across accounts, exactly.

    The split is proportional to recent actual mix, with the last account in each
    group absorbing the rounding remainder — the same ``_exact_split`` discipline
    the disaggregation uses, and for the same reason. If the parts did not sum back
    to the forecast, the ERP would hold a forecast that no longer matches the one
    Python published, and nothing would say so.
    """
    mix = account_mix(ledger)
    merged = forecast.merge(mix, on=["function", "sub_center"], how="left")

    if merged["account"].isna().any():
        orphans = sorted(
            merged.loc[merged["account"].isna(), ["function", "sub_center"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        raise ValueError(f"no account mix for cost center(s): {orphans}")

    # Exact split within each (period, cost center): scale by share, then hand the
    # residual to the last row of each group so it sums to the forecast to the cent.
    keys = ["period", "function", "sub_center"]
    out = merged.sort_values([*keys, "account"]).reset_index(drop=True)
    out["amount"] = out["forecast"] * out["share"]

    group_sum = out.groupby(keys, sort=False)["amount"].transform("sum")
    is_last = ~out.duplicated(keys, keep="last")
    out.loc[is_last, "amount"] += out.loc[is_last, "forecast"] - group_sum[is_last]

    return out[["period", "account", "function", "sub_center", "amount", "model"]]


def assert_split_ties(split: pd.DataFrame, forecast: pd.DataFrame, *, atol: float = 1e-6) -> None:
    """The account split must reproduce the leaf forecast it came from.

    Asserted rather than assumed, in the pipeline rather than in a test — the same
    rule the rest of the project runs on.
    """
    rolled = split.groupby(["period", "function", "sub_center"])["amount"].sum()
    original = forecast.set_index(["period", "function", "sub_center"])["forecast"]
    diff = (rolled - original.reindex(rolled.index)).abs().max()
    if pd.notna(diff) and diff > atol:
        raise AssertionError(
            f"account split does not tie to the forecast: worst ${diff:,.6f}"
        )


def seed_forecast_version(
    client,
    forecast: pd.DataFrame,
    ledger: pd.DataFrame,
    center_ids: dict[tuple[str, str], int],
    account_ids: dict[str, int],
    company_id: int,
) -> int:
    """Load the forecast into Odoo as ``FY<year> Forecast`` budget versions.

    Deliberately a *separate* ``crossovered.budget`` header per year rather than
    extra lines on the plan: a forecast that overwrites the plan destroys the only
    comparison anyone wanted. Plan is what was committed; forecast is what is now
    expected; the gap between them is the conversation.

    Idempotent on lines, not just the header — ``find_or_create`` returns the
    existing header on a re-run, and creating lines against it again silently
    doubles the forecast. That bug already happened once on the plan side.
    """
    from fpa.ledger.odoo_load import _sync_budget_lines, ensure_budget_positions

    installed = client.execute(
        "ir.module.module",
        "search",
        [("name", "=", "account_budget_oca"), ("state", "=", "installed")],
        limit=1,
    )
    if not installed:
        logger.warning("account_budget_oca not installed — skipping the forecast version")
        return 0

    split = forecast_by_account(forecast, ledger)
    assert_split_ties(split, forecast)

    positions = ensure_budget_positions(client, account_ids, company_id)
    model = split["model"].iloc[0] if len(split) else "unknown"

    created = 0
    for year, year_rows in split.groupby(split["period"].dt.year):
        name = f"FY{year} Forecast ({model})"
        header_id = client.find_or_create(
            "crossovered.budget",
            [("name", "=", name), ("company_id", "=", company_id)],
            {
                "name": name,
                "date_from": f"{year}-01-01",
                "date_to": f"{year}-12-31",
                "company_id": company_id,
            },
        )

        if not _sync_budget_lines(client, header_id, len(year_rows), name):
            continue

        payload = []
        for row in year_rows.itertuples():
            period = pd.Timestamp(row.period)
            payload.append(
                {
                    "crossovered_budget_id": header_id,
                    "general_budget_id": positions[row.account],
                    "analytic_account_id": center_ids[(row.function, row.sub_center)],
                    "date_from": period.replace(day=1).date().isoformat(),
                    "date_to": period.date().isoformat(),
                    # Negative, matching the plan: Odoo's analytic lines carry
                    # costs as negative amounts, so a forecast has to use the same
                    # sign or the variance column compares two different things.
                    "planned_amount": -round(float(row.amount), 2),
                }
            )

        if payload:
            client.execute("crossovered.budget.lines", "create", payload)
            created += len(payload)

    logger.info("created %d forecast lines", created)
    return created


def main(settings: Settings | None = None) -> int:
    """Publish the current forecast to Odoo as a budget version."""
    from fpa.ledger.odoo_load import OdooClient, ensure_accounts, ensure_cost_centers, ensure_user_groups
    from fpa.pipeline import run

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = settings or get_settings()

    result = run(settings)
    # The gate applies here too. A forecast that failed its controls must not be
    # published to the system of record — that is the whole point of having one.
    result.require_gate()
    if result.forecast is None:
        raise RuntimeError("no forecast in this run")

    client = OdooClient(
        settings.odoo_url, settings.odoo_db, settings.odoo_user, settings.odoo_password
    ).connect()
    ensure_user_groups(client)

    company_id = client.execute("res.company", "search", [], limit=1)[0]
    account_ids = ensure_accounts(client, company_id)
    center_ids = ensure_cost_centers(client)

    lines = seed_forecast_version(
        client, result.forecast, result.ledger, center_ids, account_ids, company_id
    )
    print(f"Published {lines} forecast lines to Odoo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
