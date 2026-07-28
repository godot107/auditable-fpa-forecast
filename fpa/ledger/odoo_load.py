"""Seed Odoo with the chart of accounts, cost centers, journal entries and budgets.

Odoo is the **system of record** for actuals. The app never queries it live —
``fpa.extract.odoo_sql`` materializes a snapshot and everything downstream reads
that. This is both demo-safe (the ERP can be off) and how FP&A actually works:
nobody runs planning queries against a live transactional ERP.

Two deliberate choices about volume:

* **Monthly journal entries, not daily.** One entry per month with a line per
  cost center gives ~78 entries and ~1,400 lines. Daily transactions would be
  ~2,400 entries and 40,000+ lines, which over XML-RPC takes long enough to
  derail a one-week build and demonstrates nothing extra.
* **Batched ``create``.** Odoo's ``create`` accepts a list, so lines go in one
  round trip per entry rather than one per line. The difference is minutes
  versus an hour.

Analytic accounts are used as cost centers — that is Odoo's native mechanism for
attributing a journal line to a department, and it is what ``mis_builder`` and
``account-budgeting`` read.
"""

from __future__ import annotations

import logging
import xmlrpc.client
from dataclasses import dataclass

import pandas as pd

from fpa.config import COST_CENTERS, Settings, balance_sheet_lines, get_settings

logger = logging.getLogger(__name__)

# Chart of accounts: our internal account codes mapped to an Odoo account code
# and type. Expense accounts only — revenue is posted to a single income account.
ACCOUNT_MAP: dict[str, tuple[str, str, str]] = {
    # internal            code     name                        odoo account_type
    "cost_of_revenue": ("500000", "Cost of Revenue", "expense_direct_cost"),
    "research_development": ("600000", "Technology & Development", "expense"),
    "marketing": ("610000", "Marketing", "expense"),
    "general_administrative": ("620000", "General & Administrative", "expense"),
}
REVENUE_ACCOUNT = ("400000", "Streaming Revenue", "income")

# The offset for the cost-center allocation journal, and it sits in the 9xxxxx
# range deliberately: it is not part of the filed balance sheet and the balance
# sheet report excludes it by code.
#
# This is what an allocation cycle does in SAP or Oracle — it redistributes cost
# across cost centers and settles against a clearing account, because allocation
# is a management-accounting overlay on top of the statutory ledger rather than a
# second set of books. The statutory side is loaded separately, at filed values,
# by :func:`seed_balance_sheet`.
CLEARING_ACCOUNT = ("990000", "Cost Allocation Offset", "liability_current")


@dataclass
class OdooClient:
    """Thin XML-RPC wrapper. Odoo's external API is `execute_kw` over `/xmlrpc/2`."""

    url: str
    db: str
    username: str
    password: str
    uid: int | None = None

    def connect(self) -> "OdooClient":
        common = xmlrpc.client.ServerProxy(f"{self.url}/xmlrpc/2/common")
        self.uid = common.authenticate(self.db, self.username, self.password, {})
        if not self.uid:
            raise RuntimeError(
                f"Odoo authentication failed for {self.username!r} on database {self.db!r}"
            )
        logger.info("connected to Odoo at %s as uid=%s", self.url, self.uid)
        return self

    @property
    def models(self):
        return xmlrpc.client.ServerProxy(f"{self.url}/xmlrpc/2/object")

    def execute(self, model: str, method: str, *args, **kwargs):
        return self.models.execute_kw(
            self.db, self.uid, self.password, model, method, list(args), kwargs or {}
        )

    def json_execute(self, model: str, method: str, *args, **kwargs):
        """Same call over JSON-RPC, for results XML-RPC cannot carry.

        XML-RPC has no null. Odoo's marshaller raises ``cannot marshal None unless
        allow_none is enabled`` on any response containing one, which makes it
        unusable for reading computed reports — a ``mis.report.instance.compute()``
        matrix is full of nulls for empty cells. The report is fine; the transport
        is not. JSON-RPC has ``null`` and handles it.

        Writes stay on XML-RPC: it is what the rest of the seeder uses and the
        payloads never contain None.
        """
        import json
        import urllib.request

        payload = {
            "jsonrpc": "2.0",
            "method": "call",
            "params": {
                "service": "object",
                "method": "execute_kw",
                "args": [self.db, self.uid, self.password, model, method, list(args), kwargs or {}],
            },
            "id": 1,
        }
        request = urllib.request.Request(
            f"{self.url}/jsonrpc",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            body = json.loads(response.read())
        if "error" in body:
            raise RuntimeError(body["error"].get("data", {}).get("message", body["error"]))
        return body["result"]

    def find_or_create(self, model: str, domain: list, values: dict) -> int:
        """Idempotent upsert — re-running the seeder must not duplicate anything."""
        existing = self.execute(model, "search", domain, limit=1)
        if existing:
            return existing[0]
        return self.execute(model, "create", values)


# Permission groups the seeder needs. Odoo 18 hides analytic accounting behind a
# feature group, so a fresh admin cannot touch analytic plans until it is granted.
# Listed as (module, xmlid) so the seeder provisions its own prerequisites instead
# of failing with a permission error and expecting someone to find the checkbox.
REQUIRED_GROUPS: tuple[tuple[str, str], ...] = (
    ("analytic", "group_analytic_accounting"),
    ("account", "group_account_manager"),
    # "Show Full Accounting Features" — gates account.budget.post (Budgetary
    # Position), which OCA's budget lines require.
    ("account", "group_account_user"),
    ("base", "group_no_one"),  # exposes technical fields the budget models need
)


def ensure_user_groups(client: OdooClient) -> list[str]:
    """Grant the connecting user the groups required to seed analytic data."""
    granted: list[str] = []
    for module, xmlid in REQUIRED_GROUPS:
        records = client.execute(
            "ir.model.data",
            "search_read",
            [("module", "=", module), ("name", "=", xmlid)],
            fields=["res_id"],
            limit=1,
        )
        if not records:
            logger.debug("group %s.%s not present in this database", module, xmlid)
            continue
        # (4, id) is Odoo's "link existing record" command for x2many writes.
        client.execute(
            "res.users", "write", [client.uid], {"groups_id": [(4, records[0]["res_id"])]}
        )
        granted.append(f"{module}.{xmlid}")

    if granted:
        logger.info("granted groups: %s", ", ".join(granted))
    return granted


def ensure_accounts(client: OdooClient, company_id: int) -> dict[str, int]:
    """Create the chart of accounts, returning internal-code → Odoo account id.

    Covers all three statements: the P&L accounts the allocation journal posts to,
    and every balance-sheet line from :mod:`fpa.config`. The cash-flow statement
    gets no accounts, because no ledger journalizes a cash-flow statement — it is
    derived from the movement in these balances, which is why a consolidation tool
    computes it and a transactional system does not.
    """
    ids: dict[str, int] = {}
    entries = [
        *((k, *v) for k, v in ACCOUNT_MAP.items()),
        ("revenue", *REVENUE_ACCOUNT),
        ("clearing", *CLEARING_ACCOUNT),
        *(
            (line.account, line.code, line.name, line.account_type)
            for line, _sign in balance_sheet_lines()
        ),
    ]
    for internal, code, name, account_type in entries:
        ids[internal] = client.find_or_create(
            "account.account",
            [("code", "=", code), ("company_ids", "in", [company_id])],
            {
                "code": code,
                "name": name,
                "account_type": account_type,
                "company_ids": [(6, 0, [company_id])],
            },
        )
        # Odoo 18 installs a default chart of accounts, so several of these codes
        # already exist under stock names ("Cost of Goods Sold", "Bank Fees").
        # Matching by code returns those, and the extract then reports the wrong
        # account name. Overwrite so the chart reads as the one this demo defines.
        client.execute(
            "account.account", "write", [ids[internal]], {"name": name, "account_type": account_type}
        )
        logger.debug("account %s (%s) -> id %s", code, name, ids[internal])
    return ids


def ensure_cost_centers(client: OdooClient) -> dict[tuple[str, str], int]:
    """Create one analytic account per leaf cost center, on the **default** plan.

    Odoo 17+ gives every analytic plan its own column on ``account.analytic.line``
    — only the first root plan writes to ``account_id``; a second plan lands in a
    generated ``x_plan<id>_id`` column whose name depends on the database. Creating
    a fresh "Cost Centers" plan therefore produced 858 analytic lines with a NULL
    ``account_id`` and an extract that returned nothing.

    So we reuse the existing root plan and rename it, which keeps ``account_id``
    populated and the SQL in ``sql/`` portable across databases.
    """
    roots = client.execute(
        "account.analytic.plan", "search", [("parent_id", "=", False)], order="id asc", limit=1
    )
    if roots:
        plan_id = roots[0]
        client.execute("account.analytic.plan", "write", [plan_id], {"name": "Cost Centers"})
    else:
        plan_id = client.execute("account.analytic.plan", "create", {"name": "Cost Centers"})

    ids: dict[tuple[str, str], int] = {}
    for function, sub_centers in COST_CENTERS.items():
        for sub_center in sub_centers:
            # Name carries the hierarchy so the ERP shows the same three levels
            # the forecast reconciles over.
            name = f"{function} / {sub_center}"
            ids[(function, sub_center)] = client.find_or_create(
                "account.analytic.account",
                [("name", "=", name)],
                {"name": name, "plan_id": plan_id},
            )
    logger.info("ensured %d analytic cost centers", len(ids))
    return ids


def _entry_lines(
    month_rows: pd.DataFrame,
    account_ids: dict[str, int],
    center_ids: dict[tuple[str, str], int],
    revenue: float = 0.0,
) -> list[tuple]:
    """Build balanced debit/credit lines for one month's income statement.

    Revenue is posted alongside the allocated expenses so the ERP holds a complete
    P&L rather than a cost ledger — without it, a `mis_builder` income-statement
    report has no top line and no margin to compute.

    The clearing account takes whatever is left, which for a profitable month is a
    *debit*. That is the correct behaviour and worth reading as what it is: the
    residual is net income awaiting close, which is exactly what a suspense account
    holds between the allocation run and the period close.
    """
    lines: list[tuple] = []
    expense_total = 0.0
    for row in month_rows.itertuples():
        amount = round(float(row.amount), 2)
        if amount == 0:
            continue
        expense_total += amount
        analytic_id = center_ids[(row.function, row.sub_center)]
        lines.append(
            (
                0,
                0,
                {
                    "account_id": account_ids[row.account],
                    "name": f"{row.function} / {row.sub_center}",
                    "debit": amount,
                    "credit": 0.0,
                    # Odoo 17+ analytic distribution: {analytic_account_id: percent}
                    "analytic_distribution": {str(analytic_id): 100.0},
                },
            )
        )

    revenue = round(float(revenue), 2)
    if revenue:
        lines.append(
            (
                0,
                0,
                {
                    "account_id": account_ids["revenue"],
                    "name": "Streaming revenue",
                    "debit": 0.0,
                    "credit": revenue,
                },
            )
        )

    # Balancing line so the entry is genuine double-entry. Debit or credit
    # depending on which side is short.
    residual = round(expense_total - revenue, 2)
    if residual:
        lines.append(
            (
                0,
                0,
                {
                    "account_id": account_ids["clearing"],
                    "name": "Cost allocation offset",
                    "debit": -residual if residual < 0 else 0.0,
                    "credit": residual if residual > 0 else 0.0,
                },
            )
        )
    return lines


def seed_journal_entries(
    client: OdooClient,
    ledger: pd.DataFrame,
    account_ids: dict[str, int],
    center_ids: dict[tuple[str, str], int],
    journal_id: int,
    revenue: pd.DataFrame | None = None,
) -> int:
    """Post one journal entry per month: revenue plus a line per cost center."""
    by_month = (
        revenue.set_index("period")["amount"] if revenue is not None else pd.Series(dtype=float)
    )

    created = 0
    new_ids: list[int] = []
    for period, month_rows in ledger.groupby("period"):
        date = pd.Timestamp(period).date().isoformat()
        ref = f"FPA-{pd.Timestamp(period):%Y-%m}"

        if client.execute("account.move", "search", [("ref", "=", ref)], limit=1):
            continue  # idempotent: already seeded

        new_ids.append(
            client.execute(
                "account.move",
                "create",
                {
                    "move_type": "entry",
                    "date": date,
                    "ref": ref,
                    "journal_id": journal_id,
                    "line_ids": _entry_lines(
                        month_rows,
                        account_ids,
                        center_ids,
                        float(by_month.get(period, 0.0)),
                    ),
                },
            )
        )
        created += 1
    logger.info("created %d journal entries", created)

    # Post them. A draft entry is not in the books, and — the reason this matters
    # here — Odoo only generates `account.analytic.line` records on posting. Those
    # analytic lines are what OCA's budget `practical_amount` sums, so leaving the
    # entries in draft yields a budget report with plans and no actuals.
    draft_ids = client.execute(
        "account.move", "search", [("ref", "like", "FPA-"), ("state", "=", "draft")]
    )
    if draft_ids:
        client.execute("account.move", "action_post", draft_ids)
        logger.info("posted %d journal entries", len(draft_ids))

    return created


def seed_balance_sheet(
    client: OdooClient,
    balance_sheet: pd.DataFrame,
    account_ids: dict[str, int],
    journal_id: int,
) -> int:
    """Post the filed balance sheet, one movement entry per quarter.

    **No clearing account, and that is the point.** Each entry books the movement
    in every balance-sheet account from the prior quarter end. Because
    ``Assets = Liabilities + Equity`` holds at both dates, the movements net to
    zero and the entry balances on its own — the debits and credits are the filed
    statement, not a construction. Odoo would reject the entry outright if the
    filed balance sheet did not balance, which makes the ERP an independent check
    on the ingest rather than just a destination for it.

    Signs follow the natural balance from :func:`fpa.config.balance_sheet_lines`:
    assets carry debit balances, liabilities and equity carry credit balances.
    Treasury stock falls out of this correctly with no special case — its filed
    value is negative and its natural side is credit, so it lands as a debit
    balance, which is exactly what contra-equity is.

    This is a period-end position load, the same shape as a consolidation system
    taking a trial balance from a subsidiary. It is not an attempt to synthesize
    the transactions behind the balances, and the README says so.
    """
    lines = [(line, sign) for line, sign in balance_sheet_lines() if line.account in balance_sheet.columns]
    if not lines:
        logger.warning("no balance-sheet lines available — skipping")
        return 0

    # Target ledger balance (debit − credit) per account, per quarter end.
    targets = pd.DataFrame(
        {line.account: balance_sheet[line.account] * sign for line, sign in lines},
        index=balance_sheet.index,
    ).sort_index()
    # First quarter posts the opening position; each later one posts the movement.
    movements = targets.diff()
    movements.iloc[0] = targets.iloc[0]

    created = 0
    for period, row in movements.iterrows():
        ref = f"BS-{pd.Timestamp(period):%Y-%m-%d}"
        if client.execute("account.move", "search", [("ref", "=", ref)], limit=1):
            continue  # idempotent

        entry_lines = []
        for line, _sign in lines:
            amount = round(float(row[line.account]), 2)
            if amount == 0:
                continue
            entry_lines.append(
                (0, 0, {
                    "account_id": account_ids[line.account],
                    "name": f"{line.name} — movement",
                    "debit": amount if amount > 0 else 0.0,
                    "credit": -amount if amount < 0 else 0.0,
                })
            )

        if not entry_lines:
            continue

        # Cent rounding on each line can leave a sub-dollar imbalance that Odoo
        # will refuse. Absorb it into the largest line rather than inventing a
        # rounding account for what is at most a few cents.
        residual = round(sum(v["debit"] - v["credit"] for _c, _z, v in entry_lines), 2)
        if residual:
            biggest = max(entry_lines, key=lambda le: abs(le[2]["debit"] - le[2]["credit"]))[2]
            net = round(biggest["debit"] - biggest["credit"] - residual, 2)
            biggest["debit"] = net if net > 0 else 0.0
            biggest["credit"] = -net if net < 0 else 0.0

        client.execute("account.move", "create", {
            "move_type": "entry",
            "date": pd.Timestamp(period).date().isoformat(),
            "ref": ref,
            "journal_id": journal_id,
            "line_ids": entry_lines,
        })
        created += 1

    logger.info("created %d balance-sheet entries", created)

    draft_ids = client.execute(
        "account.move", "search", [("ref", "like", "BS-"), ("state", "=", "draft")]
    )
    if draft_ids:
        client.execute("account.move", "action_post", draft_ids)
        logger.info("posted %d balance-sheet entries", len(draft_ids))

    return created


def reset_seed_data(client: OdooClient) -> int:
    """Delete every entry this seeder created, so a reseed starts clean.

    Needed whenever the chart of accounts changes meaning rather than just growing:
    the seeder matches accounts by code, so renumbering one leaves historical lines
    pointing at an account that now means something else. Deleting and reposting is
    the only honest way to change a chart of accounts under existing postings.
    """
    move_ids = client.execute(
        "account.move", "search", ["|", ("ref", "like", "FPA-"), ("ref", "like", "BS-")]
    )
    if not move_ids:
        return 0
    # A posted entry cannot be deleted; it has to be reset to draft first.
    client.execute("account.move", "button_draft", move_ids)
    client.execute("account.move", "unlink", move_ids)
    logger.info("deleted %d seeded journal entries", len(move_ids))
    return len(move_ids)


def ensure_budget_positions(
    client: OdooClient, account_ids: dict[str, int], company_id: int
) -> dict[str, int]:
    """Create one budgetary position per expense account.

    ``account.budget.post`` is the bridge OCA's budget module uses to tie a budget
    line to GL accounts: its ``account_ids`` are what ``practical_amount`` sums
    actuals over. Without it a budget line has a plan and no actual to compare to.
    """
    positions: dict[str, int] = {}
    for internal, (code, name, _type) in ACCOUNT_MAP.items():
        positions[internal] = client.find_or_create(
            "account.budget.post",
            [("name", "=", name), ("company_id", "=", company_id)],
            {
                "name": name,
                "company_id": company_id,
                "account_ids": [(6, 0, [account_ids[internal]])],
            },
        )
    return positions


def _sync_budget_lines(client: OdooClient, header_id: int, expected: int, label: str) -> bool:
    """Return True when ``header_id`` needs its lines written.

    Three outcomes, and the middle one is the one that matters:

    * no lines yet — write them;
    * the right number of lines — leave them alone (idempotent re-run);
    * the *wrong* number — delete and rewrite, because the plan changed shape.

    Testing only for presence is what let a six-month fiscal year stay six months
    after the plan was extended to twelve. Testing the count makes the seeder a
    sync rather than a one-shot.
    """
    existing = client.execute(
        "crossovered.budget.lines", "search", [("crossovered_budget_id", "=", header_id)]
    )
    if not existing:
        return True
    if len(existing) == expected:
        logger.info("%s already matches (%d lines) — skipping", label, expected)
        return False

    logger.info("%s changed (%d lines -> %d) — replacing", label, len(existing), expected)
    client.execute("crossovered.budget.lines", "unlink", existing)
    return True


def seed_budgets(
    client: OdooClient,
    budget: pd.DataFrame,
    center_ids: dict[tuple[str, str], int],
    account_ids: dict[str, int],
    company_id: int,
) -> int:
    """Load the plan into Odoo so budget-vs-actual is visible inside the ERP.

    Uses ``account_budget_oca``'s ``crossovered.budget`` model. Skipped with a
    warning when the module is absent — the Python planning layer is authoritative
    either way, so a missing module degrades the demo rather than breaking it.

    Actuals are not loaded alongside: Odoo derives ``practical_amount`` itself from
    the ``account.analytic.line`` records generated by the journal entries'
    analytic distribution. That is the point of posting real double-entry
    transactions rather than summary numbers — the ERP computes the comparison.
    """
    installed = client.execute(
        "ir.module.module",
        "search",
        [("name", "=", "account_budget_oca"), ("state", "=", "installed")],
        limit=1,
    )
    if not installed:
        logger.warning(
            "account_budget_oca not installed — skipping the Odoo budget load. "
            "Run docker/fetch-addons.sh, then reinstall with -i account_budget_oca."
        )
        return 0

    positions = ensure_budget_positions(client, account_ids, company_id)

    created = 0
    for year, year_rows in budget.groupby(budget["period"].dt.year):
        header_id = client.find_or_create(
            "crossovered.budget",
            [("name", "=", f"FY{year} Plan"), ("company_id", "=", company_id)],
            {
                "name": f"FY{year} Plan",
                "date_from": f"{year}-01-01",
                "date_to": f"{year}-12-31",
                "company_id": company_id,
            },
        )

        # Idempotency has to be checked on the *lines*, not just the header:
        # find_or_create returns the existing header on a re-run, and creating
        # lines against it again silently doubles the plan.
        #
        # Compare the count rather than merely testing for presence, so a plan that
        # legitimately changed shape — a fiscal year extended from six months to
        # twelve, say — actually propagates instead of being skipped forever.
        if not _sync_budget_lines(client, header_id, len(year_rows), f"FY{year} plan"):
            continue

        # One line per month x cost center x account.
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
                    # Negative: Odoo's analytic lines carry costs as negative
                    # amounts, so a plan must use the same sign to compare.
                    "planned_amount": -round(float(row.budget), 2),
                }
            )

        if payload:
            client.execute("crossovered.budget.lines", "create", payload)
            created += len(payload)

    logger.info("created %d budget lines", created)
    return created


def main(settings: Settings | None = None, *, reset: bool = False) -> int:
    """Seed a running Odoo instance from the pipeline's statements and budget."""
    from fpa.ledger.budget import build_budget
    from fpa.pipeline import build_ledger

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = settings or get_settings()

    client = OdooClient(
        settings.odoo_url, settings.odoo_db, settings.odoo_user, settings.odoo_password
    ).connect()

    ensure_user_groups(client)

    company_id = client.execute("res.company", "search", [], limit=1)[0]
    journal_id = client.find_or_create(
        "account.journal",
        [("code", "=", "FPA"), ("company_id", "=", company_id)],
        {"name": "FP&A Allocations", "code": "FPA", "type": "general", "company_id": company_id},
    )
    # Separate journal, because these are two different things: a management
    # allocation overlay and a statutory position load. Keeping them apart is what
    # lets a report read one without the other.
    bs_journal_id = client.find_or_create(
        "account.journal",
        [("code", "=", "BS"), ("company_id", "=", company_id)],
        {
            "name": "Balance Sheet — Filed Positions",
            "code": "BS",
            "type": "general",
            "company_id": company_id,
        },
    )

    if reset:
        reset_seed_data(client)

    context = build_ledger(settings)
    account_ids = ensure_accounts(client, company_id)
    center_ids = ensure_cost_centers(client)

    entries = seed_journal_entries(
        client, context.ledger, account_ids, center_ids, journal_id, context.revenue
    )
    bs_entries = seed_balance_sheet(client, context.balance_sheet, account_ids, bs_journal_id)
    # Untrimmed: the ERP holds the plan as committed, all twelve months, so the
    # rolling forecast has something to be compared against in the months that
    # have not happened yet. ``context.budget`` is trimmed for variance reporting.
    full_year_plan = build_budget(settings, context.ledger, trim_to_actuals=False)
    budget_lines = seed_budgets(client, full_year_plan, center_ids, account_ids, company_id)

    lines = len(context.ledger) + context.ledger["period"].nunique()
    print(
        f"Seeded Odoo: {entries} allocation entries (~{lines} lines), "
        f"{bs_entries} balance-sheet entries, {len(center_ids)} cost centers, "
        f"{budget_lines} budget lines."
    )
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete previously seeded entries first (required after a chart-of-accounts renumbering)",
    )
    raise SystemExit(main(reset=parser.parse_args().reset))
