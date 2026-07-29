# auditable-fpa-forecast

An auditable FP&A rolling-forecast pipeline. Portfolio piece for a Finance Data & AI
role. Built around a single concern: the gap between demo-AI and production-AI — the
reason finance teams reject a number whose provenance cannot be shown.

Actuals come from SEC EDGAR XBRL and carry the accession number of the filing they were
tagged in. All three statements are built and their articulation is asserted before
anything reads them. Cost-center detail below the filed lines is modeled and asserted to
foot back. An LLM writes variance commentary and is structurally prevented from producing
a number.

## Run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                     # set EDGAR_USER_AGENT

.venv/bin/python -m fpa                  # ingest -> controls -> forecast
.venv/bin/python -m fpa --refresh        # re-pull EDGAR instead of the pinned vintage
.venv/bin/python -m pytest               # 123 tests (pytest.ini scopes to tests/)
.venv/bin/pip install -r requirements-bayes.txt   # optional: NumPyro + JAX
.venv/bin/python -m fpa --groundedness   # + checker error rates (fast; exits 1 if unclean)
.venv/bin/python -m fpa --intervals      # + posterior-predictive calibration (~30 min)
.venv/bin/streamlit run app/Home.py

# ERP layer (optional; the app reads a snapshot, so Odoo can be off)
./docker/fetch-addons.sh
docker compose -f docker/docker-compose.yml up -d
.venv/bin/python -m fpa.ledger.odoo_load
.venv/bin/python -m fpa.ledger.odoo_load --reset   # after a chart-of-accounts change
.venv/bin/python -m fpa.ledger.forecast_version    # publish the forecast as a budget version
.venv/bin/python -m fpa.ledger.mis_reports         # load the MIS report definitions
```

## Layout

| Path | Role | Status |
|---|---|---|
| `fpa/ingest/edgar.py` | EDGAR XBRL, Q4 derivation, restatement dedupe, audit trail | Built |
| `fpa/ingest/statements.py` | Three statements, derived lines, articulation | Built |
| `fpa/ingest/segments.py` | Regional revenue from SEC Financial Statement Data Sets | Built |
| `fpa/ledger/forecast_version.py` | Forecast published to Odoo as a budget version | Built |
| `fpa/ledger/mis_reports.py` | MIS Builder KPI definitions, committed as data | Built |
| `fpa/ledger/` | Disaggregation (foots to filed), budget, Odoo seeder | Built |
| `fpa/controls/checks.py` | 23 controls, 21 blocking, registry + gate | Built |
| `fpa/forecast/` | Seasonal-naive benchmark, ETS, rolling-origin backtest | Built |
| `fpa/extract/odoo_sql.py` | SQL extract -> pinned Parquet | Built |
| `fpa/variance/bridge.py` | Spend/mix decomposition, ties by construction | Built |
| `fpa/narrative/` | Facts payload, providers, groundedness, draft gate | Built |
| `fpa/narrative/evaluation.py` | Checker error rates: FAR, FRR, parse coverage | Built |
| `fpa/audit/log.py` | Append-only approval log | Built |
| `fpa/kpi/finops.py` | Technology unit economics | **Scaffold only** |
| `fpa/forecast/bayes.py` | NumPyro local linear trend, coverage + sharpness | Built (opt-in) |

## Day-1 invariants (do not regress)

1. **Controls run inside the pipeline, every run — not only in CI.** A control that only
   runs in a test suite protects the developer; one that runs in the pipeline protects
   the number. A blocking failure stops the forecast *and* the commentary.
2. **The LLM writes the commentary, never the number.** Every figure is computed in
   Python. Any numeral in generated prose absent from the facts payload rejects the
   draft. Borrowed from `energy-batch-trader`: the LLM runs a gate, not a decision.
3. **Modeled detail must foot to filed totals.** `_exact_split` forces this; the
   `ledger_foots_to_filed` control re-proves it at 1e-12 relative.
4. **Never publish a metric without its benchmark.** MASE against seasonal-naive, and
   the series where the model loses are reported, not hidden.
5. **Everything reads a pinned vintage.** No network call on the demo path.
6. **Provenance badges on every displayed figure** — `REAL` / `MODELED` / `IMPLIED` /
   `FORECAST`.
7. **A residual is only a financial statement line if it behaves like one.** Where a
   line cannot be obtained as a tag it is derived — but it must be *named* as the
   caption it represents, its sign is asserted every run, and its magnitude is checked
   against something external. A plug that is merely "whatever is left" is not allowed.
8. **Nothing may be silently assumed to be zero.** `DataFrame.sum` skips NaN, so a
   missing input contributes nothing and the result looks complete. Every derivation
   uses `skipna=False` unless the tag is in `ABSENT_MEANS_ZERO`, and that set is
   reported on screen with a period count.

## Key decisions

- **Window starts 2020-01, and that was measured.** Q4 is never filed on a 10-Q, so it is
  derived from the 10-K. Netflix's FY2017–FY2019 annuals do not articulate against this
  chart of accounts (−$27.3M, +$3.8M, −$127.9M); FY2020+ tie to the dollar. The pipeline
  refuses to derive a quarter from a year it cannot validate.
- **Marketing is derived, not estimated.** `MarketingExpense` stops at 2024-09-30;
  recovered from `Revenue − CostOfRevenue − R&D − G&A − OperatingIncome`, exact to the
  dollar across all 19 overlap quarters.
- **MASE, not MAPE.** MAPE is dominated by small denominators, so at cost-center
  granularity a small department outweighs Content (Phillips, *Pricing & Revenue
  Optimization* 2e, p.94).
- **Two backtests, and the honest one leads.** The monthly ledger is disaggregated by
  this project, so a model scores 0.632 on it and only 0.936 on filed quarters. The gap
  is the artifact, measured rather than assumed.
- **The interval layer is opt-in, not on the demo path.** Each fold is a full NUTS
  fit — ~6 minutes against ~10 seconds for everything else, and a demo that takes six
  minutes is a demo nobody runs. `--intervals` writes `reports/interval_calibration.md`.
- **Three MCMC diagnostics, never one.** Divergences, R-hat and ESS fail differently, and
  R-hat is uncomputable from a single chain — so the sampler runs **4 vectorized chains**.
  An earlier version reported divergences alone and looked healthy; adding the other two
  showed **R-hat 1.93, ESS 3**.
- **Tuning on one diagnostic destroyed the others.** An earlier `target_accept_prob=0.99`
  was chosen because it drove divergences to ~zero. Measured properly it is catastrophic —
  the step size collapses, the chains freeze, and *nothing diverges because nothing moves*:

  | `target_accept` | divergences | R-hat | ESS |
  |---|---|---|---|
  | 0.80 | 85–142 | 1.010–1.028 | 141–551 |
  | 0.90 | 32–189 | 1.007–1.037 | 121–387 |
  | 0.99 | **0–38** | **999–1804** | **2** |

  Now 0.90. This is the sharpest illustration in the repo of the project's own thesis:
  optimizing the metric you happen to be watching can wreck the ones you are not.
- **The model had a non-identified parameter.** It carried both a level shock and
  observation noise; both explain month-to-month variation and the data cannot separate
  them, so chains settled at `sigma_obs` means spanning 0.012–0.025. Dropping the level
  shock leaves an identified smooth-trend model (R-hat 1.93 → ≤1.03).
- **Priors are checked by simulation, not asserted in a comment.** The prior predictive
  showed `trend0 ~ N(0, 0.10)` implied a 90% band on *annual* growth of 0.16×–6.4×, with
  draws at 127×. Now `N(0, 0.02)` → 0.58×–1.68×. McElreath 2e p.114: there is no other
  reliable way to understand what a prior implies.
- **Coverage is reported per series, not averaged into a headline.** The mean is 87%
  against a nominal 80%, which looks fine and hides that one series covers 100% and
  another 78%. Marketing over-covers *while losing to the benchmark* — buying calibration
  with width.
- **The interval layer is provisional, and the report says so.** 8 of 9 *backtest* fits
  fail R-hat ≤ 1.01 or ESS ≥ 400, because rolling origin trains on 42–66 months rather
  than 78 and short series identify the scales poorly. Two of the three series that beat
  naive are the worst-converged rows (R-hat 1.99 / ESS 3); the one clean fit loses. Do not
  quote a headline from this table until short-window sampling is fixed.
- **NumPyro, not Pyro/torch.** House choice (see `financial-forecasting-engine`), and it
  dropped ~2 GB. The previous scaffold imported Pyro to run a loop that did no inference —
  never ship a probabilistic *label* without inference behind it.
- **Odoo is the system of record, not the planning engine.** Planning lives in Python.
  ERP → EPM is the real enterprise pattern. OCA modules (`mis_builder`,
  `account_budget_oca`) supply the reporting half; stock Odoo is an ERP.
- **yfinance is not used to fill gaps.** A vendor figure has no accession number and uses
  a normalized basis. A documented gap beats an untraceable number.
- **The balance sheet posts with no clearing account.** `A = L + E` holds to **$0.00**
  across all 26 quarters, so a quarterly movement entry balances on its own — the debits
  and credits *are* the filed statement. Odoo rejects the entry if it does not balance,
  which makes the ERP an independent check on the ingest rather than just a destination.
- **The audit trail crosses into the ERP, in `narration` rather than `ref`.** Every
  entry carries the accession number, form, filing date and EDGAR index URL of the
  document behind it, plus a statement of which of its lines are filed and which are
  derived. Previously the trail stopped at the ERP boundary: the entry said
  `BS-2026-03-31`, a period key, so validating a figure against EDGAR meant leaving
  the ledger. It goes in `narration` because `ref` is the idempotency key — changing
  its format would re-post the whole history instead of skipping it — and existing
  entries are backfilled rather than requiring `--reset`.
- **The two journals' provenance blocks reach opposite conclusions, deliberately.**
  The balance-sheet block names the filing and separates the 12 `REAL` lines from the
  5 `IMPLIED` residuals. The allocation block names the same filing and then *denies
  being it*: "this month is not a filed figure." A note that cites a 10-Q without
  saying that implies the month came from it.
- **One accession per balance-sheet date holds only inside the window.** All 26
  quarter ends from 2020 trace to a single filing, which is what makes a one-document
  citation honest. Before the window, 33 dates draw on up to **three** — `cash` at
  2013-03-31 was last restated in a 10-Q filed July 2014 while its neighbours still
  come from the original April 2013 filing. A balance sheet *as of* a date is not a
  balance sheet *as filed in one document*. The formatter takes a list for that
  reason, and a test uses the pre-window scatter as its fixture.
- **The cash-flow statement is deliberately not posted.** No general ledger journalizes
  one; it is derived from the movement in balance-sheet accounts. Computing it in Python
  and reconciling it is the correct division of labour, not a shortcut.
- **Balances post as movements, not restated positions.** So a wrong entry in 2021 shows
  up in all twenty quarters after it. Landing on the filed figure at all 26 quarter ends
  is a much stronger statement than 26 independent comparisons.
- **Not every filed tag is a line on the face of the statement.** The ASC 842 lease tags
  are disclosures nested *inside* the "other non-current" captions. Posting them
  alongside double-counted, producing a **−$571M liability**. Settled empirically by the
  2019-Q1 adoption step: `OtherAssetsNoncurrent` jumped $816M in the quarter $812M of
  right-of-use assets were first recognised. A caption that jumps by the amount of the
  thing being adopted contains it.
- **The ERP holds the scenario dimension, not a forecasting engine.** Odoo has none and
  neither does `mis_builder` — nor should they. What an EPM tool adds over an ERP is
  Actual / Plan / Forecast against the same accounts, and `crossovered.budget` is a
  versioned container, so the Python forecast loads as `FY2026 Forecast` beside
  `FY2026 Plan`. Actual is not loaded at all: Odoo derives it from the analytic lines.
- **Scenarios propagate assumptions; they do not estimate responses.** Volume × rate lets a
  driver assumption be changed and re-footed through the hierarchy — *"if content spend
  grows 15%, what happens to margin?"* Netflix filed **one** price path, so nothing in this
  data identifies how members respond to a price change; *"what happens when we decide to
  raise price?"* needs a counterfactual the filings do not contain. The README claimed
  churn/elasticity what-ifs the code never implemented — corrected, and the honest version
  is on the roadmap. Same rule as the vendor-feed decision: a number nobody can trace to a
  source is worse than an absent one.
- **The plan published to the ERP is untrimmed; the plan used for variance is not.**
  `build_budget(trim_to_actuals=...)`. Variance must drop plan months with no actual, or
  a month that has not happened reports as 100% under budget. The ERP must keep all
  twelve, or the rolling forecast has nothing to be compared against.
- **Report definitions live in the repo, not in one database.** A report clicked together
  in the web UI cannot be reviewed, diffed or rebuilt. Same argument as keeping the SQL
  in `sql/` as files.
- **Regional revenue needs a different SEC product.** `companyfacts` has no dimension
  field, so it only ever exposes the consolidated value of a tag. The Financial Statement
  Data Sets carry a `segments` column — bulk ZIPs, ~85 MB each, streamed and filtered.
  `adsh` *is* the accession number, so the audit trail survives the switch.
- **The XBRL unit is declared per tag, never assumed.** `companyfacts` keys facts by
  unit, so reading a per-share concept from `units["USD"]` returns an empty list rather
  than raising — the tag silently vanishes. Three units are in play: `USD`, `shares`,
  `USD/shares`. `eps_consistency` exists specifically to catch a regression here.

## Odoo 18 gotchas (each cost a debugging cycle)

- `mis_builder` needs `report_xlsx` (OCA/reporting-engine), `date_range` (OCA/server-ux)
  and the `openupgradelib` pip package — hence `docker/Dockerfile`.
- Analytic accounting and budgetary positions sit behind permission groups. The seeder
  grants its own (`ensure_user_groups`).
- **Analytic lines only generate on posting.** Draft entries yield a budget report with
  plans and no actuals.
- **Only the default root analytic plan writes to `account_analytic_line.account_id`.** A
  second plan lands in a generated `x_plan<id>_id` column and the extract returns nothing.
- `account_account.code` is company-dependent JSONB (`code_store`); names are
  translatable JSONB. Neither is a plain column.
- `--json-schema` on `claude -p` takes **inline JSON**, not a file path. The prompt goes
  over stdin (ARG_MAX); the schema is small enough for argv.
- **Renumbering an account requires `--reset`.** The seeder matches accounts by code, so
  changing what a code means leaves historical lines pointing at an account that now
  means something else. `reset_seed_data` resets posted moves to draft (`button_draft`)
  before unlinking, because a posted entry cannot be deleted.
- `account.account.account_type` is a closed vocabulary. `liability_payable` and
  `asset_receivable` are reconcilable and can demand a partner on the move line, so the
  balance-sheet loader uses `liability_current` for Accounts Payable instead.
- **XML-RPC cannot carry null.** Odoo's marshaller raises `cannot marshal None unless
  allow_none is enabled` on any *response* containing one, which makes it unusable for
  reading a `mis.report.instance.compute()` matrix — empty cells are nulls. Writes stay on
  XML-RPC; reads that may contain nulls use `OdooClient.json_execute` over `/jsonrpc`.
- **Seeding budget lines is a sync, not a one-shot.** `_sync_budget_lines` compares the
  line *count* and replaces on mismatch. Testing only for presence left a fiscal year at
  six months after the plan was extended to twelve.
- `pytest -q` at the repo root collects the four vendored OCA repos under
  `docker/addons/` — ~90 test modules that import `odoo` and fail collection outside a
  running server. `pytest.ini` pins `testpaths = tests`.

## Evaluating the evaluator

- **Three failure modes, not two.** False rejection and false acceptance are visible in
  `check()`'s output; a **parse miss** is not, because an unmatched numeral gets no verdict
  at all. The `$999.9M.` bug was a parse miss — silence, not a wrong answer — so it has its
  own metric, found by disagreement with a deliberately over-inclusive second parser.
- **Labels never consult the checker.** The first version of `evaluation.py` labelled cases
  with the checker's own matching rule, making both error rates tautologies. Labels now come
  from geometry: positives within `POSITIVE_PRECISION` (2e-4), negatives beyond
  `NEGATIVE_MARGIN` (2e-2), `MATCH_RTOL` (2e-3) strictly inside the gap and never used.
  Ambiguous cases are not generated at all.
- **The evaluation is tested by breaking the checker.** Four tests revert the real regex
  bug, widen the tolerance, tighten it, and drop scale suffixes — each asserting the
  evaluation goes red. Without them, "0% false acceptance" is indistinguishable from an
  instrument that cannot see.

## The bug class this codebase keeps finding

Three separate defects, all the same shape: **a missing input that produces a plausible
number instead of an error.**

1. `_derive_q4` summed Q1–Q3 with `.sum()`. Pre-tax income is only tagged as a quarter
   from 2020-Q3, so 2020-Q4 came out as `FY − Q3` — **$1.83B too high**, with two
   quarters of the year buried inside it.
2. `apply_derivations` summed balance-sheet parts the same way, so `other_current_assets`
   reported 26 of 26 quarters populated when 10 of them rested on an undeclared
   assumption about short-term investments.
3. The original ingest hardcoded `units["USD"]`, which for a per-share tag returns `[]`
   rather than raising.

None of the three raises. Each is caught by a control that tests an *identity*, not a
value — which is the argument for having them at all.

## Grounding (textbook KB)

| Source | What it changed |
|---|---|
| Phillips, *Pricing & Revenue Optimization* 2e, p.94 | MAPE → MASE + weighted MAPE |
| Nielsen, *Practical Time Series Analysis*, p.253 | Named the hierarchy; bottom-up reconciliation |
| McElreath, *Statistical Rethinking* 2e, p.223 | Intervals need sharpness, not just coverage |
| Huyen, *AI Engineering*, pp.219–225 | Groundedness as factual-consistency checking; bounded output length |
| López de Prado, *AFML* ch.7/12 | Rolling origin with a horizon gap — purge/embargo targets *overlapping labels*, which monthly FP&A periods do not have. Use the simpler correct thing and say why. |
