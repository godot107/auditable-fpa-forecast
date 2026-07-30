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
.venv/bin/python -m pytest               # 191 tests (pytest.ini scopes to tests/)
.venv/bin/pip install -r requirements-bayes.txt   # optional: NumPyro + JAX
.venv/bin/python -m fpa.forecast.posterior        # fit + pin the posterior (~1 MB, committed)
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
| `fpa/forecast/statements.py` | Three-statement forecast: revenue forecast, rest derived | Built |
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
- **MASE flatters a trending stock and punishes a lumpy flow.** The denominator is the
  average annual increment, which is large for a cumulative balance and small for a
  stationary flow. Measured: `assets` 0.683 and `equity` 0.548 "beat" naive while
  `free_cash_flow` scores **7.571**. That spread is partly the benchmark, not the model —
  seasonal-naive is weak on anything that trends. Two consequences, both acted on: FCF is
  **not forecast anywhere**, and balance-sheet lines are never forecast independently.
- **Only revenue is forecast; the rest of the statements are derived.** The advice to
  "forecast the drivers and let the margin fall out" was in the README for weeks with
  nothing implementing it. `fpa/forecast/statements.py` does: cost ratios × forecast
  revenue → expenses, `operating_income` as the residual, balance sheet from a
  retained-earnings roll plus working-capital days, cash as the plug.
  `compare_derived_vs_direct` scores four approaches on identical folds — **derived(last)
  1.037, derived(mean) 1.240, direct 1.308, out-of-sample naive 1.417.** Derivation beats
  extrapolation by 21% and beats what naive actually achieves by 27%. Two caveats stated
  rather than buried: 1.037 is still above 1.0, because MASE's denominator is the *in-sample*
  naive error and that is a different comparison from the out-of-sample 1.417; and three
  approaches win at least one of four folds, so the ranking is unstable.
- **The margin assumption was the bottleneck, not the revenue forecast.** Measured by giving
  the model perfect foresight on revenue: fold 3 improves only 1.519 → 1.505. Revenue scores
  MASE 0.155–0.266 on three of four folds. The margin swings ±5 points between adjacent
  quarters while trending up, so a trailing four-quarter average lags it — fold 3 assumed
  23.7% against an actual 29.4%. Ratios now carry forward as the **most recent quarter**
  (`RATIO_METHOD = "last"`); a fitted slope was tested and rejected as overfitting
  (`drift(8)` 1.176, `drift(12)` 1.397). EBIT is ~30% of revenue, so a 5.7-point margin miss
  is a ~20% EBIT error with revenue exactly right.
- **A cash plug makes the identity untestable, so the plug needs a diagnostic.**
  `plug_plausibility` rebuilds the change in cash from net income, non-cash charges, capex
  and buybacks and reports the gap. It earned its place immediately: the first version of
  `forecast_balance_sheet` read `quarterly` rather than the balance-sheet frame and guarded
  each line with `if column in history.columns`, so `treasury_stock` — a **−$28B** derived
  residual absent from `quarterly` — was silently skipped. Equity compounded with no
  contra-equity and the plug absorbed it: **$84B of forecast cash against an actual $9B.**
  Fourth instance of the bug class. Lines are now required; a missing one raises.
- **Two backtests, and the honest one leads.** The monthly ledger is disaggregated by
  this project, so a model scores 0.632 on it and only 0.936 on filed quarters. The gap
  is the artifact, measured rather than assumed.
- **The posterior is pinned like the data vintage; the app never samples.** Fitting at
  read time was wrong twice — it cannot run on the hosted deploy (NumPyro and JAX are
  deliberately out of `requirements.txt`) and locally it took minutes behind a button
  labelled "~15s". `python -m fpa.forecast.posterior` fits all nine leaves offline and
  writes `data/posterior_<ticker>.<vintage>.parquet`; the app forward-simulates in
  NumPy in milliseconds. Only **terminal** level and trend are stored, plus the twelve
  seasonal offsets and two scales — everything `simulate_from_state` reads. The latent
  path is discarded, which is the difference between ~1 MB and ~30 MB.
- **`simulate_from_state` exists so the cache is not a second implementation.** The
  live path and the stored path run the same loop, and a test asserts they produce
  identical arrays. Two copies would be free to drift, and then the calibration report
  would no longer describe what the app draws.
- **Every stored series carries a digest of the numbers it was fitted to.** Caching
  model output is the same trap as everything else here: run `--refresh`, the filings
  move, the draws do not, and the fan looks entirely normal. `stale_series` recomputes
  the digest from the live ledger on every read and the page **refuses to plot** on a
  mismatch. The digest covers the period index too — the same values starting a month
  later imply different seasonal offsets.
- **The app's dependency guard could not fire.** `2_Forecast.py` wrapped
  `from fpa.forecast.bayes import forecast_intervals` in `except ImportError`. NumPyro
  is imported lazily *inside* `fit`, so that import always succeeds and the guard was
  dead code — the `ModuleNotFoundError` arrived later, at call time, outside the `try`,
  and reached the public app as a raw traceback. Serving pinned draws removes the
  failure mode rather than handling it.
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
  are *flagged*, but the aggregate lied: measured per fold, **17 of 27 folds converge** and
  every series has at least one clean fold. The shortest window (42 months) has the **best**
  pass rate at 7/9, against 4/9 at 54 and 6/9 at 66 — so "short series identify the scales
  poorly", published here for weeks, is contradicted by the data. Four of the ten failures
  miss on R-hat 1.011 against a 1.010 ceiling with ESS 545–1,106, which is a threshold
  artifact, not a failed sampler. The wrong diagnosis survived because `backtest_intervals`
  reported only `max(rhat)` across folds; it now records each fold. Two of the three series that beat
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
- **The rejection is demonstrated, not asserted — and it fires at `create`.**
  `--prove-rejection` perturbs one balance-sheet line by $1.00 and Odoo refuses with
  `Fault 2: 'The entry is not balanced.'` **at creation**, so the unbalanced entry never
  becomes a draft and never exists in the database. This spent most of the build as a
  docstring saying Odoo *would* reject — a "would" in a repo insisting on measurements.
  Two caveats owned rather than discovered: the check is *corroborating* (Python's
  `balance_sheet_balances` already proves `A = L + E`, so this is an independent second
  opinion), and `seed_balance_sheet` pre-absorbs sub-dollar cent rounding, so the entry
  arrives balanced to the cent.
- **Odoo validates the filed balance sheet only.** The allocation journal settles to
  `990000` and a plug always balances. Budget lines are not validated at all —
  `crossovered.budget.lines` has no debit, credit or balance field, so the 924 lines
  across 8 budgets are records Odoo would accept at any value. The defensible sentence:
  *"Odoo validates the filed balance sheet, because it rejects an entry that does not
  balance and that entry has no plug. It does not validate the budget or the
  allocations."*
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

- **A silent degradation is the same defect as a silent zero.** `ets` needs `2*period+1`
  observations and falls back to `drift_seasonal` below that rather than raising — correct,
  so one short leaf cannot take down a run — but it did so **silently** for the whole build.
  Measured: **11 of 66 calls, 16.7%** on the monthly backtest, all `too_short`, because fold 1
  trains on 24 months and ETS needs 25. The filed-quarterly backtest degrades **0 times**,
  which is why `0.936` is a real ETS score and the contaminated number is the one already
  disqualified. Now counted by reason, carried on `BacktestResult`, and published in the
  validation report next to the score.
- **`run_backtest` was computing every forecast twice.** Once for the detail rows, once for
  the summary. That doubled the work, required two passes to agree, and made the fallback
  counter report every degradation twice — 22 where the truth was 11. Now computed once and
  both outputs derived from it; every score is unchanged, which is what makes it a refactor.

- **The regional split is the one genuine driver decomposition, and it is built.** UCAN /
  EMEA / LATAM / APAC are filed, carry accession numbers, and sum to the streaming line.
  Bottom-up **0.311** vs direct **0.378** vs naive **1.244** — decomposition wins by 17.8% and
  is comfortably below 1.0, unlike the operating-income case. It was mislabelled a roadmap
  item because four archives gave only *annual* data (5 observations, unbacktestable); the
  quarterly facts are in the 10-Qs, which needed **24 archives and 2.4 GB**.
- **The Q4 gap exists in the segment data too, in a different SEC product.** 10-Ks report
  segments annually, so no year has a Q4 quarterly fact — the identical problem
  `_derive_q4` solves for the income statement, met again in the Financial Statement Data
  Sets and closed the identical way under the identical refusal rule. 7 years derived; 4 of
  them (2019–2022) predate the filer tagging a streaming total, so the footing check cannot
  run and they are derived **and flagged**.
- **The checker was measured against real prose, and the result is weaker than it looks.**
  12 drafts, 247 numerals, **100% parse coverage, 0 unparsed** — but the census shows every
  numeral was digit form. The hard case (*"roughly six hundred million"*) never occurred, so
  it is **untested rather than passed**. And the reason is a design decision: the JSON schema
  and length cap push the model toward compact figures, so the schema carries part of the
  load the regex appears to carry. Accept/reject accuracy on real drafts stays open, because
  deriving ground truth from the checker's own rule is the tautology `evaluation.py` exists
  to prevent.

## The bug class this codebase keeps finding

**Nine separate defects, all the same shape: something absent produces a plausible number
instead of an error.** The list keeps growing, which is the point of keeping it — every
addition was found by an instrument added for a different reason.

*A missing input, silently treated as nothing:*

1. `_derive_q4` summed Q1–Q3 with `.sum()`. Pre-tax income is only tagged as a quarter
   from 2020-Q3, so 2020-Q4 came out as `FY − Q3` — **$1.83B too high**, with two
   quarters of the year buried inside it.
2. `apply_derivations` summed balance-sheet parts the same way, so `other_current_assets`
   reported 26 of 26 quarters populated when 10 of them rested on an undeclared
   assumption about short-term investments.
3. The original ingest hardcoded `units["USD"]`, which for a per-share tag returns `[]`
   rather than raising.
4. `forecast_balance_sheet` guarded each line with `if column in history.columns` while
   reading the wrong frame, so `treasury_stock` — a **−$28B** residual — vanished. Equity
   compounded with no contra-equity and the cash plug absorbed it: **$84B forecast against
   an actual $9B.** Only visible because the plug produced an absurd number rather than a
   merely wrong one.
5. `backtest_frames` filtered the filed-series list the same way, so a renamed tag would have
   scored five series instead of six and changed the **headline MASE** with no signal. Found
   by the sweep below, not by a failure — nothing was missing on the day.

*A count derived by subtraction instead of by asking:*

6. `_auditability` computed `skipped = results − verified`, labelling two WARN failures as
   skips.
7. `gate_banner` did the same thing and printed *"2 skipped — a skip is not a pass"* on a
   run that skipped nothing. Worse than (6): the skip/pass distinction was added
   deliberately after a cold-start rehearsal, and the display of it was wrong.

*A degradation or a guard that reports nothing:*

8. `ets` fell back to `drift_seasonal` silently — **16.7%** of monthly backtest calls — so a
   score labelled `ets` was partly a different model.
9. The app guarded `from fpa.forecast.bayes import forecast_intervals` with
   `except ImportError`. NumPyro is imported lazily *inside* `fit`, so that import always
   succeeds: the guard was unreachable and the real error surfaced later, outside the `try`,
   as a traceback on the public app.

**None of the nine raises.** 1–5 are caught by controls that test an *identity* rather than
a value — which is the whole argument for having them. 6–9 were caught by reading published
output as a stranger would: on the deployed app, in a rendered README, in a report table.
Both instruments matter, and neither substitutes for the other.

### The standing sweep

All three mechanisms are **greppable**, which turns "be careful" into something a person can
actually run. Do this before shipping anything that produces a number:

```bash
# A. membership guards that silently drop what they cannot find
grep -rn "in .*\.columns" --include=*.py fpa/ app/

# B. a count derived by subtraction rather than by asking
grep -rnE "len\([^)]+\)\s*-\s*len\(" --include=*.py fpa/ app/

# C. a fallback or except that returns without recording anything
grep -rn -A3 "except Exception" --include=*.py fpa/ app/
```

**A is the one that keeps producing defects**, because it is a legitimate pattern half the
time. The rule that separates the two cases: *if the caller would be wrong without the
column, require it and raise; if the column is genuinely optional, the absence has to be
reported somewhere a reader sees.* `if x in frame.columns` followed by silently doing less is
never acceptable.

Running this sweep after the eighth defect found the **ninth** — item 5 above. Nothing was
missing on the day, which is precisely how this class survives a code review.

For B and C the rule is absolute and the sweep should stay empty: **count by asking**
(`sum(1 for x in xs if predicate(x))`), and **every fallback branch increments a counter or
logs at WARN** — see `fpa.forecast.models._fell_back`.

## Grounding (textbook KB)

| Source | What it changed |
|---|---|
| Phillips, *Pricing & Revenue Optimization* 2e, p.94 | MAPE → MASE + weighted MAPE |
| Nielsen, *Practical Time Series Analysis*, p.253 | Named the hierarchy; bottom-up reconciliation |
| McElreath, *Statistical Rethinking* 2e, p.223 | Intervals need sharpness, not just coverage |
| Huyen, *AI Engineering*, pp.219–225 | Groundedness as factual-consistency checking; bounded output length |
| López de Prado, *AFML* ch.7/12 | Rolling origin with a horizon gap — purge/embargo targets *overlapping labels*, which monthly FP&A periods do not have. Use the simpler correct thing and say why. |
