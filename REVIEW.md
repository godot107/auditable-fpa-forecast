# Methodology review against the textbook KB

A pass over every methodological choice in this project, checked against the local
textbook library rather than against my own recollection. The point of the exercise is
to find what is *missing*, so the confirmations are listed briefly and the gaps in full.

Searched: McElreath *Statistical Rethinking* 2e; Martin, Kumar & Lao *Bayesian Modeling
and Computation in Python*; López de Prado *AFML* and *Machine Learning for Asset
Managers*; Nielsen *Practical Time Series Analysis*; Phillips *Pricing and Revenue
Optimization* 2e; Huyen *AI Engineering*; Subramanyam *Financial Statement Analysis*;
Jansen *ML for Algorithmic Trading* 2e.

---

## Confirmed — choices the KB supports

| Choice | Source | Verdict |
|---|---|---|
| MASE over MAPE at cost-center grain | Phillips 2e, p.94 | Holds. MAPE is dominated by small denominators. |
| Bottom-up hierarchical reconciliation | Nielsen, p.253 | Holds. Named correctly; the trade-off against top-down is stated in the README. |
| Coverage **and** sharpness for intervals | McElreath 2e, p.223 | Holds, and is now the structure of the whole calibration report. |
| Groundedness as reference-based factual consistency | Huyen pp.219–225 | Holds. The reference here is *computed*, which is stronger than retrieved. |
| Rolling origin with a horizon gap, not purge/embargo | López de Prado *AFML* ch.7 | Holds. Purge/embargo targets overlapping labels; monthly FP&A periods do not overlap. |
| Local linear trend as a linear Gaussian state-space model | Martin et al., p.225 | Holds as a model family — but see Gap 1 for what the specification got wrong. |
| Segment disclosures as forecasting inputs | Subramanyam, p.506 | Supported: segment data is documented as useful for forecasting future profitability. |

---

## Gap 1 — the interval layer was not converged, and only one diagnostic was watching

**Severity: high. This invalidated published numbers.**

The fit reported divergent transitions and nothing else. Divergences are one of three
standard diagnostics, and the other two were absent:

- **R-hat** (Gelman-Rubin) compares between-chain to within-chain variance.
  *It cannot be computed from a single chain* — and the sampler was running one.
- **Effective sample size** measures how many independent draws the correlated ones are
  worth. A chain can be perfectly converged and still carry almost no information.

McElreath 2e p.309 and Martin et al. §2.4.2 both treat the three as a set. Jansen 2e
p.343 states the R-hat definition that makes the single-chain problem obvious. Reporting
divergences alone was reporting the diagnostic that happened to be available.

**What it hid.** Adding three chains and computing the missing diagnostics:

```
R-hat 1.93     ESS 3     (against thresholds of 1.01 and 400)
```

Not a vectorization artifact — `chain_method="sequential"` reproduced it. The worst site
was `sigma_obs`, R-hat 1.55, with chain means spanning 0.012 to 0.025. A **2× spread
across chains** in a parameter that sets the width of every interval.

**Root cause 1: a non-identified model.** The specification had both a level shock and
observation noise:

```
level_t = level_{t-1} + trend_{t-1} + level_shock     <- sigma_level
log y_t = level_t + seasonal[month] + noise           <- sigma_obs
```

Both explain month-to-month variation and the data cannot separate them, so each chain
settles somewhere different along the ridge. This is the classic local-level
identifiability problem, and the single-chain run had no way to report it.

**Root cause 2: I had tuned the sampler on the one diagnostic I was watching.**

`target_accept_prob` had been raised to 0.99 in an earlier session, documented as a
measured improvement, and published in the README, `CLAUDE.md` and the blog post — on the
grounds that it cut divergences from ~3% to ~0.4%. Measured against all three diagnostics:

| `target_accept` | divergences | R-hat | ESS |
|---|---|---|---|
| 0.80 | 85–142 | 1.010–1.028 | 141–551 |
| 0.90 | 32–189 | 1.007–1.037 | 121–387 |
| 0.95 | 9–87 | 1.006–1.069 | 85–452 |
| **0.99** | **0–38** | **999–1804** | **2** |

At 0.99 the step size collapses and the chains stop moving. **Nothing diverges because
nothing is being integrated through.** The "improvement" was the signature of chains
dying.

This is the single most useful thing the review turned up, and not because of the
parameter. It is a clean instance of the failure this whole project argues against —
optimizing the metric that happens to be instrumented while the uninstrumented ones
degrade — committed by the project itself, in the one module where I was most confident I
had been rigorous.

**Fix.** Three changes: drop the redundant level shock (identified *smooth trend* model,
the standard component set in Martin et al. pp.193, 227); `target_accept_prob` back to
0.90; four vectorized chains with R-hat and ESS computed on every fit. R-hat 1.93 → ≤1.03.
The calibration report now prints both diagnostics per series and flags any fit that did
not converge, **before** the calibration conclusions rather than in a footnote after them.

**What the fix revealed.** Re-running the full calibration on the corrected model: **8 of
the 9 backtest fits still fail** R-hat ≤ 1.01 or ESS ≥ 400. The full-data fits mostly
converge; the backtest ones often do not, because rolling origin trains on 42–66 months
rather than 78, and a shorter series identifies the trend and observation scales less well.

Two of the three series that beat the naive benchmark are the worst-converged rows in the
table (R-hat 1.99 / ESS 3, and 1.358 / ESS 10). The one cleanly converged fit *loses* to the
benchmark. So the honest status of the interval layer is **provisional**, and the README,
`CLAUDE.md`, the app and the blog post all now say so rather than quoting the three winning
rows. The remaining work is more draws for short training windows, or a simpler model below
~60 observations.

---

## Gap 2 — priors were asserted to be weakly informative, never checked

**Severity: medium.**

The scale priors carried a comment claiming `HalfNormal(0.05)` is "a ~5% month-on-month
move: loose enough not to drive the answer." That is an assertion about what a prior
*implies*, made without simulating it.

McElreath 2e p.114 is unusually blunt about this:

> To figure out what this prior implies, we have to simulate the prior predictive
> distribution. **There is no other reliable way to understand.**

The irony is not lost: this project's entire thesis is that a comment claiming an
identity holds is worth less than a control proving it, and the priors were the one place
I left a claim in a comment.

**What it hid.** Simulating the prior predictive, the priors implied a 90% band on
**annual** growth of **0.16× to 6.4×**, with draws reaching **127×**. A prior saying a
cost center might grow 127-fold in a year is not weakly informative; it is nearly
uninformative on the scale anyone cares about. The culprit was not the scale priors I had
commented on but `trend0 ~ Normal(0, 0.10)` — a 10% log-slope *per month*, which compounds
to something absurd over a year and reads as innocuous in isolation.

**Fix.** `trend0 ~ Normal(0, 0.02)`, giving a 90% band of 0.58× to 1.68×. A
`prior_predictive` function is now part of the module and a test asserts the implied
annual growth stays inside a plausible band — so the claim is enforced rather than
narrated. The tighter prior also improved sampling, which is not a coincidence.

---

## Gap 3 — the headline MASE is a selected maximum

**Severity: medium — a reporting issue, not a defect.**

Three models are scored on the same rolling-origin backtest and the best is reported:
`ets` at 0.936, against `drift_seasonal` 0.943 and `seasonal_naive` 1.150. Selecting the
winner from the evaluation you then quote makes that figure optimistically biased.

López de Prado, *AFML* p.180:

> The purpose of a backtest is to discard bad models, not to improve them. Adjusting your
> model based on the backtest results is a waste of time… and it's dangerous.

Two things in the project's favour, and one against:

- **In favour:** no model was *tuned* on the backtest. The ETS specification was fixed a
  priori, and `window_start` was set by an articulation control — a data-integrity test,
  not an accuracy one. This is the sin López de Prado actually warns about, and on the
  point-forecast side the project does not commit it.
  (An earlier draft of this review cited `target_accept_prob` here as a further example of
  tuning on a diagnostic rather than a performance metric. Gap 1 shows that tuning was
  itself harmful — so it belongs there as a failure, not here as a defence.)
- **In favour:** with three candidates the selection bias is small. This is not the
  thousands-of-simulations case that motivates the deflated Sharpe ratio.
- **Against:** it is still a selected maximum, and the README presents 0.936 as though it
  were an unbiased out-of-sample estimate.

**Fix — applied.** Stated in the README's evaluation section: 0.936 is the optimistic end
of a narrow range, not a point estimate. All three models' scores were already published
side by side, so the correction is a paragraph rather than a restructure.

---

## Gap 4 — the groundedness checker has no measured error rate

**Fix — synthetic half applied. Real-draft half still open.**

**Severity: medium, and the most expensive to close properly.**

The checker is validated by tests that feed it specific fabrications and assert rejection.
That proves it catches *those*. It does not measure:

- **False rejection rate** — how often a perfectly valid draft is rejected for a figure it
  legitimately derived. Each such rejection is a real cost, and two of them have already
  been found and fixed by hand (`2,419.3M` parsing as two numbers; magnitude-vs-signed
  matching).
- **False acceptance rate** — the dangerous direction, and the one the regex bug produced:
  `$999.9M.` at the end of a sentence was never matched at all, so a fabricated figure
  passed by never being checked.

Huyen frames groundedness as factual-consistency checking; the LangChain material
classifies this as a **reference-based heuristic evaluator**. Neither treats an evaluator
as exempt from evaluation.

**A third failure mode, and the one that actually bit.** ``check()`` reports ``checked``
— numerals the regex *matched* — so a verdict only exists for numerals the parser saw:

| Mode | What happens | Visible in output? |
|---|---|---|
| False rejection | grounded figure marked ungrounded | yes |
| False acceptance | fabricated figure judged grounded | yes |
| **Parse miss** | numeral never matched at all | **no** |

The ``$999.9M.`` bug was the third kind. Not a wrong verdict — *silence*. Any accuracy
metric computed over matched numerals would have scored that checker 100% while it was
blind, so silence needs its own metric.

**Fix — applied.** ``fpa/narrative/evaluation.py``. Labels are generated by construction
rather than annotation: positives are composed from payload values across ten surface
forms (sentence-final, parenthesised, comma-grouped, `bn` lowercase…), negatives are
grounded drafts with exactly one numeral mutated — digit perturbation, magnitude shift,
plausible-but-uncomputed derivation, invented round number. Parse misses are found by
running a deliberately **over-inclusive** second parser and comparing character spans; any
span the strict parser never covered is a numeral with no verdict at all, and detecting it
needs no labels.

Measured over 364 cases: **0.00% false acceptance, 0.00% false rejection, 100% parse
coverage.** Committed to ``reports/groundedness_evaluation.md`` via ``--groundedness``,
which exits non-zero if a fabrication is accepted or a numeral goes uninspected.

**A mistake worth recording, because it is the one this project exists to prevent.** The
first version labelled cases by running the checker's *own* matching rule over them. That
made positives pass and negatives fail by construction: both rates were tautologies, and it
reported a perfect score while measuring nothing. Labels now come from geometry with the
checker's tolerance never consulted —

```
|----positive----|      ambiguous      |----negative----|
0            2e-4    MATCH_RTOL 2e-3  2e-2          infinity
```

— so the checker is free to disagree with the corpus. Ambiguous cases are simply not
generated, because their correct label genuinely depends on which tolerance you pick.

**And the evaluation is itself tested.** Four tests break the checker on purpose — reverting
the sentence-final regex bug, widening the tolerance, tightening it, dropping scale
suffixes — and assert the evaluation goes red. Otherwise "0% false acceptance" is
indistinguishable from an instrument that cannot see.

**Still open: the real-draft half.** Synthetic phrasing is not the same distribution a
model actually writes. Closing that needs ~50 genuine drafts from the `claudecode`
provider, each numeral adjudicated once by hand and frozen as a fixture. It is the only
part that tests generalisation — a model writing "roughly six hundred million" in words
would defeat every regex here and nothing in the synthetic corpus would notice.

---

## Gap 5 — segment revenue inherits a caveat the README does not state

**Severity: low.**

Subramanyam ch.8 appendix (pp.486–487) sets out the standard warnings on segment
disclosures: segments carry different profitability, risk and growth profiles, and
inter-segment allocations of common cost are management judgements rather than measured
facts.

The regional revenue ingested here is a *revenue* disclosure, so the common-cost
allocation problem does not bite directly. But the project now displays regional figures
next to a cost-center ledger that is itself modeled, and nothing on screen distinguishes
"filed regional revenue" from "our allocation."

**Fix — applied.** The README's segment section now states that the regional figures are
`REAL` and revenue-only, that no cost is allocated to region, and that no regional margin
or contribution is claimed — with the Subramanyam reasoning for why segment *profitability*
is the treacherous part. Region and the cost-center hierarchy stay separate dimensions and
are never crossed.

---

## What did not change

- **Bottom-up rather than optimal (MinT) reconciliation.** Nielsen p.253 describes
  bottom-up, top-down and middle-out; the KB has no MinT treatment, and MinT would need a
  covariance estimate across nine leaves from 78 observations. Bottom-up guarantees the
  cost-center forecasts sum to the number the CFO sees, which is the property FP&A needs.
  Staying put, with the trade-off already stated.
- **Log-scale back-transformation.** Modelling `log y` and exponentiating would bias a
  *mean*, but quantiles are invariant under a monotone transform and only quantiles are
  reported. No defect.
- **The two-backtest honesty framing.** Nothing in the KB contradicts it and it remains
  the strongest single piece of methodological discipline in the project.
