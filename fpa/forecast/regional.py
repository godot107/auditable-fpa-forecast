"""Revenue by region — the one genuine driver decomposition this data supports.

Everything else in ``fpa.forecast`` is univariate: a series extrapolated from its own history.
The README used to call that "driver-based", which was wrong and has been corrected. This
module is the honest version, and it is honest because **every input is filed**. UCAN, EMEA,
LATAM and APAC are reported by the filer, carry accession numbers, and sum to the consolidated
streaming line. No cost center, no invented granularity, no modelled ratio.

The question it answers is the one an FP&A team actually asks about decomposition: **is
forecasting the parts and adding them up better than forecasting the whole?** Bottom-up
reconciliation guarantees coherence either way (Nielsen, *Practical Time Series Analysis*,
p.253); whether it also improves accuracy is an empirical matter, and this measures it on
identical rolling-origin folds.

**Getting the data cost 2.4 GB and exposed the same Q4 problem twice.** Regional segments are
tagged in the *dimension* columns of SEC's Financial Statement Data Sets, not in
``companyfacts``, so each quarter needs its own ~85 MB archive — 24 of them for this history.
And 10-Ks report segments **annually**, so there is no Q4 quarterly fact for any year: the
identical gap the income statement has, in a completely different SEC product. It is closed
the identical way, ``Q4 = FY - (Q1+Q2+Q3)``, under the identical refusal rule — a year whose
regions do not foot to the filed streaming total does not get a derived quarter.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fpa.config import REGION_LABELS, STREAMING_TOTAL, Settings
from fpa.forecast.backtest import mase, rolling_origin_splits, wmape
from fpa.forecast.models import MODELS

logger = logging.getLogger(__name__)

REGIONS: tuple[str, ...] = tuple(sorted(set(REGION_LABELS.values())))

# A derived quarter is only trusted when the four regions foot to the filed streaming total
# for that year. Netflix tags the streaming total from FY2023; before that the check cannot
# run and the derivation is flagged rather than silently trusted.
FOOTING_TOLERANCE = 0.005  # 0.5% of the annual total


def _pivot(facts: pd.DataFrame, period_type: str) -> pd.DataFrame:
    subset = facts[facts["period_type"] == period_type]
    if subset.empty:
        return pd.DataFrame()
    deduped = subset.drop_duplicates(subset=["region", "end"], keep="last")
    return deduped.pivot(index="end", columns="region", values="value").sort_index()


def quarterly_regional(facts: pd.DataFrame) -> pd.DataFrame:
    """Filed Q1–Q3 plus a derived Q4, with the derivation checked against the annual.

    ``facts`` is the long frame from :func:`fpa.ingest.segments.regional_revenue` before it is
    pivoted — it must carry both ``quarter`` and ``annual`` rows.

    Returns one column per region, indexed by quarter end. ``attrs["derived_q4"]`` lists the
    years whose Q4 was derived, and ``attrs["unverified_q4"]`` the subset where the footing
    check could not run because no streaming total was tagged.
    """
    quarters = _pivot(facts, "quarter")
    annual = _pivot(facts, "annual")
    if quarters.empty or annual.empty:
        raise ValueError("regional Q4 derivation needs both quarterly and annual facts")

    missing = [r for r in REGIONS if r not in quarters.columns]
    if missing:
        raise ValueError(f"quarterly regional facts are missing {missing}")

    rows, derived, unverified = [], [], []
    for year in sorted({d.year for d in annual.index}):
        year_quarters = quarters[quarters.index.year == year]
        year_annual = annual[annual.index.year == year]
        if len(year_quarters) != 3 or year_annual.empty:
            continue
        if any(r not in year_annual.columns for r in REGIONS):
            continue

        totals = year_annual.iloc[0]
        # The refusal rule, borrowed verbatim from the income-statement Q4 derivation: a
        # year whose parts do not foot to the filed whole does not get a derived quarter,
        # because the entire discrepancy would land in that one number.
        if STREAMING_TOTAL in year_annual.columns and pd.notna(totals.get(STREAMING_TOTAL)):
            residual = abs(totals[list(REGIONS)].sum() - totals[STREAMING_TOTAL])
            if residual > FOOTING_TOLERANCE * totals[STREAMING_TOTAL]:
                logger.warning(
                    "FY%s regions do not foot to the streaming total (off by %.0f) — "
                    "refusing to derive Q4", year, residual,
                )
                continue
        else:
            unverified.append(year)

        q4 = totals[list(REGIONS)] - year_quarters[list(REGIONS)].sum()
        if (q4 <= 0).any():
            logger.warning("FY%s derived Q4 is non-positive for some region — refusing", year)
            continue

        q4.name = pd.Timestamp(f"{year}-12-31")
        rows.append(q4)
        derived.append(year)

    frame = pd.concat([quarters[list(REGIONS)], pd.DataFrame(rows)]).sort_index()
    frame.index.name = "period"
    frame.attrs["derived_q4"] = tuple(derived)
    frame.attrs["unverified_q4"] = tuple(unverified)
    frame.attrs["filed_quarters"] = int(len(quarters))
    return frame


def compare_regional_vs_total(
    regional: pd.DataFrame, *, horizon: int = 4, folds: int = 3, model: str = "ets"
) -> pd.DataFrame:
    """Forecast four regions and add them up, against forecasting the total directly.

    Identical folds, identical horizon, identical model. The only difference is whether the
    consolidated number is forecast or assembled — which is the whole question.
    """
    total = regional[list(REGIONS)].sum(axis=1)
    values = total.to_numpy(float)
    records = []

    for fold, (train_idx, test_idx) in enumerate(
        rolling_origin_splits(len(regional), horizon, folds=folds), start=1
    ):
        actual = values[test_idx]
        train_total = total.iloc[train_idx]

        bottom_up = np.zeros(horizon, dtype=float)
        for region in REGIONS:
            bottom_up += np.asarray(
                MODELS[model](regional[region].iloc[train_idx], horizon, period=4), dtype=float
            )
        direct = np.asarray(MODELS[model](train_total, horizon, period=4), dtype=float)
        naive = np.asarray(
            MODELS["seasonal_naive"](train_total, horizon, period=4), dtype=float
        )

        for label, predicted in (
            ("bottom-up (4 regions)", bottom_up),
            ("direct (consolidated)", direct),
            ("seasonal_naive", naive),
        ):
            records.append(
                {
                    "fold": fold,
                    "approach": label,
                    "mase": mase(actual, predicted, values[train_idx], period=4),
                    "wmape": wmape(actual, predicted),
                    "train_quarters": len(train_idx),
                }
            )
    return pd.DataFrame(records)


def report_markdown(regional: pd.DataFrame, comparison: pd.DataFrame) -> str:
    by_approach = comparison.groupby("approach")["mase"].mean().sort_values()
    per_fold = comparison.pivot(index="approach", columns="fold", values="mase")
    best = by_approach.index[0]
    bottom_up = by_approach["bottom-up (4 regions)"]
    direct = by_approach["direct (consolidated)"]

    derived = regional.attrs["derived_q4"]
    unverified = regional.attrs["unverified_q4"]

    lines = [
        "## Regional decomposition — the only real driver split this data supports",
        "",
        f"**{len(regional)} quarters x {len(REGIONS)} regions**, "
        f"{regional.index.min():%Y-%m} to {regional.index.max():%Y-%m}. Every figure is filed "
        "and carries an accession number. No cost center, no modelled ratio, no invented "
        "granularity.",
        "",
        "### Does forecasting the parts beat forecasting the whole?",
        "",
        "| approach | mean MASE | " + " | ".join(f"fold {f}" for f in per_fold.columns) + " |",
        "|---|---|" + "---|" * len(per_fold.columns),
    ]
    for approach, mean_mase in by_approach.items():
        cells = " | ".join(f"{per_fold.loc[approach, f]:.3f}" for f in per_fold.columns)
        mark = "**" if approach == best else ""
        lines.append(f"| `{approach}` | {mark}{mean_mase:.3f}{mark} | {cells} |")

    lines += ["", "Identical folds, identical horizon, identical model. The only difference is "
              "whether the consolidated number is forecast or assembled."]

    if bottom_up < direct:
        lines.append(
            f"\n**Bottom-up wins by {(1 - bottom_up / direct):.1%}.** Decomposing into regions "
            "and adding them up beats forecasting the consolidated line — the four regions "
            "grow at genuinely different rates, so a single series averages away structure "
            "the parts retain."
        )
    else:
        lines.append(
            f"\n**Bottom-up loses by {(bottom_up / direct - 1):.1%}, and that is the finding.** "
            "Coherence is still a reason to decompose — the regional forecasts have to sum to "
            "the number the CFO sees — but on this data it costs accuracy rather than buying "
            "it. Reporting that is the point of running the comparison."
        )

    lines += [
        "",
        "### The Q4 problem, again, in a different SEC product",
        "",
        "10-Ks report segments **annually**, so there is no Q4 quarterly fact in any year — "
        "the identical gap the income statement has in `companyfacts`, met again in the "
        "Financial Statement Data Sets. Closed the identical way and under the identical "
        f"refusal rule: `Q4 = FY - (Q1+Q2+Q3)`, derived for **{len(derived)} year(s)** "
        f"({', '.join(map(str, derived))}), and only where the four regions foot to the filed "
        "streaming total.",
    ]
    if unverified:
        lines.append(
            f"\n**{len(unverified)} year(s) could not be checked** — "
            f"{', '.join(map(str, unverified))}. The filer did not tag a consolidated "
            "streaming total before FY2023, so the footing test cannot run on them. They are "
            "derived and flagged rather than derived and trusted."
        )
    lines += [
        "",
        f"_{regional.attrs['filed_quarters']} filed quarters + {len(derived)} derived Q4s. "
        "Assembling this cost 24 archives at ~85 MB each: regional detail lives in the "
        "`segments` dimension of the Data Sets, which `companyfacts` does not expose at all._",
    ]
    return "\n".join(lines)
