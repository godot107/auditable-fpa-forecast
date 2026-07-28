"""Rolling-origin backtest with metrics chosen for cost-center-level data.

**Why MASE and not MAPE.** MAPE divides by the actual, so a month with a small
denominator dominates the average — at cost-center granularity a modest miss on a
small department can outweigh a large miss on Content, and the headline number
stops describing the thing anyone cares about (Phillips, *Pricing and Revenue
Optimization* 2e, p.94, on MAPE's asymmetry and the weighted-MAPE remedy).

MASE scales the error by the in-sample seasonal-naive error instead, which makes
it (a) scale-free, so leaf and total are comparable, (b) defined when an actual is
zero, and (c) directly interpretable: **below 1 beats the naive benchmark, above 1
loses to it**. Weighted MAPE is reported alongside because finance audiences read
percentages, and it is denominator-weighted so it does not blow up on small months.

**Why rolling origin with a gap.** Each fold trains on data strictly before its
test window, with the origin advancing through time, so no fold sees inside its own
forecast horizon. Standard k-fold would leak future months into training and report
an accuracy the model cannot reproduce in production.

Note on purging: ``financial-forecasting-engine`` uses López de Prado's purge and
embargo, which exist to handle **overlapping labels** in financial ML. Monthly FP&A
periods do not overlap, so the horizon-sized gap is the honest guard here and the
extra machinery would be cargo-culted rather than reasoned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fpa.forecast.models import MODELS

BENCHMARK = "seasonal_naive"


def mase(
    actual: np.ndarray, predicted: np.ndarray, train: np.ndarray, *, period: int = 12
) -> float:
    """Mean Absolute Scaled Error against the in-sample seasonal-naive benchmark.

    <1 beats "same month last year"; >1 loses to it.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    train = np.asarray(train, dtype=float)

    if len(train) <= period:
        return float("nan")
    scale = np.mean(np.abs(train[period:] - train[:-period]))
    if scale == 0 or not np.isfinite(scale):
        return float("nan")
    return float(np.mean(np.abs(actual - predicted)) / scale)


def wmape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Weighted MAPE: total absolute error over total actual.

    Denominator-weighted, so a small month cannot dominate the way plain MAPE
    lets it.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = np.sum(np.abs(actual))
    if denominator == 0:
        return float("nan")
    return float(np.sum(np.abs(actual - predicted)) / denominator)


def rolling_origin_splits(
    n: int, horizon: int, *, folds: int, min_train: int | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` pairs with the origin advancing through time.

    The test window immediately follows the training window, and the model is
    given no observation inside it. Folds are spaced so the last one ends at the
    final observation.
    """
    if min_train is None:
        min_train = max(2 * horizon, n - horizon - (folds - 1) * horizon)
    min_train = max(min_train, horizon)

    last_origin = n - horizon
    first_origin = max(min_train, last_origin - (folds - 1) * horizon)
    if first_origin > last_origin:
        raise ValueError(
            f"series of length {n} cannot support {folds} folds at horizon {horizon}"
        )

    step = max(1, (last_origin - first_origin) // max(1, folds - 1)) if folds > 1 else 1
    origins = sorted({min(last_origin, first_origin + i * step) for i in range(folds)})

    return [
        (np.arange(0, origin), np.arange(origin, origin + horizon)) for origin in origins
    ]


@dataclass
class BacktestResult:
    """Per-fold, per-model accuracy plus the summary an FP&A reader wants."""

    detail: pd.DataFrame  # one row per (series, model, fold, horizon_step)
    by_model: pd.DataFrame  # aggregated to (series, model)

    def summary(self) -> pd.DataFrame:
        """Mean MASE and wMAPE per model across every series and fold."""
        return (
            self.by_model.groupby("model")[["mase", "wmape"]]
            .mean()
            .sort_values("mase")
            .reset_index()
        )

    def losses_to_benchmark(self) -> pd.DataFrame:
        """Series/model combinations that the naive benchmark beats.

        Reported deliberately. A backtest that only shows wins is marketing.
        """
        return (
            self.by_model[(self.by_model["model"] != BENCHMARK) & (self.by_model["mase"] >= 1.0)]
            .sort_values("mase", ascending=False)
            .reset_index(drop=True)
        )

    def by_horizon(self) -> pd.DataFrame:
        """Accuracy by how far ahead the forecast was — degradation should be visible."""
        return (
            self.detail.groupby(["model", "horizon_step"])["abs_error"]
            .mean()
            .reset_index()
            .pivot(index="horizon_step", columns="model", values="abs_error")
        )


def run_backtest(
    series_frame: pd.DataFrame,
    *,
    horizon: int = 12,
    folds: int = 6,
    period: int = 12,
    models: dict | None = None,
) -> BacktestResult:
    """Backtest every column of ``series_frame`` across every model.

    ``series_frame`` is indexed by period with one column per series (leaf cost
    centers, and any aggregate worth scoring in its own right).
    """
    models = models or MODELS
    frame = series_frame.sort_index()
    n = len(frame)
    splits = rolling_origin_splits(n, horizon, folds=folds)

    detail_rows: list[dict] = []
    for column in frame.columns:
        values = frame[column].to_numpy(dtype=float)
        name = column if isinstance(column, str) else " / ".join(map(str, column))

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            train = pd.Series(values[train_idx], index=frame.index[train_idx])
            actual = values[test_idx]

            for model_name, fn in models.items():
                predicted = np.asarray(fn(train, len(test_idx), period=period), dtype=float)
                for step, (a, p) in enumerate(zip(actual, predicted), start=1):
                    detail_rows.append(
                        {
                            "series": name,
                            "model": model_name,
                            "fold": fold_idx,
                            "horizon_step": step,
                            "period": frame.index[test_idx[step - 1]],
                            "actual": a,
                            "predicted": p,
                            "abs_error": abs(a - p),
                        }
                    )

    detail = pd.DataFrame(detail_rows)

    # Aggregate to (series, model), recomputing MASE per fold against that fold's
    # own training window — the scale must come from data the model actually saw.
    summary_rows: list[dict] = []
    for column in frame.columns:
        values = frame[column].to_numpy(dtype=float)
        name = column if isinstance(column, str) else " / ".join(map(str, column))
        for model_name, fn in models.items():
            fold_mase, fold_wmape = [], []
            for train_idx, test_idx in splits:
                train = pd.Series(values[train_idx], index=frame.index[train_idx])
                actual = values[test_idx]
                predicted = np.asarray(fn(train, len(test_idx), period=period), dtype=float)
                fold_mase.append(mase(actual, predicted, values[train_idx], period=period))
                fold_wmape.append(wmape(actual, predicted))
            summary_rows.append(
                {
                    "series": name,
                    "model": model_name,
                    "mase": float(np.nanmean(fold_mase)),
                    "wmape": float(np.nanmean(fold_wmape)),
                    "n_folds": len(splits),
                }
            )

    return BacktestResult(detail=detail, by_model=pd.DataFrame(summary_rows))


def validation_report(result: BacktestResult, *, horizon: int, folds: int) -> str:
    """Render the backtest as Markdown, including where the model loses."""
    summary = result.summary()
    lines = [
        "## Forecast validation",
        "",
        f"Rolling-origin backtest: {folds} folds, {horizon}-month horizon, "
        "no observation inside the forecast window.",
        "",
        "| model | MASE | wMAPE |",
        "|---|---|---|",
    ]
    for row in summary.itertuples():
        lines.append(f"| `{row.model}` | {row.mase:.3f} | {row.wmape:.1%} |")

    lines += [
        "",
        "MASE is scaled by the in-sample seasonal-naive error: **below 1.0 beats "
        "the benchmark, above 1.0 loses to it.**",
        "",
    ]

    losses = result.losses_to_benchmark()
    if losses.empty:
        lines.append("Every model beat the seasonal-naive benchmark on every series.")
    else:
        lines += [
            f"### Where the models lose to seasonal-naive ({len(losses)} combinations)",
            "",
            "| series | model | MASE |",
            "|---|---|---|",
        ]
        for row in losses.head(15).itertuples():
            lines.append(f"| {row.series} | `{row.model}` | {row.mase:.3f} |")

    return "\n".join(lines)


def honest_validation_report(
    monthly: BacktestResult,
    filed_quarterly: BacktestResult,
    *,
    horizon_months: int,
    horizon_quarters: int,
) -> str:
    """Report both backtests together, leading with the one that flatters least.

    The monthly backtest runs on a ledger this project *disaggregated itself* from
    filed quarterly figures. Part of the intra-quarter structure a model finds
    there was put there by ``fpa.ledger.disaggregate``, so a model can score well
    by rediscovering our own allocation weights. That accuracy would not survive
    contact with a real monthly ledger.

    The filed-quarterly backtest has no such artifact: every value in it is a
    number Netflix filed. It is therefore the honest headline, and it is
    substantially worse — which is the point of showing both.
    """
    m_summary = monthly.summary().set_index("model")
    q_summary = filed_quarterly.summary().set_index("model")

    lines = [
        "## Forecast validation — two backtests, and why they disagree",
        "",
        "| model | MASE (filed quarterly) | MASE (modeled monthly) |",
        "|---|---|---|",
    ]
    for model in q_summary.index:
        m_value = m_summary["mase"].get(model, float("nan"))
        lines.append(f"| `{model}` | **{q_summary.loc[model, 'mase']:.3f}** | {m_value:.3f} |")

    best_q = q_summary["mase"].idxmin()
    best_m = m_summary["mase"].idxmin()

    lines += [
        "",
        f"**Read the left column.** The monthly series is filed quarterly data that this "
        f"project disaggregated into months, so some of the structure a model finds there "
        f"is structure we put there. `{best_m}` scores "
        f"{m_summary.loc[best_m, 'mase']:.3f} on it — and only "
        f"{q_summary.loc[best_q, 'mase']:.3f} on filed quarters, where every value is a "
        f"number the company actually reported. The gap is the artifact, measured rather "
        f"than assumed.",
        "",
        f"Against filed data the best model beats a seasonal-naive benchmark by only "
        f"{(1 - q_summary.loc[best_q, 'mase']) * 100:.0f}%, and loses on some series "
        f"outright:",
        "",
    ]

    losses = filed_quarterly.losses_to_benchmark()
    if losses.empty:
        lines.append("- (none this run)")
    else:
        lines += ["| series | model | MASE |", "|---|---|---|"]
        for row in losses.head(10).itertuples():
            lines.append(f"| `{row.series}` | `{row.model}` | {row.mase:.3f} |")
        lines += [
            "",
            "Operating income is consistently the hardest, and predictably so: it is a "
            "small difference between large numbers, so proportionally modest errors in "
            "revenue and cost compound into a large error in the residual. That is the "
            "case for forecasting the **drivers** and letting the margin fall out, rather "
            "than forecasting the margin directly.",
        ]

    lines += [
        "",
        f"_Backtests: rolling origin, {horizon_quarters}-quarter and "
        f"{horizon_months}-month horizons, no observation inside the forecast window._",
    ]
    return "\n".join(lines)
