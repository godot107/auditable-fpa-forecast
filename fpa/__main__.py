"""Thin CLI wrapper. All logic lives in :mod:`fpa.pipeline`."""

from __future__ import annotations

import argparse
import logging
import sys

from fpa.config import get_settings
from fpa.pipeline import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fpa", description=__doc__)
    parser.add_argument(
        "--refresh", action="store_true", help="re-pull from SEC EDGAR instead of the pinned vintage"
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress logging")
    parser.add_argument(
        "--groundedness",
        action="store_true",
        help="measure the groundedness checker's error rates (fast; writes reports/)",
    )
    parser.add_argument(
        "--intervals",
        action="store_true",
        help="also evaluate posterior-predictive intervals (needs requirements-bayes.txt; ~6 min)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s"
    )

    settings = get_settings()
    result = run(settings, refresh=args.refresh)

    print()
    print(result.controls.to_markdown())
    print()

    if not result.gate_passed:
        print("Pipeline halted by the control gate. No forecast was produced.")
        return 1

    print(result.validation_markdown())

    if args.groundedness:
        from fpa.narrative import from_pipeline
        from fpa.narrative.evaluation import build_corpus, evaluate, report_markdown
        from fpa.variance import build_variance_report

        variance = build_variance_report(
            result.ledger, result.budget, result.revenue, result.revenue_budget,
            result.drivers, periods=12,
        )
        facts = from_pipeline(result, variance)
        report = evaluate(build_corpus(facts), facts)
        markdown = report_markdown(report)
        print()
        print(markdown)

        destination = settings.reports_dir / "groundedness_evaluation.md"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(markdown + "\n")
        print(f"\nWritten to {destination}")

        if not report.clean:
            # A fabrication accepted or a numeral never inspected is a failing gate,
            # not a metric. The exit code has to say so.
            return 1

    if args.intervals:
        try:
            from fpa.forecast.bayes import evaluate_hierarchy, interval_report_markdown
        except ImportError:  # pragma: no cover - depends on the optional stack
            print("\nNumPyro/JAX not installed. `pip install -r requirements-bayes.txt`")
            return 0

        reports = evaluate_hierarchy(result.ledger, horizon=settings.horizon_months)
        markdown = interval_report_markdown(reports)
        print()
        print(markdown)

        destination = settings.reports_dir / "interval_calibration.md"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(markdown + "\n")
        print(f"\nWritten to {destination}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
