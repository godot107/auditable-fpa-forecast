"""Regional revenue from SEC's Financial Statement Data Sets.

``companyfacts`` — the API the rest of this project's ingest uses — carries **no
dimension field**. Its per-fact keys are exactly ``accn``, ``end``, ``filed``,
``form``, ``fp``, ``frame``, ``fy``, ``start`` and ``val``. That is not an omission
to work around: it means the API only ever exposes the consolidated value of a tag.
Revenue by region is a *dimensional* fact, so no amount of querying companyfacts
will produce it.

The dimensional data lives in SEC's quarterly **Financial Statement Data Sets**,
where ``num.txt`` carries a ``segments`` column. That is a different pipeline with
different properties, and the differences are the interesting part:

* **Bulk, not per-company.** Each quarterly ZIP is ~85 MB compressed and ~560 MB
  expanded, covering every filer. Extracting one registrant means streaming and
  filtering, not loading.
* **One ZIP per filing quarter.** A time series is assembled across archives rather
  than returned by a query.
* **The audit trail survives.** ``adsh`` *is* the accession number, so a regional
  figure traces back to its filing exactly like every other actual in this project.
  That is the reason this source was chosen over a vendor feed with the numbers
  already assembled.

The trap this module exists to handle: Netflix tags **two different** geographic
breakdowns, and they overlap. ``Geographical=US`` is a standalone country
disclosure; the four-region operating breakdown is tagged
``Geographical=<region>;ProductOrService=Streaming``. Summing everything with a
``Geographical=`` axis double-counts the United States. The filter is explicit and
``segment_revenue_foots_to_filed`` re-proves the result against filed total revenue
on every run.
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from pathlib import Path

import httpx
import pandas as pd

from fpa.config import REGION_LABELS, SEGMENT_REVENUE_TAG, STREAMING_TOTAL, Settings

logger = logging.getLogger(__name__)

FSDS_URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{year}q{quarter}.zip"

# ``qtrs`` in num.txt is the number of quarters the fact spans: 0 = instant,
# 1 = one quarter, 4 = one year. Revenue is a duration, so 0 is never wanted.
DURATION_BY_QTRS = {1: "quarter", 4: "annual"}


def _archive_path(settings: Settings, year: int, quarter: int) -> Path:
    return settings.data_dir / "fsds" / f"{year}q{quarter}.zip"


def download_archive(settings: Settings, year: int, quarter: int, *, refresh: bool = False) -> Path:
    """Fetch one quarterly archive, cached on disk.

    ~85 MB each, so the cache is not an optimization — re-downloading four of these
    per demo run would be antisocial toward SEC and slow enough to notice.
    """
    path = _archive_path(settings, year, quarter)
    if path.exists() and not refresh:
        logger.debug("archive cache hit: %s", path.name)
        return path

    url = FSDS_URL.format(year=year, quarter=quarter)
    logger.info("downloading %s", url)
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": settings.edgar_user_agent}
    with httpx.stream("GET", url, headers=headers, timeout=300.0, follow_redirects=True) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_bytes(chunk_size=1 << 20):
                handle.write(chunk)
    return path


def _submissions_for_cik(archive: zipfile.ZipFile, cik: int) -> dict[str, dict]:
    """Map accession number → submission metadata, for one registrant."""
    with archive.open("sub.txt") as handle:
        reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8", errors="replace"), delimiter="\t")
        return {
            row["adsh"]: {"form": row["form"], "period": row["period"], "fy": row["fy"], "fp": row["fp"]}
            for row in reader
            if row["cik"].strip() and int(row["cik"]) == cik
        }


def _parse_segments(segments: str) -> dict[str, str]:
    """``"Geographical=EMEA;ProductOrService=Streaming;"`` → ``{axis: member}``."""
    out: dict[str, str] = {}
    for part in segments.split(";"):
        if "=" in part:
            axis, _, member = part.partition("=")
            out[axis.strip()] = member.strip()
    return out


def extract_segment_revenue(path: Path, cik: int) -> pd.DataFrame:
    """Stream one archive and return this registrant's regional revenue rows.

    ``num.txt`` is ~560 MB expanded, so it is read as a stream through the ZIP
    rather than extracted or loaded into a frame. The filter runs per row.
    """
    rows: list[dict] = []
    with zipfile.ZipFile(path) as archive:
        submissions = _submissions_for_cik(archive, cik)
        if not submissions:
            logger.warning("%s contains no filings for CIK %s", path.name, cik)
            return pd.DataFrame()

        with archive.open("num.txt") as handle:
            reader = csv.DictReader(
                io.TextIOWrapper(handle, encoding="utf-8", errors="replace"), delimiter="\t"
            )
            for row in reader:
                if row["adsh"] not in submissions:
                    continue
                if row["tag"] != SEGMENT_REVENUE_TAG or row["uom"] != "USD":
                    continue

                axes = _parse_segments(row["segments"])
                if axes.get("ProductOrService") != "Streaming":
                    continue

                region = axes.get("Geographical")
                if region is None:
                    # Streaming revenue with no geographic axis: the consolidated
                    # streaming line. Captured because it is what the four regions
                    # actually sum to — see ``STREAMING_TOTAL``.
                    label = STREAMING_TOTAL
                elif region in REGION_LABELS:
                    label = REGION_LABELS[region]
                else:
                    # Any other geographic member, notably the standalone
                    # ``Geographical=US`` country disclosure. It overlaps UCAN and
                    # summing it in would inflate revenue by ~40%.
                    continue

                qtrs = int(row["qtrs"])
                if qtrs not in DURATION_BY_QTRS:
                    continue

                submission = submissions[row["adsh"]]
                rows.append(
                    {
                        "end": pd.Timestamp(row["ddate"]),
                        "region": label,
                        "period_type": DURATION_BY_QTRS[qtrs],
                        "value": float(row["value"]),
                        "member": region or "consolidated",
                        "form": submission["form"],
                        # The audit trail, carried through exactly as the
                        # companyfacts ingest carries it.
                        "accn": row["adsh"],
                        "source": path.name,
                    }
                )

    logger.info("%s: %d regional revenue facts", path.name, len(rows))
    return pd.DataFrame(rows)


def load_segment_revenue(
    settings: Settings,
    quarters: tuple[tuple[int, int], ...] | None = None,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Regional revenue across several archives, deduped and pinned to Parquet.

    Later filings restate earlier periods, exactly as they do in companyfacts, so
    the same ``(region, period)`` can appear in several archives. The most recently
    filed value wins — the same rule ``_dedupe_restatements`` applies to the main
    fact table, kept deliberately identical so the two ingests cannot disagree about
    what a restatement is.
    """
    from fpa.cache import cached_parquet

    quarters = quarters or settings.segment_quarters
    path = settings.vintage_path(f"segment_revenue_{settings.ticker.lower()}")

    def fetch() -> pd.DataFrame:
        frames = []
        for year, quarter in quarters:
            archive = download_archive(settings, year, quarter, refresh=refresh)
            frame = extract_segment_revenue(archive, int(settings.cik))
            if not frame.empty:
                frames.append(frame)

        if not frames:
            raise ValueError(f"no regional revenue found for CIK {settings.cik}")

        combined = pd.concat(frames, ignore_index=True)
        # Archives are named YYYYqN in filing order, so the last occurrence of a
        # period is the most recently filed one.
        combined = combined.drop_duplicates(
            subset=["region", "end", "period_type"], keep="last"
        )
        return combined.sort_values(["period_type", "end", "region"]).reset_index(drop=True)

    return cached_parquet(path, fetch, refresh=refresh)


def regional_revenue(
    settings: Settings, *, period_type: str = "annual", refresh: bool = False
) -> pd.DataFrame:
    """Wide regional revenue: one row per period end, one column per region.

    Defaults to **annual**, and that is a coverage statement rather than a
    preference. A 10-K restates the prior two years, so four archives yield five
    complete fiscal years; a 10-Q carries only its own quarter and the comparative,
    so the quarterly series is as long as the number of 10-Q archives pulled. Widen
    ``Settings.segment_quarters`` to lengthen it — at ~85 MB per archive, that is a
    cost worth declaring rather than defaulting into.
    """
    facts = load_segment_revenue(settings, refresh=refresh)
    subset = facts[facts["period_type"] == period_type]
    if subset.empty:
        available = sorted(facts["period_type"].unique())
        raise ValueError(f"no {period_type!r} segment facts; available: {available}")

    wide = subset.pivot_table(index="end", columns="region", values="value", aggfunc="last")
    regions = [c for c in wide.columns if c != STREAMING_TOTAL]
    wide["total"] = wide[regions].sum(axis=1)
    # Set attrs last: pandas does not carry them through pivot/assignment.
    wide.attrs["period_type"] = period_type
    wide.attrs["accessions"] = sorted(subset["accn"].unique().tolist())
    return wide
