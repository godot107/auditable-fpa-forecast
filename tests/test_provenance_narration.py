"""Tests for the provenance block written onto each Odoo journal entry.

The project's central claim is that every actual traces to the filing it came from.
Until now that trail stopped at the ERP boundary: the entry carried ``BS-2026-03-31``,
a period key, and nothing identifying the document. Validating a figure against EDGAR
meant leaving the ledger.

These tests pin the two things that make the narration worth having — that it names a
real accession with a URL that follows EDGAR's archive layout, and that it tells the
truth about which lines are filed and which are modeled. A provenance note that
overstates is worse than no note, so the allocation narration is asserted to *deny*
being a filed figure.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from fpa.config import DERIVED_BALANCE_SHEET, balance_sheet_lines, get_settings
from fpa.ingest.edgar import accession_index, filing_url
from fpa.ledger.odoo_load import (
    _quarter_end,
    allocation_narration,
    balance_sheet_narration,
)


@pytest.fixture(scope="module")
def accessions():
    return accession_index(get_settings())


@pytest.fixture(scope="module")
def cik():
    return get_settings().cik


# ---------------------------------------------------------------------------
# The URL
# ---------------------------------------------------------------------------
def test_filing_url_matches_edgars_archive_layout():
    """The accession appears twice in two forms, and the CIK is unpadded.

    Verified against a live 200 from sec.gov while writing this. If the shape drifts
    the link silently 404s, which is worse than an absent link because it looks
    checkable.
    """
    url = filing_url("0001065280", "0001065280-26-000138")
    assert url == (
        "https://www.sec.gov/Archives/edgar/data/1065280/"
        "000106528026000138/0001065280-26-000138-index.htm"
    )


def test_filing_url_accepts_padded_and_unpadded_cik():
    assert filing_url("0001065280", "0001065280-26-000138") == filing_url(
        1065280, "0001065280-26-000138"
    )


# ---------------------------------------------------------------------------
# Balance sheet: filed positions
# ---------------------------------------------------------------------------
def test_balance_sheet_narration_names_the_filing(accessions, cik):
    html = balance_sheet_narration(accessions, pd.Timestamp("2026-03-31"), cik)

    assert "0001065280-26-000138" in html
    assert "10-Q" in html
    assert filing_url(cik, "0001065280-26-000138") in html


def test_balance_sheet_narration_separates_filed_from_derived(accessions, cik):
    """The point of the block: an auditor opening the entry cold learns that
    content assets is a residual without having to be told."""
    html = balance_sheet_narration(accessions, pd.Timestamp("2026-03-31"), cik)

    assert "REAL" in html and "IMPLIED" in html
    for derived in DERIVED_BALANCE_SHEET:
        assert derived in html
    # ... and the derived lines are named under IMPLIED, not under REAL.
    real_block, implied_block = html.split("<b>IMPLIED</b>")
    for derived in DERIVED_BALANCE_SHEET:
        assert derived not in real_block
        assert derived in implied_block


def test_every_balance_sheet_line_is_accounted_for_as_real_or_implied(accessions, cik):
    """No line may be silently omitted — that is how a residual passes as filed."""
    html = balance_sheet_narration(accessions, pd.Timestamp("2026-03-31"), cik)
    for line, _sign in balance_sheet_lines():
        assert line.account in html, f"{line.account} missing from the provenance block"


def test_balance_sheet_narration_degrades_when_no_filing_is_on_record(accessions, cik):
    html = balance_sheet_narration(accessions, pd.Timestamp("1999-12-31"), cik)
    assert "No accession on file" in html
    assert "sec.gov" not in html


# ---------------------------------------------------------------------------
# Allocations: modeled detail
# ---------------------------------------------------------------------------
def test_allocation_narration_denies_being_a_filed_figure(accessions, cik):
    """A provenance note that overstates is worse than none.

    The monthly entry cites a filing, so the note must be explicit that the month
    itself is not in it — otherwise citing the 10-Q implies the month came from it.
    """
    html = allocation_narration(accessions, pd.Timestamp("2026-01-31"), cik)

    assert "MODELED" in html
    assert "not a filed figure" in html
    assert "phasing" in html and "cost center" in html
    # It still points at the document the quarterly total came from.
    assert "sec.gov" in html


def test_allocation_narration_cites_the_quarter_the_month_belongs_to(accessions, cik):
    html = allocation_narration(accessions, pd.Timestamp("2026-01-31"), cik)
    assert "2026-03-31" in html, "January must cite the Q1 filing, not its own month end"


def test_an_untagged_expense_line_is_named_not_omitted(accessions, cik):
    """The gap that made the citation overstate.

    ``MarketingExpense`` stops being tagged after 2024-09-30, so the accession behind a
    2026 quarter covers three of the four expense lines. Citing one 10-Q and saying
    nothing else implies all four came from it. The block must name the untagged line
    and the identity that produced it.
    """
    html = allocation_narration(accessions, pd.Timestamp("2026-06-30"), cik)

    assert "0001065280-26-000212" in html
    assert "IMPLIED" in html
    assert "marketing" in html
    assert "not covered by the accession above" in html
    assert "OperatingIncome" in html, "the identity that recovers it must be stated"


def test_a_fully_tagged_quarter_carries_no_implied_block(accessions, cik):
    """The caveat must be conditional, or it stops meaning anything.

    2021-Q2 has all four expense lines tagged, so there is nothing to disclaim.
    """
    html = allocation_narration(accessions, pd.Timestamp("2021-05-31"), cik)
    assert "IMPLIED" not in html
    assert "0001065280-22-000257" in html


def test_a_multi_filing_quarter_says_which_value_won(accessions, cik):
    """2024-Q3 draws on two filings, and the reason is not obvious.

    Three lines come from a 2025 comparative because the dedupe keeps the most recently
    filed value; marketing comes from the original 2024 10-Q because that is the last
    time it was tagged. Without the note, a reader asks why a 2024 entry points at a
    2025 document.
    """
    html = allocation_narration(accessions, pd.Timestamp("2024-08-31"), cik)

    assert "0001065280-24-000287" in html
    assert "0001065280-25-000406" in html
    assert "More than one filing backs this quarter" in html
    assert "most recently filed value wins" in html


# ---------------------------------------------------------------------------
# The HTML has to survive Odoo's sanitizer
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("month", ["2026-06-30", "2024-08-31", "2021-05-31"])
def test_emitted_html_is_round_trip_stable(accessions, cik, month):
    """Why the backfill is idempotent rather than rewriting every entry each run.

    ``_backfill_narration`` writes when the stored text differs from the generated text.
    Odoo normalises HTML on write — ``<br/>`` becomes ``<br>`` and a bare ``&`` becomes
    ``&amp;`` — so emitting either form guarantees a permanent mismatch and a log line
    claiming 104 entries changed on every run. Measured against a live instance before
    this test was written.
    """
    for html in (
        allocation_narration(accessions, pd.Timestamp(month), cik),
        balance_sheet_narration(accessions, pd.Timestamp(month), cik),
    ):
        assert "<br/>" not in html, "Odoo stores <br>; emitting <br/> never matches"
        # Every ampersand must already be an entity, or Odoo escapes it on write.
        assert not re.search(r"&(?!amp;|lt;|gt;|quot;|#)", html), "unescaped & in narration"


@pytest.mark.parametrize(
    ("month", "quarter"),
    [
        ("2026-01-31", "2026-03-31"),
        ("2026-02-28", "2026-03-31"),
        ("2026-03-31", "2026-03-31"),
        ("2024-07-31", "2024-09-30"),
        ("2020-12-31", "2020-12-31"),
    ],
)
def test_quarter_end_maps_a_month_to_its_filed_quarter(month, quarter):
    assert _quarter_end(pd.Timestamp(month)) == pd.Timestamp(quarter)


# ---------------------------------------------------------------------------
# The property that makes a single citation honest
# ---------------------------------------------------------------------------
def _filed_accounts():
    return [
        line.account
        for line, _ in balance_sheet_lines()
        if line.account not in DERIVED_BALANCE_SHEET
    ]


def test_one_accession_per_balance_sheet_date_inside_the_window(accessions):
    """Inside the posted window every balance-sheet date traces to one filing.

    All 26 quarter ends from 2020 on draw from a single accession, which is what
    makes a one-document citation honest rather than a convenient simplification.
    """
    settings = get_settings()
    rows = accessions[accessions["account"].isin(_filed_accounts())]
    in_window = rows[rows["end"] >= pd.Timestamp(settings.window_start)]

    per_date = in_window.groupby("end")["accn"].nunique()
    assert len(per_date) == 26
    assert per_date.max() == 1, f"multiple filings behind: {per_date[per_date > 1]}"


def test_a_balance_sheet_date_can_draw_on_several_filings(accessions):
    """And outside the window it does, which is why the formatter takes a list.

    A balance sheet *as of* a date is not the same thing as a balance sheet *as filed
    in one document*. At 2013-03-31 the cash figure was last restated in a 10-Q filed
    in July 2014, while its neighbours still come from the original April 2013 filing —
    three distinct accessions behind one date. The dedupe is correct to prefer the
    latest value; the narration would be wrong to cite only the first.
    """
    rows = accessions[accessions["account"].isin(_filed_accounts())]
    pre_window = rows[rows["end"] < pd.Timestamp(get_settings().window_start)]
    per_date = pre_window.groupby("end")["accn"].nunique()

    assert per_date.max() > 1, "expected restatement scatter in the pre-window history"

    multi = per_date[per_date > 1].index[0]
    html = balance_sheet_narration(accessions, multi, get_settings().cik)
    expected = set(rows[rows["end"] == multi]["accn"])
    for accn in expected:
        assert accn in html, f"{accn} dropped from a multi-source provenance block"
