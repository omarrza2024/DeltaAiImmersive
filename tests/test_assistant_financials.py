"""Tests for the SEC-financials memo: pure render + mocked fetch failure."""

import pytest

from finance_dashboard.assistant.financials import (
    CompanyMemo,
    FinancialsUnavailableError,
    render_memo_markdown,
)


def sample_memo() -> CompanyMemo:
    return CompanyMemo(
        ticker="AAPL",
        name="Apple Inc.",
        cik="0000320193",
        industry="Technology - Consumer Electronics",
        fiscal_year_end="September",
        description="Apple designs, manufactures, and markets smartphones, computers, and services.",
        years=["2023", "2022", "2021"],
        income_statement={
            "Revenue":          {"2023": 383285.0, "2022": 394328.0, "2021": 365817.0},
            "Gross Profit":     {"2023": 169148.0, "2022": 170782.0, "2021": 152836.0},
            "Operating Income": {"2023": 114301.0, "2022": 119437.0, "2021": 108949.0},
            "Net Income":       {"2023":  96995.0, "2022":  99803.0, "2021":  94680.0},
        },
        balance_sheet={
            "Total Assets":      {"2023": 352583.0, "2022": 352755.0, "2021": 351002.0},
            "Total Liabilities": {"2023": 290437.0, "2022": 302083.0, "2021": 287912.0},
            "Total Equity":      {"2023":  62146.0, "2022":  50672.0, "2021":  63090.0},
            "Cash & Equivalents":{"2023":  29965.0, "2022":  23646.0, "2021":  34940.0},
        },
        cash_flow={
            "Operating": {"2023": 110543.0, "2022": 122151.0, "2021": 104038.0},
            "Investing": {"2023":   3705.0, "2022": -22354.0, "2021": -14545.0},
            "Financing": {"2023":-108488.0, "2022":-110749.0, "2021": -93353.0},
        },
        recent_filings=[
            {"date": "2024-02-01", "form": "10-Q", "description": "Q1 2024"},
            {"date": "2023-11-03", "form": "10-K", "description": "FY 2023 annual report"},
        ],
    )


def test_render_includes_header_and_disclaimer():
    md = render_memo_markdown(sample_memo())
    assert "Apple Inc." in md
    assert "AAPL" in md
    assert "0000320193" in md
    assert "SEC EDGAR" in md
    assert "historical" in md.lower()


def test_render_includes_all_three_statements():
    md = render_memo_markdown(sample_memo())
    assert "Revenue" in md and "Net Income" in md
    assert "Total Assets" in md and "Total Equity" in md
    assert "Operating" in md and "Financing" in md


def test_render_formats_large_dollars_readable():
    md = render_memo_markdown(sample_memo())
    # 383285 (millions) should render as either $383,285M or $383.3B — either is fine,
    # but the raw unformatted "383285" alone would be a bug.
    assert "383,285" in md or "383.3" in md


def test_render_includes_years_in_order():
    md = render_memo_markdown(sample_memo())
    # Years should appear in the exact order provided (newest -> oldest).
    i2023, i2022, i2021 = md.index("2023"), md.index("2022"), md.index("2021")
    assert i2023 < i2022 < i2021


def test_render_lists_recent_filings():
    md = render_memo_markdown(sample_memo())
    assert "10-K" in md and "10-Q" in md
    assert "2024-02-01" in md


def test_render_handles_missing_optional_fields():
    memo = CompanyMemo(
        ticker="FOO",
        name="Foo Corp",
        cik=None,
        industry=None,
        fiscal_year_end=None,
        description=None,
        years=[],
        income_statement={},
        balance_sheet={},
        cash_flow={},
        recent_filings=[],
    )
    md = render_memo_markdown(memo)
    # Doesn't crash and still identifies the company.
    assert "Foo Corp" in md
    assert "FOO" in md


def test_fetch_raises_clear_error_when_identity_missing(monkeypatch):
    """Regression: without SEC_IDENTITY the library refuses; we must surface that plainly."""
    import finance_dashboard.assistant.financials as fin

    monkeypatch.setattr(fin, "get_sec_identity", lambda: None)
    with pytest.raises(FinancialsUnavailableError, match="SEC_IDENTITY"):
        fin.fetch_company_memo("AAPL")
