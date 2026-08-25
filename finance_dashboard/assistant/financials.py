"""SEC-financials investment memo via EdgarTools."""

from __future__ import annotations

import os
from dataclasses import dataclass

MEMO_DISCLAIMER = (
    "_Data sourced from SEC EDGAR via edgartools. "
    "Historical filings do not guarantee future performance._"
)

# Rows we pull from each statement, keyed by the human-readable label and
# mapped to the XBRL concept names edgartools uses as DataFrame indices.
# Multiple aliases are probed in order — different filers use different concepts.
_INCOME_ROWS: dict[str, tuple[str, ...]] = {
    "Revenue": (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "TotalRevenue",
    ),
    "Gross Profit":     ("GrossProfit",),
    "Operating Income": ("OperatingIncomeLoss", "OperatingIncome"),
    "Net Income":       ("NetIncomeLoss", "NetIncome"),
}
_BALANCE_ROWS: dict[str, tuple[str, ...]] = {
    "Total Assets":       ("Assets", "TotalAssets"),
    "Total Liabilities":  ("Liabilities", "TotalLiabilities"),
    "Total Equity":       ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "TotalEquity"),
    "Cash & Equivalents": ("CashAndCashEquivalentsAtCarryingValue", "Cash", "CashAndCashEquivalentsAtCarryingValueIncludingRestrictedCash"),
}
_CASHFLOW_ROWS: dict[str, tuple[str, ...]] = {
    "Operating": ("NetCashProvidedByUsedInOperatingActivities",),
    "Investing": ("NetCashProvidedByUsedInInvestingActivities",),
    "Financing": ("NetCashProvidedByUsedInFinancingActivities",),
}


class FinancialsUnavailableError(RuntimeError):
    """Raised when SEC data cannot be fetched — identity missing, ticker not filed, etc."""


@dataclass
class CompanyMemo:
    ticker: str
    name: str
    cik: str | None
    industry: str | None
    fiscal_year_end: str | None
    description: str | None
    years: list[str]
    income_statement: dict[str, dict[str, float]]
    balance_sheet: dict[str, dict[str, float]]
    cash_flow: dict[str, dict[str, float]]
    recent_filings: list[dict]


def get_sec_identity() -> str | None:
    """Streamlit secrets first, env var second."""
    try:
        import streamlit as st
        return st.secrets.get("SEC_IDENTITY") or os.getenv("SEC_IDENTITY")
    except Exception:
        return os.getenv("SEC_IDENTITY")


def _pick_row(df, aliases: tuple[str, ...]) -> dict[str, float]:
    """Extract one row (in $M) from a period-only statement DataFrame."""
    import pandas as pd

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {}
    for alias in aliases:
        if alias in df.index:
            series = df.loc[alias].dropna()
            return {str(col): float(v) / 1_000_000 for col, v in series.items()}
    return {}


def _statement_dataframe(company, method_names: tuple[str, ...]):
    """Call the first working statement method; return a period-only DataFrame or None."""
    for name in method_names:
        method = getattr(company, name, None)
        if not callable(method):
            continue
        try:
            statement = method()
        except Exception:
            continue
        # MultiPeriodStatement in edgartools 5.x — normalize to DataFrame.
        df = statement.to_dataframe() if hasattr(statement, "to_dataframe") else statement
        if df is None or not hasattr(df, "columns"):
            continue
        period_cols = [c for c in df.columns if str(c).startswith(("FY ", "Q"))]
        if period_cols:
            return df[period_cols]
    return None


def fetch_company_memo(ticker: str) -> CompanyMemo:
    """Fetch a memo for `ticker` from SEC EDGAR. Raises FinancialsUnavailableError on any failure."""
    identity = get_sec_identity()
    if not identity:
        raise FinancialsUnavailableError(
            "SEC_IDENTITY not set. Add `SEC_IDENTITY = \"Your Name your@email.com\"` "
            "to `.streamlit/secrets.toml`."
        )

    try:
        from edgar import Company, set_identity
    except ImportError as exc:
        raise FinancialsUnavailableError(f"edgartools not installed: {exc}") from exc

    set_identity(identity)

    try:
        company = Company(ticker)
    except Exception as exc:
        raise FinancialsUnavailableError(
            f"No SEC filings for {ticker} (foreign issuer, ETF, or delisted): {exc}"
        ) from exc

    income = _statement_dataframe(company, ("income_statement",))
    balance = _statement_dataframe(company, ("balance_sheet",))
    cash = _statement_dataframe(company, ("cash_flow_statement", "cashflow_statement", "cash_flow"))

    years = _extract_years(income, balance, cash)
    income_data = {label: _pick_row(income, aliases) for label, aliases in _INCOME_ROWS.items()}
    balance_data = {label: _pick_row(balance, aliases) for label, aliases in _BALANCE_ROWS.items()}
    cash_data = {label: _pick_row(cash, aliases) for label, aliases in _CASHFLOW_ROWS.items()}

    return CompanyMemo(
        ticker=ticker,
        name=_safe_attr(company, "name", ticker),
        cik=_format_cik(_safe_attr(company, "cik", None)),
        industry=_safe_attr(company, "industry", None) or _safe_attr(company, "business_category", None),
        fiscal_year_end=_safe_attr(company, "fiscal_year_end", None),
        description=None,  # edgartools 5.x doesn't expose a short description on Company.
        years=years,
        income_statement=income_data,
        balance_sheet=balance_data,
        cash_flow=cash_data,
        recent_filings=_recent_filings(company, limit=5),
    )


def _extract_years(*dfs) -> list[str]:
    for df in dfs:
        if df is not None and hasattr(df, "columns") and len(df.columns) > 0:
            # edgartools columns already come newest-first; keep as-is.
            return [str(c) for c in df.columns]
    return []


def _format_cik(cik) -> str | None:
    """Left-pad numeric CIK to 10 digits (SEC's canonical form)."""
    if cik is None:
        return None
    s = str(cik).strip()
    return s.zfill(10) if s.isdigit() else s


def _safe_attr(obj, name: str, default):
    try:
        val = getattr(obj, name, default)
        return val() if callable(val) else val
    except Exception:
        return default


def _recent_filings(company, limit: int) -> list[dict]:
    try:
        filings = company.get_filings(form=["10-K", "10-Q", "8-K"])
    except Exception:
        try:
            filings = company.get_filings()
        except Exception:
            return []
    out: list[dict] = []
    for f in list(filings)[:limit]:
        out.append(
            {
                "date": str(getattr(f, "filing_date", getattr(f, "date", ""))),
                "form": str(getattr(f, "form", "")),
                "description": str(getattr(f, "primary_document_description", "") or getattr(f, "description", "")),
            }
        )
    return out


def _fmt_dollars_millions(value: float) -> str:
    """Format a value already in $M."""
    if value is None:
        return "—"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}B"
    return f"${value:,.0f}M"


def _render_statement(title: str, rows: dict[str, dict[str, float]], years: list[str]) -> list[str]:
    if not years or not any(rows.values()):
        return [f"### {title}\n\n_No data available._\n"]
    lines = [f"### {title}\n"]
    lines.append("| Line item | " + " | ".join(years) + " |")
    lines.append("|" + "---|" * (len(years) + 1))
    for label, values in rows.items():
        cells = [_fmt_dollars_millions(values.get(y)) if values.get(y) is not None else "—" for y in years]
        lines.append(f"| **{label}** | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _pct_change(newest: float | None, older: float | None) -> str:
    if newest is None or older is None or not older:
        return "n/a"
    return f"{(newest / older - 1.0) * 100:+.0f}% YoY"


def _first_present_year_values(row: dict[str, float], years: list[str]) -> list[tuple[str, float]]:
    return [(y, row[y]) for y in years if y in row]


def financials_summary(ticker: str) -> str:
    """Compact plain-text SEC snapshot the LLM can drop into an answer.

    Meant to be injected into the system prompt when the user asks about a
    specific stock. Short enough to keep context bloat low; concrete numbers so
    the LLM never has to invent them.
    """
    try:
        memo = fetch_company_memo(ticker)
    except FinancialsUnavailableError as exc:
        return f"[{ticker}: SEC data unavailable — {exc}]"

    lines: list[str] = [f"=== {memo.name or ticker} ({ticker}) — SEC data ==="]
    if memo.industry:
        lines.append(f"Industry: {memo.industry}")

    years = memo.years[:3]
    if not years:
        lines.append("No structured financial data available in recent filings.")
        return "\n".join(lines)

    def row_line(label: str, row: dict[str, float]) -> str | None:
        vals = _first_present_year_values(row, years)
        if not vals:
            return None
        formatted = ", ".join(f"{y} {_fmt_dollars_millions(v)}" for y, v in vals)
        yoy = _pct_change(vals[0][1] if vals else None,
                          vals[1][1] if len(vals) > 1 else None)
        return f"{label}: {formatted} ({yoy})"

    for label, row in memo.income_statement.items():
        line = row_line(label, row)
        if line:
            lines.append(line)
    for label, row in memo.balance_sheet.items():
        line = row_line(label, row)
        if line:
            lines.append(line)
    op_cf = memo.cash_flow.get("Operating", {})
    op_line = row_line("Operating Cash Flow", op_cf)
    if op_line:
        lines.append(op_line)

    if memo.recent_filings:
        latest = memo.recent_filings[0]
        lines.append(f"Latest filing: {latest.get('form', '?')} on {latest.get('date', '?')}")

    return "\n".join(lines)


def render_memo_markdown(memo: CompanyMemo) -> str:
    """Format a CompanyMemo as a markdown investment memo."""
    lines: list[str] = [
        f"## Investment Memo — {memo.name} ({memo.ticker})",
        "",
        "**Company Overview**",
        "",
    ]
    header_facts = [
        ("CIK", memo.cik),
        ("Industry", memo.industry),
        ("Fiscal Year End", memo.fiscal_year_end),
    ]
    for label, value in header_facts:
        if value:
            lines.append(f"- **{label}:** {value}")
    lines.append("")

    if memo.description:
        lines += ["**Business Description**", "", f"> {memo.description}", ""]

    lines += _render_statement("Income Statement", memo.income_statement, memo.years)
    lines += _render_statement("Balance Sheet", memo.balance_sheet, memo.years)
    lines += _render_statement("Cash Flow Statement", memo.cash_flow, memo.years)

    if memo.recent_filings:
        lines += ["### Recent Filings", ""]
        for f in memo.recent_filings:
            desc = f" — {f['description']}" if f.get("description") else ""
            lines.append(f"- **{f['date']}** {f['form']}{desc}")
        lines.append("")

    lines.append(MEMO_DISCLAIMER)
    return "\n".join(lines)
