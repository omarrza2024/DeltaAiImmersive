"""Universe resolution: ETF holdings (live with static fallback), sectors, custom tickers."""

from __future__ import annotations

from typing import Callable

from finance_dashboard.models import UserInputError, dedupe_tickers, normalize_ticker
from finance_dashboard.recommender.models import RecommendationRequest

# Static fallbacks: live holdings via yfinance are preferred, but frequently
# return partial or empty data, so every supported ETF needs an offline list.
# Keys cover the major market-cap indices (SPY/QQQ/DIA/IWM/VTI/VOO) and the
# full set of Select Sector SPDR ETFs so a user can pick any slice they want.
ETF_FALLBACK_HOLDINGS: dict[str, list[str]] = {
    # --- Broad-market / major indices ---
    "SPY":  ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "LLY", "JPM", "UNH",
             "XOM", "V", "JNJ", "PG", "HD"],
    "VOO":  ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "LLY", "JPM", "UNH"],
    "QQQ":  ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "TSLA", "COST", "NFLX",
             "AMD", "PEP", "ADBE", "CSCO", "INTU"],
    "DIA":  ["UNH", "GS", "MSFT", "HD", "CRM", "V", "MCD", "CAT", "AXP", "AMGN",
             "SHW", "TRV", "HON", "IBM", "JPM"],
    "IWM":  ["SMCI", "STRL", "ANF", "FTAI", "APPF", "ONTO", "MLI", "FLR", "FIX", "RMBS"],
    "VTI":  ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "BRK-B", "LLY", "JPM"],
    # --- SPDR Select Sector ETFs ---
    "XLK":  ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "CSCO", "AMD", "ADBE", "ACN",
             "IBM", "INTU", "NOW", "TXN", "QCOM"],
    "XLV":  ["LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "AMGN", "ISRG", "DHR",
             "PFE", "VRTX", "SYK", "BSX", "GILD"],
    "XLF":  ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "AXP",
             "BLK", "C", "SCHW", "CB", "PGR"],
    "XLE":  ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "OXY", "WMB", "VLO"],
    "XLY":  ["AMZN", "TSLA", "HD", "MCD", "BKNG", "NKE", "LOW", "SBUX", "TJX", "ABNB"],
    "XLP":  ["WMT", "PG", "COST", "KO", "PEP", "PM", "MDLZ", "MO", "CL", "TGT"],
    "XLI":  ["GE", "CAT", "RTX", "HON", "UNP", "BA", "DE", "UPS", "LMT", "ETN"],
    "XLU":  ["NEE", "SO", "DUK", "SRE", "AEP", "D", "EXC", "XEL", "ED", "PCG"],
    "XLC":  ["META", "GOOGL", "GOOG", "NFLX", "DIS", "VZ", "T", "CMCSA", "EA", "TTWO"],
    "XLRE": ["PLD", "AMT", "EQIX", "WELL", "SPG", "DLR", "O", "PSA", "CCI", "EXR"],
    "XLB":  ["LIN", "SHW", "ECL", "APD", "FCX", "NEM", "CTVA", "DD", "NUE", "DOW"],
    # --- Vanguard industry ETFs ---
    "VGT":  ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "CSCO", "AMD", "ADBE", "ACN"],
    "VHT":  ["LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "AMGN", "ISRG", "DHR"],
    "VFH":  ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "AXP"],
}

# Full-name display label for each supported ETF. The UI pairs these with
# the symbol so a user who doesn't memorize three-letter ETF tickers can
# still tell which fund they're picking.
ETF_DISPLAY_NAMES: dict[str, str] = {
    "SPY":  "SPDR S&P 500 ETF",
    "VOO":  "Vanguard S&P 500 ETF",
    "QQQ":  "Invesco QQQ Trust (Nasdaq 100)",
    "DIA":  "SPDR Dow Jones Industrial Average",
    "IWM":  "iShares Russell 2000 ETF",
    "VTI":  "Vanguard Total Stock Market ETF",
    "XLK":  "Technology Select Sector SPDR",
    "XLV":  "Health Care Select Sector SPDR",
    "XLF":  "Financial Select Sector SPDR",
    "XLE":  "Energy Select Sector SPDR",
    "XLY":  "Consumer Discretionary Select Sector SPDR",
    "XLP":  "Consumer Staples Select Sector SPDR",
    "XLI":  "Industrial Select Sector SPDR",
    "XLU":  "Utilities Select Sector SPDR",
    "XLC":  "Communication Services Select Sector SPDR",
    "XLRE": "Real Estate Select Sector SPDR",
    "XLB":  "Materials Select Sector SPDR",
    "VGT":  "Vanguard Information Technology ETF",
    "VHT":  "Vanguard Health Care ETF",
    "VFH":  "Vanguard Financials ETF",
}

# Sector universes drawn from across the supported ETFs; selectable without an ETF.
PREDEFINED_UNIVERSES: dict[str, list[str]] = {
    "Tech":        ["AAPL", "MSFT", "NVDA", "AVGO", "GOOGL", "META", "AMD", "CRM", "ADBE", "ORCL"],
    "Medical":     ["LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "PFE", "AMGN", "ISRG", "VRTX"],
    "Finance":     ["JPM", "BRK-B", "V", "MA", "BAC", "WFC", "GS", "MS", "BLK", "AXP"],
    "Energy":      ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "OXY", "WMB", "VLO"],
    "Consumer":    ["AMZN", "TSLA", "HD", "MCD", "WMT", "PG", "COST", "KO", "PEP", "NKE"],
    "Industrials": ["CAT", "GE", "HON", "UNP", "BA", "DE", "UPS", "LMT", "ETN", "RTX"],
    "Communications": ["META", "GOOGL", "GOOG", "NFLX", "DIS", "VZ", "T", "CMCSA", "EA", "TTWO"],
    "Utilities":   ["NEE", "SO", "DUK", "SRE", "AEP", "D", "EXC", "XEL", "ED", "PCG"],
    "Real Estate": ["PLD", "AMT", "EQIX", "WELL", "SPG", "DLR", "O", "PSA", "CCI", "EXR"],
    "Materials":   ["LIN", "SHW", "ECL", "APD", "FCX", "NEM", "CTVA", "DD", "NUE", "DOW"],
}


def _yfinance_holdings(etf: str) -> list[str]:
    """Fetch top holdings for an ETF from Yahoo Finance."""
    import yfinance as yf

    holdings = yf.Ticker(etf).funds_data.top_holdings
    return [str(s) for s in holdings.index] if holdings is not None else []


class UniverseProvider:
    """Resolves a request's universe sources into one deduplicated ticker list."""

    def __init__(self, holdings_fetcher: Callable[[str], list[str]] | None = None):
        self._holdings_fetcher = holdings_fetcher or _yfinance_holdings

    def etf_holdings(self, etf: str) -> list[str]:
        if etf not in ETF_FALLBACK_HOLDINGS:
            raise UserInputError(
                f"Unknown ETF: {etf}. Supported: {', '.join(sorted(ETF_FALLBACK_HOLDINGS))}"
            )
        try:
            live = self._holdings_fetcher(etf)
        except Exception:
            live = []
        return [normalize_ticker(t) for t in live] or ETF_FALLBACK_HOLDINGS[etf]

    def resolve(self, request: RecommendationRequest) -> list[str]:
        """Union of ETF holdings, selected universes, and custom tickers, in that order."""
        tickers: list[str] = []
        if request.etf:
            tickers.extend(self.etf_holdings(request.etf))
        for universe in request.universes:
            if universe not in PREDEFINED_UNIVERSES:
                raise UserInputError(
                    f"Unknown universe: {universe}. "
                    f"Supported: {', '.join(sorted(PREDEFINED_UNIVERSES))}"
                )
            tickers.extend(PREDEFINED_UNIVERSES[universe])
        tickers.extend(normalize_ticker(t) for t in request.custom_tickers)

        tickers = dedupe_tickers([t for t in tickers if t])
        if not tickers:
            raise UserInputError(
                "Select at least one source: an ETF, a predefined universe, or custom tickers."
            )
        return tickers
