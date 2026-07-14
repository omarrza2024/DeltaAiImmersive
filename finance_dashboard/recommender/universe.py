"""Universe resolution: ETF holdings (live with static fallback), sectors, custom tickers."""

from __future__ import annotations

from typing import Callable

from finance_dashboard.models import UserInputError, dedupe_tickers, normalize_ticker
from finance_dashboard.recommender.models import RecommendationRequest

# Static fallbacks: live holdings via yfinance are preferred, but frequently
# return partial or empty data, so every supported ETF needs an offline list.
ETF_FALLBACK_HOLDINGS: dict[str, list[str]] = {
    "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "CSCO", "AMD", "ADBE", "ACN",
            "IBM", "INTU", "NOW", "TXN", "QCOM"],
    "XLV": ["LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "AMGN", "ISRG", "DHR",
            "PFE", "VRTX", "SYK", "BSX", "GILD"],
    "XLF": ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "AXP",
            "BLK", "C", "SCHW", "CB", "PGR"],
    "QQQ": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "TSLA", "COST", "NFLX",
            "AMD", "PEP", "ADBE", "CSCO", "INTU"],
    "SPY": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "LLY", "JPM", "UNH",
            "XOM", "V", "JNJ", "PG", "HD"],
}

# Sector universes drawn from across the supported ETFs; selectable without an ETF.
PREDEFINED_UNIVERSES: dict[str, list[str]] = {
    "Tech": ["AAPL", "MSFT", "NVDA", "AVGO", "GOOGL", "META", "AMD", "CRM", "ADBE", "ORCL"],
    "Medical": ["LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "PFE", "AMGN", "ISRG", "VRTX"],
    "Finance": ["JPM", "BRK-B", "V", "MA", "BAC", "WFC", "GS", "MS", "BLK", "AXP"],
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
