"""Per-stock scalar features computed from price history (plus pluggable extras)."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from finance_dashboard.analytics import annualized_volatility
from finance_dashboard.recommender.models import FeatureName

SMA_WINDOW = 20
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
MIN_HISTORY = max(SMA_WINDOW, MACD_SLOW + MACD_SIGNAL)


def sma_ratio(prices: pd.Series, window: int = SMA_WINDOW) -> float:
    """Last price relative to its SMA, minus 1 — positive when trading above trend."""
    p = prices.dropna()
    if len(p) < window:
        return float("nan")
    sma = p.rolling(window).mean().iloc[-1]
    if not sma or np.isnan(sma):
        return float("nan")
    return float(p.iloc[-1] / sma - 1.0)


def macd_histogram(prices: pd.Series) -> float:
    """Latest MACD histogram value normalized by price — scalar momentum signal."""
    p = prices.dropna()
    if len(p) < MACD_SLOW + MACD_SIGNAL:
        return float("nan")
    fast = p.ewm(span=MACD_FAST, adjust=False).mean()
    slow = p.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = fast - slow
    signal = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    hist = (macd_line - signal).iloc[-1]
    last = p.iloc[-1]
    return float(hist / last) if last else float("nan")


def total_return(prices: pd.Series) -> float:
    """Total return over the series as a fraction (0.10 == +10%)."""
    p = prices.dropna()
    if len(p) < 2 or not p.iloc[0]:
        return float("nan")
    return float(p.iloc[-1] / p.iloc[0] - 1.0)


def _neutral_sentiment(ticker: str) -> float:
    """Placeholder sentiment source — neutral until a real news feed is plugged in."""
    return 0.0


def _yfinance_fundamentals(ticker: str) -> dict[str, float]:
    import yfinance as yf

    info = yf.Ticker(ticker).info or {}
    return {
        "beta": float(info.get("beta") or "nan"),
        "trailing_pe": float(info.get("trailingPE") or "nan"),
    }


class FeatureProvider:
    """
    Builds the per-stock feature matrix.

    Volatility and returns are always computed — risk labeling depends on them —
    while the remaining columns follow the user's feature selection.
    """

    def __init__(
        self,
        fundamentals_lookup: Callable[[str], dict[str, float]] | None = None,
        sentiment_lookup: Callable[[str], float] | None = None,
    ):
        self._fundamentals = fundamentals_lookup or _yfinance_fundamentals
        self._sentiment = sentiment_lookup or _neutral_sentiment

    def compute(
        self,
        prices: pd.DataFrame,
        features: tuple[FeatureName, ...],
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        """Return (feature matrix indexed by ticker, skipped tickers with reasons)."""
        rows: dict[str, dict[str, float]] = {}
        skipped: dict[str, str] = {}

        for ticker in prices.columns:
            series = prices[ticker].dropna()
            if len(series) < MIN_HISTORY:
                skipped[ticker] = f"insufficient history ({len(series)} rows)"
                continue

            daily = series.pct_change().dropna()
            row: dict[str, float] = {
                "volatility": annualized_volatility(daily),
                "returns": total_return(series),
            }
            if FeatureName.SMA in features:
                row["sma_ratio"] = sma_ratio(series)
            if FeatureName.MACD in features:
                row["macd_hist"] = macd_histogram(series)
            if FeatureName.SENTIMENT in features:
                row["sentiment"] = self._sentiment(ticker)
            if FeatureName.FUNDAMENTALS in features:
                try:
                    row.update(self._fundamentals(ticker))
                except Exception:
                    skipped[ticker] = "fundamentals unavailable"
                    continue

            if any(np.isnan(v) for v in (row["volatility"], row["returns"])):
                skipped[ticker] = "could not compute volatility/returns"
                continue
            rows[ticker] = row

        matrix = pd.DataFrame.from_dict(rows, orient="index")
        # NaNs in optional columns (e.g. missing P/E) would break clustering.
        if not matrix.empty:
            matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0.0)
        return matrix, skipped
