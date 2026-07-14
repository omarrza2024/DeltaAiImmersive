"""Domain models, enums, and input parsing."""

from __future__ import annotations

import re
from enum import Enum


class UserInputError(ValueError):
    """Raised when user-supplied inputs fail validation."""


class DashboardName(str, Enum):
    PRICE_METRICS = "Prices, % change & volatility"
    COMPARE_TWO = "Compare two stocks"
    CORRELATION = "Correlation matrix"
    RECOMMENDER = "Stock recommender"


def normalize_ticker(raw: str) -> str:
    """Strip whitespace and uppercase a single ticker symbol."""
    s = raw.strip().upper()
    return re.sub(r"\s+", "", s)


def parse_ticker_list(text: str) -> list[str]:
    """Split comma/semicolon/newline-separated input into normalized tickers."""
    parts = re.split(r"[\s,;]+", text.strip())
    return [normalize_ticker(p) for p in parts if p.strip()]


def dedupe_tickers(tickers: list[str]) -> list[str]:
    """Remove duplicates while preserving insertion order."""
    return list(dict.fromkeys(tickers))


def period_index(default_period: str, labels: list[str]) -> int:
    """Return selectbox index for a period label, falling back to default."""
    if default_period in labels:
        return labels.index(default_period)
    from finance_dashboard.config import DEFAULT_PERIOD_LABEL

    return labels.index(DEFAULT_PERIOD_LABEL) if DEFAULT_PERIOD_LABEL in labels else 0
