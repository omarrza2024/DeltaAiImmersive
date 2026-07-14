"""Shared constants and configuration."""

TRADING_DAYS = 252
ROLLING_WINDOW = 21
CACHE_TTL_SECONDS = 300

# LLD time ranges mapped to yfinance `period` values.
PERIOD_OPTIONS: dict[str, str] = {
    "1 day": "1d",
    "1 week": "5d",
    "1 month": "1mo",
    "1 year": "1y",
    "5 years": "5y",
}

DEFAULT_PERIOD_LABEL = "1 year"
