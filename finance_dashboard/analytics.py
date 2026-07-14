"""Financial metric calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from finance_dashboard.config import ROLLING_WINDOW, TRADING_DAYS


def annualized_volatility(returns: pd.Series) -> float:
    """Scale daily return std-dev to an annual figure: σ_annual = σ_daily × √252."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std() * np.sqrt(TRADING_DAYS))


def period_pct_change(prices: pd.Series) -> float:
    """Total percentage change from first to last price in the series."""
    if prices.empty:
        return float("nan")
    first, last = prices.iloc[0], prices.iloc[-1]
    if not first or first == 0:
        return float("nan")
    return float((last / first - 1.0) * 100)


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns — standard input for correlation analysis."""
    return np.log(prices / prices.shift(1)).dropna()


def rolling_annualized_volatility(
    returns: pd.DataFrame,
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """Rolling window of daily std-dev, annualized."""
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
