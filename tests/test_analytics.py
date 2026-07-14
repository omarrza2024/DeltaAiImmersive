"""Tests for financial calculations."""

import numpy as np
import pandas as pd
import pytest

from finance_dashboard.analytics import (
    annualized_volatility,
    log_returns,
    period_pct_change,
    rolling_annualized_volatility,
)
from finance_dashboard.config import TRADING_DAYS


def test_annualized_volatility_known_series():
    # Constant daily returns → zero volatility.
    rets = pd.Series([0.01] * 30)
    assert annualized_volatility(rets) == pytest.approx(0.0)

    # Two points is enough for a std-dev.
    rets = pd.Series([0.01, -0.01])
    expected = rets.std() * np.sqrt(TRADING_DAYS)
    assert annualized_volatility(rets) == pytest.approx(expected)


def test_annualized_volatility_insufficient_data_returns_nan():
    assert np.isnan(annualized_volatility(pd.Series([0.01])))
    assert np.isnan(annualized_volatility(pd.Series(dtype=float)))


def test_period_pct_change():
    prices = pd.Series([100.0, 110.0])
    assert period_pct_change(prices) == pytest.approx(10.0)


def test_period_pct_change_zero_start_returns_nan():
    prices = pd.Series([0.0, 110.0])
    assert np.isnan(period_pct_change(prices))


def test_log_returns():
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]})
    lr = log_returns(prices)
    assert len(lr) == 2
    assert lr["A"].iloc[0] == pytest.approx(np.log(1.1))


def test_rolling_annualized_volatility_shape():
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    prices = pd.DataFrame({"A": np.linspace(100, 130, 30)}, index=idx)
    rets = prices.pct_change().dropna()
    roll = rolling_annualized_volatility(rets, window=5)
    assert roll.shape == rets.shape
    assert roll.columns.tolist() == ["A"]
