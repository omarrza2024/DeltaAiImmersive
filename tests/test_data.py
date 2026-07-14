"""Tests for yfinance response normalization (no network)."""

import pandas as pd

from finance_dashboard.data import extract_close_prices


def test_extract_close_prices_single_ticker_flat_columns():
    df = pd.DataFrame({"Close": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2))
    out = extract_close_prices(df, ["AAPL"])
    assert list(out.columns) == ["AAPL"]
    assert out["AAPL"].tolist() == [1.0, 2.0]


def test_extract_close_prices_single_ticker_multiindex():
    cols = pd.MultiIndex.from_tuples([("Close", "AAPL"), ("Volume", "AAPL")])
    df = pd.DataFrame({cols[0]: [1.0, 2.0], cols[1]: [100, 200]}, index=pd.date_range("2024-01-01", periods=2))
    out = extract_close_prices(df, ["AAPL"])
    assert list(out.columns) == ["AAPL"]


def test_extract_close_prices_multiple_tickers():
    cols = pd.MultiIndex.from_product([["Close"], ["AAPL", "MSFT"]])
    df = pd.DataFrame(
        {("Close", "AAPL"): [1.0, 2.0], ("Close", "MSFT"): [10.0, 20.0]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    out = extract_close_prices(df, ["AAPL", "MSFT"])
    assert set(out.columns) == {"AAPL", "MSFT"}
