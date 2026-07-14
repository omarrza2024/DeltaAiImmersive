"""Market data fetching and normalization."""

from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf

from finance_dashboard.config import CACHE_TTL_SECONDS


def extract_close_prices(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Normalize a yfinance download into a date-indexed DataFrame of adj. close prices.

    yfinance column layout differs for single vs. multiple tickers.
    """
    if df.empty:
        return df

    if len(tickers) == 1:
        if isinstance(df.columns, pd.MultiIndex):
            out = df["Close"].copy()
        else:
            out = df[["Close"]].copy()
        out.columns = [tickers[0]]
        return out

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            out = df["Close"].copy()
        else:
            out = df.xs("Close", axis=1, level=0, drop_level=True)
    else:
        out = df[["Close"]].copy() if "Close" in df.columns else df.iloc[:, :1].copy()
    return out.sort_index()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_history(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    """Download adjusted close history for one or more tickers."""
    if not tickers:
        return pd.DataFrame()

    t = list(tickers)
    df = yf.download(
        t,
        period=period,
        progress=False,
        auto_adjust=True,
        threads=True,
    )
    return extract_close_prices(df, t)
