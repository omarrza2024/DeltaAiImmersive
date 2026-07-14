"""Dashboard implementations — each renders one analysis view in Streamlit."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import streamlit as st

from finance_dashboard.analytics import (
    annualized_volatility,
    log_returns,
    period_pct_change,
    rolling_annualized_volatility,
)
from finance_dashboard.charts import (
    fig_correlation_heatmap,
    fig_pct_change,
    fig_price_history,
    fig_rolling_vol,
    fig_volatility_bar,
)
from finance_dashboard.config import PERIOD_OPTIONS, ROLLING_WINDOW, TRADING_DAYS
from finance_dashboard.data import fetch_history
from finance_dashboard.models import UserInputError


class BaseDashboard(ABC):
    """Shared contract: validate inputs, fetch data, render charts."""

    name: str

    @abstractmethod
    def validate(self, tickers: list[str], period_label: str, **kwargs) -> str:
        """Return the yfinance period code. Raise UserInputError on bad input."""

    @abstractmethod
    def display(self, tickers: list[str], period_label: str, **kwargs) -> None:
        """Render the dashboard in the current Streamlit context."""


class PriceMetricsDashboard(BaseDashboard):
    """Single-ticker view: prices, cumulative % change, and rolling volatility."""

    name = "Prices, % change & volatility"

    def validate(self, tickers: list[str], period_label: str, **kwargs) -> str:
        if not tickers or not tickers[0]:
            raise UserInputError("Enter a ticker symbol.")
        if period_label not in PERIOD_OPTIONS:
            raise UserInputError(f"Unknown time range: {period_label}")
        return PERIOD_OPTIONS[period_label]

    def display(self, tickers: list[str], period_label: str, **kwargs) -> None:
        period = self.validate(tickers, period_label)
        ticker = tickers[0]

        with st.spinner(f"Loading {ticker}…"):
            hist = fetch_history((ticker,), period)

        if hist.empty or ticker not in hist.columns:
            raise UserInputError(f"No data for **{ticker}**. Check the symbol and try again.")

        series = hist[ticker]
        chg = period_pct_change(series)
        rets = hist.pct_change().dropna()
        ann_vol = annualized_volatility(rets[ticker]) if not rets.empty else float("nan")

        m1, m2, m3 = st.columns(3)
        m1.metric("Last adj. close", f"{series.iloc[-1]:,.2f}")
        m2.metric("Period change", f"{chg:+.2f}%" if np.isfinite(chg) else "—")
        m3.metric("Ann. volatility", f"{ann_vol * 100:.2f}%" if np.isfinite(ann_vol) else "—")

        st.plotly_chart(
            fig_price_history(hist, f"{ticker} — price history"),
            use_container_width=True,
        )

        cum_pct = (hist / hist.iloc[0] - 1.0) * 100
        st.plotly_chart(
            fig_pct_change(cum_pct, f"{ticker} — cumulative % change"),
            use_container_width=True,
        )

        roll = rolling_annualized_volatility(rets)
        st.plotly_chart(
            fig_rolling_vol(roll, f"{ticker} — rolling volatility ({ROLLING_WINDOW}-day)"),
            use_container_width=True,
        )


class CompareTwoDashboard(BaseDashboard):
    """Side-by-side volatility comparison for two tickers."""

    name = "Compare two stocks"

    def validate(self, tickers: list[str], period_label: str, **kwargs) -> str:
        if len(tickers) < 2 or not tickers[0] or not tickers[1]:
            raise UserInputError("Enter both tickers.")
        if tickers[0] == tickers[1]:
            raise UserInputError("Choose two different tickers to compare.")
        if period_label not in PERIOD_OPTIONS:
            raise UserInputError(f"Unknown time range: {period_label}")
        return PERIOD_OPTIONS[period_label]

    def display(self, tickers: list[str], period_label: str, **kwargs) -> None:
        period = self.validate(tickers, period_label)
        a, b = tickers[0], tickers[1]

        with st.spinner("Loading prices…"):
            hist = fetch_history((a, b), period)

        if hist.empty or not {a, b}.issubset(set(hist.columns)):
            raise UserInputError("Could not load both series. Verify tickers and try again.")

        rets = hist.pct_change().dropna()
        vol_a = annualized_volatility(rets[a])
        vol_b = annualized_volatility(rets[b])
        roll = rolling_annualized_volatility(rets)

        c1, c2 = st.columns(2)
        c1.metric(f"{a} ann. volatility", f"{vol_a * 100:.2f}%" if np.isfinite(vol_a) else "—")
        c2.metric(f"{b} ann. volatility", f"{vol_b * 100:.2f}%" if np.isfinite(vol_b) else "—")

        st.plotly_chart(
            fig_rolling_vol(roll, f"Rolling volatility — {a} vs {b}"),
            use_container_width=True,
        )
        st.plotly_chart(
            fig_volatility_bar(
                [a, b],
                [vol_a * 100 if np.isfinite(vol_a) else np.nan, vol_b * 100 if np.isfinite(vol_b) else np.nan],
                f"Annualized volatility comparison (daily returns, √{TRADING_DAYS})",
            ),
            use_container_width=True,
        )


class CorrelationDashboard(BaseDashboard):
    """Heatmap of pairwise log-return correlations."""

    name = "Correlation matrix"

    def validate(self, tickers: list[str], period_label: str, **kwargs) -> str:
        if len(tickers) < 2:
            raise UserInputError("Enter at least two distinct tickers.")
        if period_label not in PERIOD_OPTIONS:
            raise UserInputError(f"Unknown time range: {period_label}")
        return PERIOD_OPTIONS[period_label]

    def display(self, tickers: list[str], period_label: str, **kwargs) -> None:
        period = self.validate(tickers, period_label)

        with st.spinner("Downloading series…"):
            hist = fetch_history(tuple(tickers), period)

        missing = [t for t in tickers if t not in hist.columns]
        if missing:
            raise UserInputError(f"No data for: {', '.join(missing)}")

        hist = hist[[c for c in tickers if c in hist.columns]]
        if hist.shape[1] < 2:
            raise UserInputError("Need at least two valid series for correlation.")

        log_rets = log_returns(hist)
        if len(log_rets) < 10:
            st.warning("Very few overlapping days; correlation may be unreliable.")

        corr = log_rets.corr()
        st.plotly_chart(fig_correlation_heatmap(corr), use_container_width=True)
        with st.expander("Numeric correlation matrix"):
            st.dataframe(corr.round(3), use_container_width=True)
