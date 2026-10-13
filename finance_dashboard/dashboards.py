"""Dashboard implementations — each renders one analysis view in Streamlit."""

from __future__ import annotations

import html
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
from finance_dashboard.data import company_name, fetch_history
from finance_dashboard.models import UserInputError


def _stock_header(ticker: str, period_label: str) -> None:
    """Render a two-line header: ticker + company name, then period caption."""
    name = company_name(ticker) or ""
    st.markdown(
        f'<div class="stock-header">'
        f'<span class="stock-ticker">{html.escape(ticker)}</span>'
        f'<span class="stock-name">{html.escape(name)}</span>'
        f'</div>'
        f'<div class="stock-sub">{html.escape(period_label)}</div>',
        unsafe_allow_html=True,
    )


def _kpi(label: str, value: str, tone: str = "") -> str:
    """Return an HTML string for one KPI tile — used inside a chip-row layout."""
    return (
        f'<div class="kpi {tone}">'
        f'<div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'</div>'
    )


def _render_kpis(kpis: list[str]) -> None:
    st.markdown(f'<div class="kpi-row">{"".join(kpis)}</div>', unsafe_allow_html=True)


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
        # Optional per-caller prefix so the same ticker rendered from multiple
        # tabs (e.g. Prices + Watchlist detail) doesn't collide on plotly keys.
        key_prefix = kwargs.get("key_prefix", "price-")

        with st.spinner(f"Loading {ticker}…"):
            hist = fetch_history((ticker,), period)

        if hist.empty or ticker not in hist.columns:
            raise UserInputError(f"No data for **{ticker}**. Check the symbol and try again.")

        series = hist[ticker]
        chg = period_pct_change(series)
        rets = hist.pct_change().dropna()
        ann_vol = annualized_volatility(rets[ticker]) if not rets.empty else float("nan")

        _stock_header(ticker, period_label)
        tone = "pos" if np.isfinite(chg) and chg >= 0 else "neg"
        _render_kpis([
            _kpi("Last adj. close", f"{series.iloc[-1]:,.2f}"),
            _kpi("Period change", f"{chg:+.2f}%" if np.isfinite(chg) else "—", tone=tone),
            _kpi("Ann. volatility", f"{ann_vol * 100:.2f}%" if np.isfinite(ann_vol) else "—"),
        ])

        st.plotly_chart(
            fig_price_history(hist, f"{ticker} — price history"),
            use_container_width=True,
            config={"displayModeBar": False},
            key=f"{key_prefix}hist-{ticker}",
        )

        cum_pct = (hist / hist.iloc[0] - 1.0) * 100
        st.plotly_chart(
            fig_pct_change(cum_pct, f"{ticker} — cumulative % change"),
            use_container_width=True,
            config={"displayModeBar": False},
            key=f"{key_prefix}pct-{ticker}",
        )

        roll = rolling_annualized_volatility(rets)
        st.plotly_chart(
            fig_rolling_vol(roll, f"{ticker} — rolling volatility ({ROLLING_WINDOW}-day)"),
            use_container_width=True,
            config={"displayModeBar": False},
            key=f"{key_prefix}vol-{ticker}",
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
        with c1:
            _stock_header(a, period_label)
            _render_kpis([_kpi("Ann. volatility", f"{vol_a * 100:.2f}%" if np.isfinite(vol_a) else "—")])
        with c2:
            _stock_header(b, period_label)
            _render_kpis([_kpi("Ann. volatility", f"{vol_b * 100:.2f}%" if np.isfinite(vol_b) else "—")])

        st.plotly_chart(
            fig_rolling_vol(roll, f"Rolling volatility — {a} vs {b}"),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.plotly_chart(
            fig_volatility_bar(
                [a, b],
                [vol_a * 100 if np.isfinite(vol_a) else np.nan, vol_b * 100 if np.isfinite(vol_b) else np.nan],
                f"Annualized volatility comparison (daily returns, √{TRADING_DAYS})",
            ),
            use_container_width=True,
            config={"displayModeBar": False},
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
        _render_kpis([
            _kpi("Tickers", str(hist.shape[1])),
            _kpi("Trading days", f"{len(log_rets):,}"),
            _kpi("Time range", period_label),
        ])
        st.plotly_chart(fig_correlation_heatmap(corr), use_container_width=True, config={"displayModeBar": False})
        with st.expander("Numeric correlation matrix"):
            st.dataframe(corr.round(3), use_container_width=True)
