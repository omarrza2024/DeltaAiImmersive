"""
Interactive Finance Dashboard — stock trends, volatility comparison, correlation.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

TRADING_DAYS = 252
PERIOD_OPTIONS = {
    "1 month": "1mo",
    "3 months": "3mo",
    "6 months": "6mo",
    "1 year": "1y",
    "2 years": "2y",
    "5 years": "5y",
    "Max": "max",
}
ROLLING_WINDOW = 21


def normalize_ticker(raw: str) -> str:
    s = raw.strip().upper()
    s = re.sub(r"\s+", "", s)
    return s


def parse_ticker_list(text: str) -> list[str]:
    parts = re.split(r"[\s,;]+", text.strip())
    return [normalize_ticker(p) for p in parts if p.strip()]


@st.cache_data(ttl=300, show_spinner=False)
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
    if df.empty:
        return df
    if len(t) == 1:
        if isinstance(df.columns, pd.MultiIndex):
            out = df["Close"].copy()
        else:
            out = df[["Close"]].copy()
        out.columns = [t[0]]
        return out
    # Multiple tickers: columns are often MultiIndex (field, ticker)
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            out = df["Close"].copy()
        else:
            out = df.xs("Close", axis=1, level=0, drop_level=True)
    else:
        out = df[["Close"]].copy() if "Close" in df.columns else df.iloc[:, :1].copy()
    out = out.sort_index()
    return out


def annualized_volatility(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std() * np.sqrt(TRADING_DAYS))


def fig_price_history(closes: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for col in closes.columns:
        fig.add_trace(
            go.Scatter(
                x=closes.index,
                y=closes[col],
                mode="lines",
                name=col,
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price (adj. close)",
        hovermode="x unified",
        template="plotly_dark",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=48, r=24, t=56, b=48),
    )
    return fig


def fig_rolling_vol(rolling: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for col in rolling.columns:
        fig.add_trace(
            go.Scatter(
                x=rolling.index,
                y=rolling[col] * 100,
                mode="lines",
                name=col,
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Annualized vol. (%, 21-day rolling)",
        hovermode="x unified",
        template="plotly_dark",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=48, r=24, t=56, b=48),
    )
    return fig


def fig_correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="ρ"),
            hovertemplate="%{y} vs %{x}<br>ρ = %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Return correlation matrix",
        template="plotly_dark",
        height=max(360, 48 * len(corr)),
        margin=dict(l=80, r=24, t=56, b=80),
    )
    return fig


def main() -> None:
    st.set_page_config(
        page_title="Finance Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.25rem; }
        h1 { letter-spacing: -0.02em; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Finance Dashboard")
    st.caption("Live market data via Yahoo Finance — for education only, not investment advice.")

    with st.sidebar:
        st.subheader("Defaults")
        default_period_label = st.selectbox(
            "Default history range",
            list(PERIOD_OPTIONS.keys()),
            index=3,
        )
        default_period = PERIOD_OPTIONS[default_period_label]

    tab_trends, tab_vol, tab_corr = st.tabs(
        ["Stock trends", "Volatility compare", "Correlation"]
    )

    # --- Tab 1: single stock trends ---
    with tab_trends:
        c1, c2 = st.columns([1, 2])
        with c1:
            t_single = st.text_input("Ticker", value="AAPL", key="trend_ticker")
            period_t = st.selectbox(
                "Period",
                list(PERIOD_OPTIONS.keys()),
                index=list(PERIOD_OPTIONS.values()).index(default_period)
                if default_period in PERIOD_OPTIONS.values()
                else 3,
                key="trend_period",
            )
        ticker = normalize_ticker(t_single)
        period_key = PERIOD_OPTIONS[period_t]

        if not ticker:
            st.warning("Enter a ticker symbol.")
        else:
            with st.spinner(f"Loading {ticker}…"):
                hist = fetch_history((ticker,), period_key)
            if hist.empty or ticker not in hist.columns:
                st.error(f"No data for **{ticker}**. Check the symbol and try again.")
            else:
                last = hist[ticker].iloc[-1]
                first = hist[ticker].iloc[0]
                chg = (last / first - 1.0) * 100 if first and first != 0 else float("nan")
                m1, m2, m3 = st.columns(3)
                m1.metric("Last adj. close", f"{last:,.2f}")
                m2.metric("Period change", f"{chg:+.2f}%" if np.isfinite(chg) else "—")
                m3.metric("Trading days", len(hist))

                fig = fig_price_history(hist, f"{ticker} — price history")
                st.plotly_chart(fig, use_container_width=True)

    # --- Tab 2: two-stock volatility ---
    with tab_vol:
        st.markdown("Pick two tickers to compare **annualized volatility** and **rolling volatility**.")
        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            ta = st.text_input("Stock A", value="AAPL", key="vol_a")
        with vc2:
            tb = st.text_input("Stock B", value="MSFT", key="vol_b")
        with vc3:
            period_v = st.selectbox(
                "Period",
                list(PERIOD_OPTIONS.keys()),
                index=list(PERIOD_OPTIONS.values()).index(default_period)
                if default_period in PERIOD_OPTIONS.values()
                else 3,
                key="vol_period",
            )

        a, b = normalize_ticker(ta), normalize_ticker(tb)
        pv = PERIOD_OPTIONS[period_v]

        if not a or not b:
            st.warning("Enter both tickers.")
        elif a == b:
            st.warning("Choose two different tickers to compare.")
        else:
            with st.spinner("Loading prices…"):
                h2 = fetch_history((a, b), pv)
            if h2.empty or not {a, b}.issubset(set(h2.columns)):
                st.error("Could not load both series. Verify tickers and try again.")
            else:
                rets = h2.pct_change().dropna()
                vol_a = annualized_volatility(rets[a])
                vol_b = annualized_volatility(rets[b])
                roll = rets.rolling(ROLLING_WINDOW).std() * np.sqrt(TRADING_DAYS)

                b1, b2 = st.columns(2)
                b1.metric(f"{a} ann. volatility", f"{vol_a * 100:.2f}%" if np.isfinite(vol_a) else "—")
                b2.metric(f"{b} ann. volatility", f"{vol_b * 100:.2f}%" if np.isfinite(vol_b) else "—")

                st.plotly_chart(
                    fig_rolling_vol(roll, f"Rolling volatility — {a} vs {b}"),
                    use_container_width=True,
                )

                comp = pd.DataFrame(
                    {
                        "Ticker": [a, b],
                        "Annualized volatility (%)": [
                            vol_a * 100 if np.isfinite(vol_a) else np.nan,
                            vol_b * 100 if np.isfinite(vol_b) else np.nan,
                        ],
                    }
                )
                fig_bar = go.Figure(
                    data=go.Bar(
                        x=comp["Ticker"],
                        y=comp["Annualized volatility (%)"],
                        marker_color=["#636EFA", "#EF553B"],
                        text=[f"{v:.2f}%" if np.isfinite(v) else "—" for v in comp["Annualized volatility (%)"]],
                        textposition="outside",
                    )
                )
                fig_bar.update_layout(
                    title="Annualized volatility comparison (daily returns, √252)",
                    template="plotly_dark",
                    yaxis_title="Volatility (%)",
                    height=360,
                    showlegend=False,
                    margin=dict(l=48, r=24, t=56, b=48),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

    # --- Tab 3: correlation ---
    with tab_corr:
        st.markdown(
            "Add tickers (comma or newline separated). Correlations use **daily log returns** over the selected period."
        )
        tickers_text = st.text_area(
            "Tickers",
            value="AAPL, MSFT, GOOGL",
            height=100,
            key="corr_text",
        )
        period_c = st.selectbox(
            "Period",
            list(PERIOD_OPTIONS.keys()),
            index=list(PERIOD_OPTIONS.values()).index(default_period)
            if default_period in PERIOD_OPTIONS.values()
            else 3,
            key="corr_period",
        )
        tickers = parse_ticker_list(tickers_text)
        tickers = list(dict.fromkeys(tickers))
        pc = PERIOD_OPTIONS[period_c]

        if len(tickers) < 2:
            st.warning("Enter at least two distinct tickers.")
        else:
            with st.spinner("Downloading series…"):
                hc = fetch_history(tuple(tickers), pc)
            missing = [t for t in tickers if t not in hc.columns]
            if missing:
                st.error(f"No data for: {', '.join(missing)}")
            hc = hc[[c for c in tickers if c in hc.columns]]
            if hc.shape[1] < 2:
                st.error("Need at least two valid series for correlation.")
            else:
                log_rets = np.log(hc / hc.shift(1)).dropna()
                if len(log_rets) < 10:
                    st.warning("Very few overlapping days; correlation may be unreliable.")
                corr = log_rets.corr()
                st.plotly_chart(fig_correlation_heatmap(corr), use_container_width=True)
                with st.expander("Numeric correlation matrix"):
                    st.dataframe(corr.round(3), use_container_width=True)


if __name__ == "__main__":
    main()
