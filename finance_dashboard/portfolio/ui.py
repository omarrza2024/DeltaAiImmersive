"""Streamlit page for the user's portfolio — Bavest-inspired premium layout."""

from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from finance_dashboard.auth.db import get_db
from finance_dashboard.auth.ui import current_user
from finance_dashboard.charts import CHART_CONFIG, GAIN_COLOR, LOSS_COLOR
from finance_dashboard.data import company_name, fetch_history
from finance_dashboard.portfolio import analytics, store
from finance_dashboard.recommender.universe import ETF_FALLBACK_HOLDINGS, PREDEFINED_UNIVERSES


_POPULAR_TICKERS = sorted(
    {t for ts in ETF_FALLBACK_HOLDINGS.values() for t in ts}
    | {t for ts in PREDEFINED_UNIVERSES.values() for t in ts}
)


@st.cache_data(ttl=300, show_spinner=False)
def _current_price(ticker: str) -> float | None:
    try:
        prices = fetch_history((ticker,), "5d")
        if prices is None or prices.empty or ticker not in prices.columns:
            return None
        series = prices[ticker].dropna()
        return float(series.iloc[-1]) if not series.empty else None
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def _sector(ticker: str) -> str:
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        return info.get("sector") or "Unknown"
    except Exception:
        return "Unknown"


def get_enriched_for_user(db, username: str) -> dict:
    """Load + enrich the user's portfolio (used by both the UI and the AI context)."""
    portfolio = store.get_or_create(db, username)
    positions = portfolio.get("positions", [])
    tickers = [p["ticker"] for p in positions]
    prices = {t: _current_price(t) for t in tickers}
    sectors = {t: _sector(t) for t in tickers}
    return analytics.enrich(positions, prices, sectors)


def _render_header(enriched: dict) -> None:
    total = enriched["total_value"]
    pnl = enriched["total_pnl"]
    pnl_pct = enriched["total_pnl_pct"]
    pos_count = len(enriched["positions"])

    if pos_count == 0:
        st.markdown(
            '<div class="portfolio-hero empty">'
            '<div class="portfolio-hero-label">MY PORTFOLIO</div>'
            '<div class="portfolio-hero-value">$0.00</div>'
            '<div class="portfolio-hero-sub">No positions yet — add one below to get started.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    pnl_class = "pos" if pnl >= 0 else "neg"
    arrow = "↑" if pnl >= 0 else "↓"
    st.markdown(
        f'<div class="portfolio-hero">'
        f'<div class="portfolio-hero-label">MY PORTFOLIO</div>'
        f'<div class="portfolio-hero-value">${total:,.2f}</div>'
        f'<div class="portfolio-hero-delta {pnl_class}">'
        f'{arrow} ${abs(pnl):,.2f} <span class="pct">({pnl_pct:+.2f}%)</span>'
        f'</div>'
        f'<div class="portfolio-hero-sub">{pos_count} position(s) · unrealized</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_positions_table(enriched: dict) -> None:
    positions = enriched["positions"]
    if not positions:
        return

    df = pd.DataFrame([
        {
            "Ticker":      p["ticker"],
            "Company":     company_name(p["ticker"]) or "",
            "Sector":      p["sector"],
            "Shares":      p["shares"],
            "Avg Cost":    p["avg_cost"],
            "Price":       p["current_price"] if p["current_price"] is not None else 0.0,
            "Value":       p["market_value"],
            "P&L":         p["pnl"],
            "P&L %":       p["pnl_pct"],
            "Weight":      p["weight_pct"],
        }
        for p in positions
    ])
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker":   st.column_config.TextColumn(width="small"),
            "Company":  st.column_config.TextColumn(width="medium"),
            "Sector":   st.column_config.TextColumn(width="small"),
            "Shares":   st.column_config.NumberColumn(format="%.4g"),
            "Avg Cost": st.column_config.NumberColumn(format="$%.2f"),
            "Price":    st.column_config.NumberColumn(format="$%.2f"),
            "Value":    st.column_config.NumberColumn(format="$%.2f"),
            "P&L":      st.column_config.NumberColumn(format="$%+.2f"),
            "P&L %":    st.column_config.NumberColumn(format="%+.2f %%"),
            "Weight":   st.column_config.ProgressColumn(
                "Weight", min_value=0, max_value=100, format="%.1f%%", width="medium"
            ),
        },
    )


def _render_sector_donut(enriched: dict) -> None:
    alloc = enriched["sector_allocation"]
    if not alloc:
        return
    labels, values = zip(*sorted(alloc.items(), key=lambda kv: -kv[1]))
    fig = go.Figure(go.Pie(
        labels=list(labels),
        values=list(values),
        hole=0.62,
        marker=dict(colors=["#3B82F6", "#8B5CF6", "#00E676", "#FFB800", "#FF3B30", "#22D3EE"]),
        textinfo="percent",
        insidetextorientation="radial",
        hovertemplate="%{label}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        title="Sector allocation",
        height=320,
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(color="#E8ECEF", family="Inter, sans-serif"),
        margin=dict(l=8, r=8, t=48, b=8),
        legend=dict(orientation="v", x=1.02, y=0.5, font=dict(size=12)),
    )
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)


def _render_add_form(db, username: str) -> None:
    with st.expander("＋ Add position", expanded=False):
        with st.form("add_position_form", clear_on_submit=True):
            c1, c2, c3 = st.columns([2, 1, 1])
            ticker_choices = ["-- pick or type --"] + _POPULAR_TICKERS
            ticker_sel = c1.selectbox(
                "Ticker",
                options=_POPULAR_TICKERS,
                index=None,
                placeholder="Type or pick a ticker",
                accept_new_options=True,
                key="add_ticker",
            )
            shares = c2.number_input("Shares", min_value=0.0001, value=1.0, step=1.0,
                                     key="add_shares")
            avg_cost = c3.number_input("Avg cost / share ($)", min_value=0.0, value=100.0,
                                       step=1.0, key="add_avg_cost")
            submitted = st.form_submit_button("Add to portfolio")
            if submitted:
                if not ticker_sel:
                    st.error("Please pick or type a ticker.")
                else:
                    try:
                        store.add_position(db, username, ticker=ticker_sel,
                                           shares=shares, avg_cost=avg_cost)
                        st.success(f"Added {ticker_sel.upper()} to your portfolio.")
                        _current_price.clear()  # let the new position show a live price
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))


def _render_remove_controls(db, username: str, enriched: dict) -> None:
    if not enriched["positions"]:
        return
    with st.expander("Remove a position", expanded=False):
        tickers = [p["ticker"] for p in enriched["positions"]]
        selected = st.selectbox("Position to remove", tickers, key="remove_ticker")
        if st.button(f"Remove {selected}", key="remove_btn"):
            store.remove_position(db, username, selected)
            st.success(f"Removed {selected}.")
            st.rerun()


def render_portfolio_tab() -> None:
    db = get_db()
    username = current_user()
    if db is None or not username:
        st.info("Sign in and configure MongoDB to use the portfolio.")
        return

    # Inject portfolio-specific CSS once per render.
    st.markdown(_PORTFOLIO_CSS, unsafe_allow_html=True)

    with st.spinner("Loading positions and current prices…"):
        enriched = get_enriched_for_user(db, username)

    _render_header(enriched)

    if enriched["positions"]:
        col_table, col_donut = st.columns([3, 2])
        with col_table:
            _render_positions_table(enriched)
        with col_donut:
            _render_sector_donut(enriched)
    _render_add_form(db, username)
    _render_remove_controls(db, username, enriched)


_PORTFOLIO_CSS = """
<style>
.portfolio-hero {
    background: linear-gradient(135deg, #0E1015 0%, #000 100%);
    border: 1px solid #1E2027;
    border-left: 3px solid #3B82F6;
    border-radius: 6px;
    padding: 1.4rem 1.6rem;
    margin: 0 0 1.2rem 0;
}
.portfolio-hero.empty { border-left-color: #262A33; }
.portfolio-hero-label {
    color: #6B7280; font-size: 0.7rem; letter-spacing: 0.15em;
    font-weight: 600; text-transform: uppercase;
    font-family: "Inter", sans-serif;
}
.portfolio-hero-value {
    color: #E8ECEF; font-size: 2.8rem; font-weight: 700;
    letter-spacing: -0.02em; margin-top: 0.35rem;
    font-variant-numeric: tabular-nums;
    font-family: "Inter", sans-serif;
}
.portfolio-hero-delta {
    display: inline-flex; align-items: baseline; gap: 0.4rem;
    font-size: 1.1rem; font-weight: 600; margin-top: 0.3rem;
    font-variant-numeric: tabular-nums;
}
.portfolio-hero-delta.pos { color: #00E676; }
.portfolio-hero-delta.neg { color: #FF3B30; }
.portfolio-hero-delta .pct { font-weight: 500; opacity: 0.85; font-size: 0.95rem; }
.portfolio-hero-sub {
    color: #8A9BA8; font-size: 0.78rem; margin-top: 0.55rem;
    letter-spacing: 0.03em;
}
</style>
"""
