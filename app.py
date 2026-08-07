"""
Interactive Finance Dashboard — stock trends, volatility comparison, correlation.

Run with: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from finance_dashboard.config import DEFAULT_PERIOD_LABEL, PERIOD_OPTIONS
from finance_dashboard.controller import VisualizationController
from finance_dashboard.models import (
    DashboardName,
    UserInputError,
    dedupe_tickers,
    normalize_ticker,
    parse_ticker_list,
    period_index,
)
from finance_dashboard.assistant.ui import render_assistant_tab
from finance_dashboard.recommender.models import FeatureName, RecommendationRequest, StrategyName
from finance_dashboard.recommender.universe import ETF_FALLBACK_HOLDINGS, PREDEFINED_UNIVERSES

# Curated autocomplete pool for the correlation tab — every ticker we already
# know about, deduped and sorted. Users can still add anything else via
# multiselect's accept_new_options.
_POPULAR_TICKERS: list[str] = sorted(
    {t for ts in ETF_FALLBACK_HOLDINGS.values() for t in ts}
    | {t for ts in PREDEFINED_UNIVERSES.values() for t in ts}
)

# Human-readable labels for recommender controls.
_FEATURE_LABELS: dict[str, str] = {
    "Volatility": FeatureName.VOLATILITY.value,
    "Returns": FeatureName.RETURNS.value,
    "SMA": FeatureName.SMA.value,
    "MACD": FeatureName.MACD.value,
    "Fundamentals (beta, P/E)": FeatureName.FUNDAMENTALS.value,
    "Sentiment (experimental)": FeatureName.SENTIMENT.value,
}
_STRATEGY_LABELS: dict[str, str] = {
    "K-Means": StrategyName.KMEANS.value,
    "Hard-coded rules": StrategyName.RULES.value,
    "DBSCAN": StrategyName.DBSCAN.value,
}
# Grouping needs enough history for volatility/SMA/MACD; intraday ranges don't.
_RECOMMENDER_PERIODS = ["1 month", "1 year", "5 years"]


def _render_sidebar_defaults() -> str:
    """Sidebar holds shared defaults only; dashboards live in tabs."""
    with st.sidebar:
        st.subheader("Defaults")
        return st.selectbox(
            "Default time range",
            list(PERIOD_OPTIONS.keys()),
            index=period_index(DEFAULT_PERIOD_LABEL, list(PERIOD_OPTIONS.keys())),
            key="default_period",
        )


def _render_price_inputs(default_period: str) -> tuple[list[str], str]:
    ticker_raw = st.text_input("Ticker", value="AAPL", key="price_ticker")
    period_label = st.selectbox(
        "Time range",
        list(PERIOD_OPTIONS.keys()),
        index=period_index(default_period, list(PERIOD_OPTIONS.keys())),
        key="price_period",
    )
    return [normalize_ticker(ticker_raw)], period_label


def _render_compare_inputs(default_period: str) -> tuple[list[str], str, None]:
    st.markdown("Compare **annualized** and **rolling volatility** for two stocks.")
    c1, c2, c3 = st.columns(3)
    with c1:
        ta = st.text_input("Stock A", value="AAPL", key="compare_a")
    with c2:
        tb = st.text_input("Stock B", value="MSFT", key="compare_b")
    with c3:
        period_label = st.selectbox(
            "Time range",
            list(PERIOD_OPTIONS.keys()),
            index=period_index(default_period, list(PERIOD_OPTIONS.keys())),
            key="compare_period",
        )
    return [normalize_ticker(ta), normalize_ticker(tb)], period_label, None


def _render_correlation_inputs(default_period: str) -> tuple[list[str], str, None]:
    st.markdown(
        "Type to search tickers, click to add. Press Enter to add any symbol "
        "not in the list. Correlations use **daily log returns**."
    )
    selected = st.multiselect(
        "Tickers",
        options=_POPULAR_TICKERS,
        default=["AAPL", "MSFT", "GOOGL"],
        accept_new_options=True,
        key="corr_tickers",
    )
    period_label = st.selectbox(
        "Time range",
        list(PERIOD_OPTIONS.keys()),
        index=period_index(default_period, list(PERIOD_OPTIONS.keys())),
        key="corr_period",
    )
    return dedupe_tickers([normalize_ticker(t) for t in selected]), period_label, None


def _render_recommender_tab(controller: VisualizationController, default_period: str) -> None:
    st.markdown(
        "Group stocks into **risk buckets** (Low / Medium / High ±) from an ETF's "
        "holdings, predefined universes, and/or your own tickers."
    )
    with st.form("recommender_form"):
        c1, c2, c3 = st.columns(3)
        etf = c1.selectbox("ETF (optional)", ["None"] + sorted(ETF_FALLBACK_HOLDINGS))
        universes = c2.multiselect("Predefined universes", list(PREDEFINED_UNIVERSES))
        period_label = c3.selectbox(
            "Time range",
            _RECOMMENDER_PERIODS,
            index=period_index(default_period, _RECOMMENDER_PERIODS),
        )
        custom_text = st.text_input("Extra tickers (comma separated)", value="")
        features = st.multiselect(
            "Features", list(_FEATURE_LABELS), default=["Volatility", "Returns"]
        )
        submitted = st.form_submit_button("Get recommendations")

    if not submitted:
        st.info("Choose your sources and press **Get recommendations**.")
        return

    try:
        request = RecommendationRequest.from_user_input(
            etf=None if etf == "None" else etf,
            universe_names=universes,
            custom_tickers_text=custom_text,
            feature_names=[_FEATURE_LABELS[f] for f in features],
            strategy_name=StrategyName.KMEANS.value,
            period=PERIOD_OPTIONS[period_label],
        )
        controller.show_appropriate_dashboard(
            DashboardName.RECOMMENDER,
            tickers=[],
            period_label=period_label,
            request=request,
        )
    except UserInputError as exc:
        st.error(str(exc))


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
        /* Selectboxes: plain dropdown feel — pointer cursor, no caret, no typing.
           Multiselects keep typing enabled so autocomplete/search works. */
        div[data-baseweb="select"],
        div[data-baseweb="select"] * { cursor: pointer !important; }
        .stSelectbox div[data-baseweb="select"] input {
            caret-color: transparent !important;
            pointer-events: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Finance Dashboard")
    st.caption("Live market data via Yahoo Finance — for education only, not investment advice.")

    default_period = _render_sidebar_defaults()
    controller = VisualizationController()

    tab_price, tab_compare, tab_corr, tab_recommend, tab_assistant = st.tabs(
        ["Prices & volatility", "Compare two stocks", "Correlation", "Recommender", "Investment Assistant"]
    )

    with tab_price:
        tickers, period_label = _render_price_inputs(default_period)
        try:
            controller.show_appropriate_dashboard(
                DashboardName.PRICE_METRICS,
                tickers=tickers,
                period_label=period_label,
            )
        except UserInputError as exc:
            st.error(str(exc))

    with tab_compare:
        tickers, period_label, _ = _render_compare_inputs(default_period)
        try:
            controller.show_appropriate_dashboard(
                DashboardName.COMPARE_TWO,
                tickers=tickers,
                period_label=period_label,
            )
        except UserInputError as exc:
            st.error(str(exc))

    with tab_corr:
        tickers, period_label, _ = _render_correlation_inputs(default_period)
        try:
            controller.show_appropriate_dashboard(
                DashboardName.CORRELATION,
                tickers=tickers,
                period_label=period_label,
            )
        except UserInputError as exc:
            st.error(str(exc))

    with tab_recommend:
        _render_recommender_tab(controller, default_period)

    with tab_assistant:
        render_assistant_tab()


if __name__ == "__main__":
    main()
