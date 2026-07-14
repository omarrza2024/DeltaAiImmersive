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
from finance_dashboard.recommender.models import FeatureName, RecommendationRequest, StrategyName
from finance_dashboard.recommender.universe import ETF_FALLBACK_HOLDINGS, PREDEFINED_UNIVERSES

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
        "Enter tickers (comma or newline separated). Correlations use **daily log returns**."
    )
    tickers_text = st.text_area("Tickers", value="AAPL, MSFT, GOOGL", height=100, key="corr_text")
    period_label = st.selectbox(
        "Time range",
        list(PERIOD_OPTIONS.keys()),
        index=period_index(default_period, list(PERIOD_OPTIONS.keys())),
        key="corr_period",
    )
    return dedupe_tickers(parse_ticker_list(tickers_text)), period_label, None


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
        c4, c5 = st.columns(2)
        features = c4.multiselect(
            "Features", list(_FEATURE_LABELS), default=["Volatility", "Returns"]
        )
        strategy = c5.selectbox("Grouping strategy", list(_STRATEGY_LABELS))
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
            strategy_name=_STRATEGY_LABELS[strategy],
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
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Finance Dashboard")
    st.caption("Live market data via Yahoo Finance — for education only, not investment advice.")

    default_period = _render_sidebar_defaults()
    controller = VisualizationController()

    tab_price, tab_compare, tab_corr, tab_recommend = st.tabs(
        ["Prices & volatility", "Compare two stocks", "Correlation", "Recommender"]
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


if __name__ == "__main__":
    main()
