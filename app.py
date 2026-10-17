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
from finance_dashboard.auth.ui import (
    is_guest,
    render_sidebar_account_panel,
    require_login,
)
from finance_dashboard.widgets import (
    POPULAR_TICKERS,
    etf_from_label,
    label_for_etf,
    label_for_ticker,
    ticker_from_label,
    ticker_labels,
    ticker_prefix_autocomplete,
    ticker_select,
)

# Features that cost API credits or depend on third-party services. For
# guests these appear in the UI but can't be selected — a subtle sign-up
# nudge explains why. Signing in unlocks them.
_API_GATED_FEATURES = {"Sentiment"}
from finance_dashboard.watchlist.ui import render_watchlist_tab
from finance_dashboard.recommender.models import FeatureName, RecommendationRequest, StrategyName
from finance_dashboard.recommender.universe import ETF_FALLBACK_HOLDINGS, PREDEFINED_UNIVERSES

# Autocomplete pool is sourced from `finance_dashboard.widgets.POPULAR_TICKERS`
# so Prices, Compare, Correlation and Watchlist all see the same ticker list.

# Human-readable labels for recommender controls.
_FEATURE_LABELS: dict[str, str] = {
    "Volatility": FeatureName.VOLATILITY.value,
    "Returns": FeatureName.RETURNS.value,
    "SMA": FeatureName.SMA.value,
    "MACD": FeatureName.MACD.value,
    "Fundamentals (SEC DATA REPORTS)": FeatureName.FUNDAMENTALS.value,
    "Sentiment": FeatureName.SENTIMENT.value,
}
_STRATEGY_LABELS: dict[str, str] = {
    "K-Means": StrategyName.KMEANS.value,
    "Hard-coded rules": StrategyName.RULES.value,
    "DBSCAN": StrategyName.DBSCAN.value,
}
# Grouping needs enough history for volatility/SMA/MACD; intraday ranges don't.
_RECOMMENDER_PERIODS = ["1 month", "1 year", "5 years"]


def _render_default_period_selector() -> str:
    with st.sidebar:
        st.divider()
        st.subheader("Defaults")
        return st.selectbox(
            "Default time range",
            list(PERIOD_OPTIONS.keys()),
            index=period_index(DEFAULT_PERIOD_LABEL, list(PERIOD_OPTIONS.keys())),
            key="default_period",
        )


def _render_price_inputs(default_period: str) -> tuple[list[str], str]:
    # Handoff from the Watchlist tab: when the user clicks a ticker row there,
    # seed the Prices ticker picker BEFORE it instantiates so the clicked
    # ticker shows up in the dropdown.
    pending = st.session_state.pop("_wl_pending_ticker", None)
    if pending:
        st.session_state["price_ticker_sel"] = [label_for_ticker(pending)]
    ticker_raw = ticker_select(
        "Ticker", key="price_ticker_sel", default="AAPL",
    )
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
        ta = ticker_select("Stock A", key="compare_a_sel", default="AAPL")
    with c2:
        tb = ticker_select("Stock B", key="compare_b_sel", default="MSFT")
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
    # Options show the company name alongside the ticker. ticker_from_label
    # strips the suffix back off when we hand tickers to the dashboard.
    # Guard: Streamlit requires every `default` value to already be in
    # `options`, so we prepend any missing default labels (can happen when
    # ticker_labels()'s cache warmed with a bare ticker before yfinance
    # could resolve the company name).
    options = list(ticker_labels())
    default_labels = [label_for_ticker(t) for t in ("AAPL", "MSFT", "GOOGL")]
    for lbl in default_labels:
        if lbl not in options:
            options.append(lbl)
    selected = st.multiselect(
        "Tickers",
        options=options,
        default=default_labels,
        accept_new_options=True,
        key="corr_tickers",
    )
    period_label = st.selectbox(
        "Time range",
        list(PERIOD_OPTIONS.keys()),
        index=period_index(default_period, list(PERIOD_OPTIONS.keys())),
        key="corr_period",
    )
    tickers = [normalize_ticker(ticker_from_label(label)) for label in selected]
    return dedupe_tickers(tickers), period_label, None


def _render_recommender_tab(controller: VisualizationController, default_period: str) -> None:
    st.caption(
        "All three sources below are **optional** — but pick **at least one** so "
        "there are stocks to analyze."
    )

    # Features selector OUTSIDE the form so the time-range widget can hide
    # itself when the user picks only time-independent features (Fundamentals
    # or Sentiment — those read SEC filings / news, which aren't period-based).
    #
    # For guests, API-gated features (Sentiment) are stripped from the list —
    # they appear as a disabled note instead so a random visitor can't burn
    # the marketaux quota. A subtle sign-up nudge sits next to the control.
    guest = is_guest()
    available_labels = [
        label for label in _FEATURE_LABELS
        if not (guest and label in _API_GATED_FEATURES)
    ]
    features = st.multiselect(
        "Features", available_labels, default=["Volatility", "Returns"],
        key="recommender_features",
    )
    if guest:
        locked = ", ".join(sorted(_API_GATED_FEATURES))
        lock_c1, lock_c2 = st.columns([5, 1])
        lock_c1.caption(
            f"🔒 **{locked}** uses a live news API — sign in to unlock it."
        )
        if lock_c2.button("Sign in", key="sentiment_signin_cta",
                          use_container_width=True):
            st.session_state["_force_signin"] = True
            st.rerun()
    _TIME_DEPENDENT = {"Volatility", "Returns", "SMA", "MACD"}
    needs_time_range = any(f in _TIME_DEPENDENT for f in features)

    # ETF options show the fund's full name alongside the ticker so a user
    # doesn't have to recognize three-letter ETF symbols by heart.
    etf_options = ["None"] + [label_for_etf(e) for e in sorted(ETF_FALLBACK_HOLDINGS)]

    with st.form("recommender_form"):
        if needs_time_range:
            c1, c2 = st.columns(2)
            etf_label = c1.selectbox("ETF (optional)", etf_options)
            period_label = c2.selectbox(
                "Time range",
                _RECOMMENDER_PERIODS,
                index=period_index(default_period, _RECOMMENDER_PERIODS),
            )
        else:
            etf_label = st.selectbox("ETF (optional)", etf_options)
            # Time range is irrelevant for fundamentals/sentiment-only runs,
            # but the recommender request still needs a value — pass the
            # user's default so nothing downstream breaks.
            period_label = default_period if default_period in _RECOMMENDER_PERIODS else "1 year"

        # Predefined universes: pills instead of a multiselect dropdown.
        # Each sector is a clickable pill — click to toggle on/off. No
        # typing, no "Choose options" placeholder confusion, and every
        # option is visible at a glance.
        universes = st.pills(
            "Predefined universes (optional)",
            options=list(PREDEFINED_UNIVERSES),
            selection_mode="multi",
            default=[],
            key="recommender_universes",
        ) or []

        # Extra tickers: searchable multiselect with company-name labels,
        # matching the Correlation and Compare UX. accept_new_options=True
        # lets a user still add an unknown symbol by typing it + Enter.
        extra_selected = st.multiselect(
            "Extra tickers (optional)",
            options=ticker_labels(),
            default=[],
            accept_new_options=True,
            key="recommender_extra_tickers",
            placeholder="Type a letter to search (e.g. A for AAPL, AMD...)",
        )
        # Convert selections back to a comma-separated ticker string that the
        # existing RecommendationRequest.from_user_input() parser understands.
        custom_text = ",".join(ticker_from_label(lbl) for lbl in extra_selected)

        submitted = st.form_submit_button("Get recommendations")

    if not submitted:
        st.info("Choose your sources and press **Get recommendations**.")
        return

    etf = etf_from_label(etf_label)

    try:
        request = RecommendationRequest.from_user_input(
            etf=etf,
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
        page_title="TradeMatrix",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* === FINANCE TYPOGRAPHY === Inter everywhere, tabular-nums on data. */
        html, body, [class*="css"], .stApp, .stMarkdown, .stTextInput, .stSelectbox,
        .stMultiSelect, .stDataFrame, .stChatInput, .stButton, .stExpander {
            font-family: "Inter", -apple-system, "Segoe UI", sans-serif !important;
            font-size: 14px !important;
        }
        .stApp { background: #000000; color: #E8ECEF; font-feature-settings: "ss01", "cv11"; }

        /* Tabular figures on any numeric surface — clean alignment without a coding feel. */
        .number, .ticker, .kpi-value, .rec-ticker, .match-bar .val,
        .stDataFrame [role="gridcell"], .stDataFrame [role="columnheader"] {
            font-variant-numeric: tabular-nums;
        }

        /* === TERMINAL HEADER === */
        .block-container { padding-top: 3rem; padding-bottom: 3rem; max-width: 1400px; }
        .term-header {
            display:flex; align-items:baseline; justify-content:space-between;
            padding:0.55rem 0.95rem; margin:0 0 1rem;
            background:linear-gradient(90deg,#0E1015 0%,#000 100%);
            border:1px solid #1E2027; border-left:3px solid var(--accent, #3B82F6);
        }
        .term-title {
            font-family:"Inter", sans-serif !important;
            font-size:1.15rem; font-weight:800; letter-spacing:0.18em;
            color:#E8ECEF; text-transform:uppercase;
        }
        .term-title span.slash { color:var(--accent, #3B82F6); font-weight:600; }
        .term-meta {
            font-family:"Inter", sans-serif !important;
            font-size:0.72rem; color:#6B7280; letter-spacing:0.1em; text-transform:uppercase;
        }
        .term-meta span.dot { color:#22C55E; margin-right:0.3rem; }

        /* === TABS: full-width, uppercase, no overlap ===
           Streamlit's own stylesheet sets display: flex and some tab padding
           on the tab-list; we override hard with max specificity + important.
           Target every variant Streamlit has shipped: [data-baseweb="tab"],
           button[role="tab"], and [data-testid="stTabs"] descendants.

           Two things that previously failed on the deployed app:
           1) `overflow: hidden` on the tablist, so tabs can't horizontally
              scroll out of view and flex:1 actually spreads them.
           2) styling the inner <p> — Streamlit wraps every tab label in a
              <p> whose own typography was swallowing the uppercase/weight
              rules set on the <button> itself. */
        div.stTabs,
        div[data-testid="stTabs"] { width:100% !important; }

        /* CSS Grid instead of flex — grid-auto-flow:column + grid-auto-columns:1fr
           makes every tab share the full width equally regardless of how many
           tabs there are. More reliable than flex:1, which Streamlit's own
           min-width rules can defeat. html body prefix maxes specificity so
           our rules beat Streamlit's injected stylesheet. */
        html body div.stTabs div[data-baseweb="tab-list"],
        html body div[data-testid="stTabs"] div[data-baseweb="tab-list"],
        html body div[data-testid="stTabs"] div[role="tablist"] {
            display:grid !important;
            grid-auto-flow:column !important;
            grid-auto-columns:1fr !important;
            width:100% !important;
            gap:0 !important;
            border-bottom:1px solid #1E2027 !important;
            background:transparent !important;
            overflow:hidden !important;
        }

        html body div.stTabs button[data-baseweb="tab"],
        html body div.stTabs button[role="tab"],
        html body div[data-testid="stTabs"] button[data-baseweb="tab"],
        html body div[data-testid="stTabs"] button[role="tab"] {
            width:100% !important;
            min-width:0 !important;
            max-width:none !important;
            display:flex !important;
            align-items:center !important;
            justify-content:center !important;
            text-align:center !important;
            padding:1rem 0.75rem !important;
            margin:0 !important;
            color:#6B7280 !important;
            background:transparent !important;
            border:none !important;
            border-right:1px solid #1E2027 !important;
            border-top:2px solid transparent !important;
            border-radius:0 !important;
            box-shadow:none !important;
            cursor:pointer !important;
        }
        div.stTabs button[data-baseweb="tab"]:last-child,
        div.stTabs button[role="tab"]:last-child,
        div[data-testid="stTabs"] button[data-baseweb="tab"]:last-child,
        div[data-testid="stTabs"] button[role="tab"]:last-child {
            border-right:none !important;
        }

        /* Streamlit wraps each tab label in a <p>; its own typography beats
           anything set on the <button>, so we restate font rules here too. */
        div.stTabs button[data-baseweb="tab"] p,
        div.stTabs button[role="tab"] p,
        div[data-testid="stTabs"] button[data-baseweb="tab"] p,
        div[data-testid="stTabs"] button[role="tab"] p,
        div.stTabs button[data-baseweb="tab"] div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stTabs"] button[role="tab"] div[data-testid="stMarkdownContainer"] p {
            font-family:"Inter", sans-serif !important;
            font-size:0.9rem !important;
            font-weight:600 !important;
            letter-spacing:0.07em !important;
            text-transform:uppercase !important;
            text-align:center !important;
            line-height:1.2 !important;
            margin:0 !important;
            padding:0 !important;
            color:inherit !important;
            white-space:nowrap !important;
            overflow:hidden !important;
            text-overflow:ellipsis !important;
        }

        div.stTabs button[data-baseweb="tab"]:hover,
        div.stTabs button[role="tab"]:hover,
        div[data-testid="stTabs"] button[data-baseweb="tab"]:hover,
        div[data-testid="stTabs"] button[role="tab"]:hover {
            color:#E8ECEF !important;
            background:#0E1015 !important;
        }

        div.stTabs button[aria-selected="true"],
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color:#3B82F6 !important;
            background:#0E1015 !important;
            border-top-color:#3B82F6 !important;
        }
        div.stTabs button[aria-selected="true"] p,
        div[data-testid="stTabs"] button[aria-selected="true"] p {
            color:#3B82F6 !important;
        }

        /* Kill the baseweb underline sliders Streamlit draws under the tabs. */
        div.stTabs [data-baseweb="tab-highlight"],
        div.stTabs [data-baseweb="tab-border"],
        div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
        div[data-testid="stTabs"] [data-baseweb="tab-border"] {
            display:none !important;
        }

        /* Form submit buttons (Log in / Create account) — full-width, taller,
           primary-CTA feel. The nested <p> again carries its own font rules. */
        div[data-testid="stFormSubmitButton"],
        div[data-testid="stFormSubmitButton"] > button {
            width:100% !important;
        }
        div[data-testid="stFormSubmitButton"] > button {
            padding:0.85rem 1.5rem !important;
            min-height:3rem !important;
            font-size:1rem !important;
            font-weight:600 !important;
            border-radius:8px !important;
        }
        div[data-testid="stFormSubmitButton"] > button p {
            font-size:1rem !important;
            font-weight:600 !important;
            margin:0 !important;
        }

        /* === HEADINGS === */
        h1, h2, h3, h4 {
            font-family:"Inter", sans-serif !important;
            font-weight:600 !important;
            letter-spacing:-0.01em !important;
        }
        h2 { font-size:1.1rem !important; text-transform:uppercase; letter-spacing:0.08em !important;
             color:#E8ECEF; border-bottom:1px solid #1E2027; padding-bottom:0.4rem;
             margin-top:1.5rem !important; }

        /* === INPUTS: softer, more product-like === */
        .stTextInput input, .stTextArea textarea {
            background:#0E1015 !important; border:1px solid #262A33 !important;
            border-radius:8px !important; color:#E8ECEF !important;
            padding:0.55rem 0.75rem !important;
            font-size:14px !important;
        }
        div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
            background:#0E1015 !important; border:1px solid #262A33 !important;
            border-radius:8px !important; color:#E8ECEF !important;
            font-size:14px !important;
        }
        .stTextInput input:focus, div[data-baseweb="select"]:focus-within > div,
        .stTextArea textarea:focus {
            border-color:var(--accent, #3B82F6) !important;
            box-shadow:0 0 0 3px color-mix(in srgb, var(--accent, #3B82F6) 20%, transparent) !important;
            outline:none !important;
        }

        /* Selectboxes + multiselects: click ANYWHERE on the bar to open the
           dropdown (not just the chevron). The old rule set pointer-events
           to none on the input, which meant only the arrow was clickable —
           that broke the natural "click the field" interaction users expect.
           We still hide the caret and disable text selection, so single
           selects read as click-to-pick instead of inviting typing. */
        [data-testid="stSelectbox"] div[data-baseweb="select"],
        [data-testid="stSelectbox"] div[data-baseweb="select"] *,
        [data-testid="stMultiSelect"] div[data-baseweb="select"],
        [data-testid="stMultiSelect"] div[data-baseweb="select"] * {
            cursor: pointer !important;
        }
        [data-testid="stSelectbox"] input {
            caret-color: transparent !important;
            user-select: none !important;
            cursor: pointer !important;
        }

        /* Multiselect chips: amber tinted. */
        div[data-baseweb="select"] [data-baseweb="tag"] {
            background:color-mix(in srgb, var(--accent, #3B82F6) 16%, transparent) !important;
            color:#93C5FD !important;
            border:1px solid color-mix(in srgb, var(--accent, #3B82F6) 30%, transparent) !important;
            border-radius:2px !important;
            font-family:"Inter", sans-serif !important;
            font-size:0.72rem !important; font-weight:600 !important;
            letter-spacing:0.05em !important; text-transform:uppercase;
        }

        /* === BUTTONS: filled, product-app feel === */
        .stButton > button, .stFormSubmitButton > button {
            background:var(--accent, #3B82F6) !important;
            color:#FFFFFF !important;
            border:1px solid var(--accent, #3B82F6) !important;
            border-radius:8px !important;
            font-family:"Inter", sans-serif !important;
            font-size:0.85rem !important; font-weight:600 !important;
            letter-spacing:0.01em !important; text-transform:none !important;
            padding:0.5rem 1.2rem !important;
            transition:filter 120ms ease;
        }
        .stButton > button:hover, .stFormSubmitButton > button:hover {
            filter:brightness(1.1) !important;
        }

        /* === CONTAINERS: sharp corners, dark card === */
        [data-testid="stForm"], .st-emotion-cache-r421ms {
            background:#0A0C10 !important;
            border:1px solid #1E2027 !important;
            border-radius:2px !important;
        }
        [data-testid="stExpander"] {
            border:1px solid #1E2027 !important;
            border-radius:2px !important;
            background:#0A0C10 !important;
        }

        /* === KPI TILES: dense terminal readout === */
        .kpi-row { display:flex; flex-wrap:wrap; gap:0; margin:0.5rem 0 1rem;
                   border:1px solid #1E2027; }
        .kpi { flex:1 1 140px; padding:0.65rem 0.9rem; min-width:120px;
               background:#0A0C10; border-right:1px solid #1E2027; }
        .kpi:last-child { border-right:none; }
        .kpi-label { color:#6B7280; font-size:0.65rem; text-transform:uppercase;
                     letter-spacing:0.14em; font-weight:600;
                     font-family:"Inter", sans-serif; }
        .kpi-value { color:#E8ECEF; font-size:1.35rem; font-weight:600;
                     margin-top:0.2rem; letter-spacing:-0.01em; }
        .kpi.pos .kpi-value  { color:#00E676; }
        .kpi.neg .kpi-value  { color:#FF3B30; }
        .kpi.low .kpi-value  { color:#00E676; }
        .kpi.med .kpi-value  { color:#FFB800; }
        .kpi.high .kpi-value { color:#FF3B30; }

        /* === STOCK HEADER === */
        .stock-header { display:flex; align-items:baseline; gap:0.7rem; margin:0.5rem 0 0;
                        padding-bottom:0.35rem; border-bottom:1px solid #1E2027; }
        .stock-ticker { font-size:1.5rem; font-weight:700; color:var(--accent, #3B82F6);
                        font-family:"Inter", sans-serif; letter-spacing:0.03em; }
        .stock-name   { color:#9AA3AD; font-size:0.85rem; font-weight:500; }
        .stock-sub    { color:#6B7280; font-size:0.7rem; letter-spacing:0.1em;
                        text-transform:uppercase; margin:0.35rem 0 0.5rem;
                        font-family:"Inter", sans-serif; }

        /* === RECOMMENDATION CARDS === */
        .rec-card { padding: 0.25rem 0.25rem 0 0.25rem; }
        .rec-header { display:flex; align-items:baseline; gap:0.7rem; margin-bottom:0.3rem;
                      padding-bottom:0.3rem; border-bottom:1px solid #1E2027; }
        .rec-ticker { font-size:1.35rem; font-weight:700; color:var(--accent, #3B82F6);
                      letter-spacing:0.03em; font-family:"Inter", sans-serif; }
        .rec-name   { color:#9AA3AD; font-size:0.85rem; }
        .rec-rank   { background:var(--accent, #3B82F6); color:#000; font-weight:700;
                      padding:0.15rem 0.55rem; border-radius:2px; font-size:0.65rem;
                      letter-spacing:0.15em; font-family:"Inter", sans-serif; }
        .chip-row   { display:flex; flex-wrap:wrap; gap:0.4rem; margin:0.4rem 0 0.5rem; }
        .chip       { padding:0.2rem 0.55rem; border-radius:2px; font-size:0.7rem;
                      font-weight:600; background:#151820; color:#B8C0C8;
                      border:1px solid #1E2027; letter-spacing:0.05em;
                      font-family:"Inter", sans-serif; text-transform:uppercase; }
        .chip.pos   { background:color-mix(in srgb, #00E676 12%, transparent); color:#00E676; border-color:color-mix(in srgb, #00E676 30%, transparent); }
        .chip.neg   { background:color-mix(in srgb, #FF3B30 12%, transparent); color:#FF3B30; border-color:color-mix(in srgb, #FF3B30 30%, transparent); }
        .chip.low   { background:color-mix(in srgb, #00E676 12%, transparent); color:#00E676; border-color:color-mix(in srgb, #00E676 30%, transparent); }
        .chip.med   { background:rgba(255,184,0,0.1); color:#FFB800; border-color:rgba(255,184,0,0.3); }
        .chip.high  { background:color-mix(in srgb, #FF3B30 12%, transparent); color:#FF3B30; border-color:color-mix(in srgb, #FF3B30 30%, transparent); }

        /* Match-score bar — sharper */
        .match-bar { display:flex; align-items:center; gap:0.6rem; margin:0.25rem 0 0.4rem; }
        .match-bar .label { font-size:0.65rem; color:#6B7280; min-width:85px;
                            letter-spacing:0.1em; text-transform:uppercase;
                            font-family:"Inter", sans-serif; font-weight:600; }
        .match-bar .track { flex:1; height:4px; background:#151820; border-radius:0;
                            border:1px solid #1E2027; overflow:hidden; }
        .match-bar .fill  { height:100%; background:var(--accent, #3B82F6); }
        .match-bar .val   { font-size:0.9rem; color:#E8ECEF; font-weight:700;
                            min-width:38px; text-align:right;
                            font-family:"Inter", sans-serif; }

        /* === DATAFRAMES === */
        .stDataFrame { border:1px solid #1E2027; border-radius:2px; }
        .stDataFrame [role="columnheader"] {
            background:#0A0C10 !important; color:#6B7280 !important;
            font-size:0.65rem !important; letter-spacing:0.1em !important;
            text-transform:uppercase !important; font-weight:600 !important;
        }

        /* === CHAT === */
        /* === CHAT: Claude/ChatGPT-style — AI unboxed, user in a subtle card. === */
        [data-testid="stChatMessage"] {
            background:transparent !important;
            border:none !important;
            padding:0.4rem 0 !important;
            margin-bottom:1rem !important;
            gap:0.9rem !important;
        }
        /* User bubble: subtle contrast — clearly a "you said" card. */
        [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
            background:#151A22 !important;
            border:1px solid #262A33 !important;
            border-radius:14px !important;
            padding:0.9rem 1.15rem !important;
        }
        /* AI message: prose flows freely — no card, more breathing room. */
        [data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) [data-testid="stMarkdownContainer"] p {
            line-height:1.6 !important;
            margin-bottom:0.6rem !important;
        }

        [data-testid="stChatInput"] textarea {
            background:#0E1015 !important; border:1px solid #262A33 !important;
            border-radius:12px !important; padding:0.75rem 0.95rem !important;
            font-size:14px !important;
        }
        [data-testid="stChatInput"] textarea:focus {
            border-color:#3B82F6 !important;
            box-shadow:0 0 0 3px color-mix(in srgb, #3B82F6 20%, transparent) !important;
        }

        /* === TYPING DOTS — shows while waiting for the LLM === */
        .typing-indicator {
            display:inline-flex; align-items:center; gap:0.35rem;
            padding:0.4rem 0.1rem;
        }
        .typing-indicator span {
            width:8px; height:8px; border-radius:50%;
            background:#3B82F6;
            animation:typing-bounce 1.2s infinite ease-in-out;
        }
        .typing-indicator span:nth-child(2) { animation-delay:0.15s; }
        .typing-indicator span:nth-child(3) { animation-delay:0.30s; }
        @keyframes typing-bounce {
            0%, 60%, 100% { transform:translateY(0);    opacity:0.35; }
            30%           { transform:translateY(-6px); opacity:1;    }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="term-header">'
        '<span class="term-title">TRADE<span class="slash">MATRIX</span></span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Auth gate: nothing renders below unless the user is signed in.
    if require_login() is None:
        return

    # Sidebar: profile / chats / logout / comparison-color pickers (from Mongo).
    # The two returned colors feed into palette() for multi-series charts only —
    # semantic gain/loss coloring is hardcoded green/red in CSS + charts.py.
    render_sidebar_account_panel()
    default_period = _render_default_period_selector()

    controller = VisualizationController()

    (tab_price, tab_compare, tab_corr, tab_recommend,
     tab_watchlist, tab_assistant) = st.tabs(
        ["Prices & volatility", "Compare two stocks", "Correlation", "Recommender",
         "Watchlist", "Investment Assistant"]
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

    with tab_watchlist:
        render_watchlist_tab()

    with tab_assistant:
        render_assistant_tab()


if __name__ == "__main__":
    main()
