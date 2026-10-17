"""Streamlit Watchlist tab — modern app-style layout.

- Market overview strip at the top (SPY / QQQ / DIA / IWM).
- Each watchlist renders as a borderless section with icon buttons for
  rename and delete, triggering real modal dialogs (@st.dialog).
- Clicking a ticker swaps the whole tab to an inline detail view (reusing
  PriceMetricsDashboard) — no "please switch tabs" message, no reload.
"""

from __future__ import annotations

import html

import streamlit as st

from finance_dashboard.auth.db import get_db
from finance_dashboard.auth.ui import current_user
from finance_dashboard.config import PERIOD_OPTIONS
from finance_dashboard.data import company_name, fetch_history
from finance_dashboard.watchlist import store
from finance_dashboard.widgets import (
    POPULAR_TICKERS,
    ticker_from_label,
    ticker_labels,
    ticker_prefix_autocomplete,
)

_INDICES = [
    ("SPY", "S&P 500"),
    ("QQQ", "Nasdaq 100"),
    ("DIA", "Dow Jones"),
    ("IWM", "Russell 2000"),
]

# Per-stock time-range options. The yfinance period column is the fetch window;
# the computation decides whether we compare day-over-day or across the window.
_RANGES = ["1D", "1W", "1M", "3M", "1Y"]
_RANGE_PERIODS = {
    "1D": "5d", "1W": "5d", "1M": "1mo", "3M": "3mo", "1Y": "1y",
}


# --- Shared services (no duplication) ------------------------------------


@st.cache_data(ttl=300, show_spinner=False)
def _ticker_exists(ticker: str) -> bool:
    try:
        df = fetch_history((ticker.upper(),), "5d")
        return not df.empty and ticker.upper() in df.columns
    except Exception:
        return False


@st.cache_data(ttl=120, show_spinner=False)
def _quote(ticker: str, range_key: str = "1D") -> dict:
    """Return price + change over the chosen window.

    1D compares today vs previous close; longer windows compare the first vs
    last close in the fetched period.
    """
    period = _RANGE_PERIODS.get(range_key, "5d")
    try:
        df = fetch_history((ticker.upper(),), period)
    except Exception:
        return {"price": None, "base": None, "pct_change": None}
    if df.empty or ticker.upper() not in df.columns:
        return {"price": None, "base": None, "pct_change": None}
    series = df[ticker.upper()].dropna()
    if len(series) < 1:
        return {"price": None, "base": None, "pct_change": None}
    price = float(series.iloc[-1])
    if range_key == "1D":
        base = float(series.iloc[-2]) if len(series) >= 2 else None
    else:
        base = float(series.iloc[0]) if len(series) >= 1 else None
    pct = ((price / base - 1.0) * 100.0) if base else None
    return {"price": price, "base": base, "pct_change": pct}


# --- Market overview ------------------------------------------------------


def _render_market_overview() -> None:
    cols = st.columns(len(_INDICES))
    for (ticker, label), col in zip(_INDICES, cols):
        q = _quote(ticker, "1D")
        price = q["price"]
        pct = q["pct_change"]
        tone = "wl-flat"
        pct_txt = "—"
        if pct is not None:
            tone = "wl-pos" if pct >= 0 else "wl-neg"
            pct_txt = f"{pct:+.2f}%"
        price_txt = f"{price:,.2f}" if price is not None else "—"
        col.markdown(
            f'<div class="wl-idx-card">'
            f'<div class="wl-idx-label">{html.escape(label)}</div>'
            f'<div class="wl-idx-price">{price_txt}</div>'
            f'<div class="wl-idx-pct {tone}">{pct_txt}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# --- Stock rows -----------------------------------------------------------


def _render_stock_row(wid, ticker: str, username: str, db) -> None:
    """Row = logo | ticker | change | range | price | ✕. Add-to-compare now
    lives in the global picker dialog opened from the compare bar."""
    range_sel_key = f"wl-range-{wid}-{ticker}"
    current_range = st.session_state.get(range_sel_key, "1D")

    q = _quote(ticker, current_range)
    price = q["price"]
    pct = q["pct_change"]
    base = q["base"]

    if pct is None or price is None or base is None:
        pct_cls = "wl-flat"
        pct_txt = "—"
        chg_txt = ""
    else:
        pct_cls = "wl-pos" if pct >= 0 else "wl-neg"
        pct_txt = f"{pct:+.2f}%"
        chg_txt = f"{price - base:+,.2f}"
    price_txt = f"{price:,.2f}" if price is not None else "—"

    letter = html.escape(ticker[0]) if ticker else "?"
    logo_html = (
        f'<img class="wl-logo" '
        f'src="https://assets.parqet.com/logos/symbol/{html.escape(ticker)}" '
        f'onerror="this.onerror=null;'
        f'this.src=\'https://financialmodelingprep.com/image-stock/{html.escape(ticker)}.png\';'
        f'this.addEventListener(\'error\',()=>'
        f'this.replaceWith(Object.assign(document.createElement(\'div\'),'
        f'{{className:\'wl-logo-fallback\',innerText:\'{letter}\'}})),{{once:true}})" />'
    )

    st.markdown('<div class="wl-row">', unsafe_allow_html=True)
    cols = st.columns([0.7, 1.4, 1.3, 0.85, 1.4, 0.35],
                      vertical_alignment="center")

    cols[0].markdown(logo_html, unsafe_allow_html=True)
    with cols[1]:
        if st.button(ticker, key=f"sel-{wid}-{ticker}", type="tertiary"):
            st.session_state["_wl_detail_ticker"] = ticker
            st.rerun()
    cols[2].markdown(
        f'<div class="wl-change {pct_cls}">'
        f'<div class="wl-pct-big">{pct_txt}</div>'
        f'<div class="wl-chg-sml">{chg_txt}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    with cols[3]:
        st.selectbox(
            "Range",
            options=_RANGES,
            index=_RANGES.index(current_range) if current_range in _RANGES else 0,
            key=range_sel_key,
            label_visibility="collapsed",
        )
    cols[4].markdown(
        f'<div class="wl-price-big">{price_txt}</div>',
        unsafe_allow_html=True,
    )
    with cols[5]:
        if st.button("✕", key=f"rm-{wid}-{ticker}", type="tertiary"):
            store.remove_ticker(db, username, wid, ticker)
            _quote.clear()
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# --- Compare bar (always visible, single clean row) --------------------


def _render_compare_bar(selected: list[str], db, username: str) -> None:
    """One compact bar with: [+ picker] [status message] [Compare] [Clear]."""
    count = len(selected)
    can_compare = count >= 2

    if count == 0:
        message = (
            '<span class="wl-cmp-dim">Click the <b>+</b> to pick stocks '
            'from your watchlists and compare them side-by-side.</span>'
        )
    elif count == 1:
        message = (
            f'<b>{html.escape(selected[0])}</b> added · '
            '<span class="wl-cmp-dim">pick one more to compare</span>'
        )
    else:
        tickers_str = ", ".join(selected)
        if len(tickers_str) > 60:
            tickers_str = f"{count} stocks"
        message = f'<b>{html.escape(tickers_str)}</b> ready to compare'

    # Define the picker dialog here so it has `db` and `username` in scope.
    @st.dialog("Pick stocks to compare", width="large")
    def _picker():
        _render_picker_dialog(db, username)

    bar = st.columns([0.6, 4.4, 1.2, 1.1], vertical_alignment="center")
    with bar[0]:
        if st.button("＋", key="wl-cmp-add",
                     type="primary", use_container_width=True,
                     help="Pick stocks from any watchlist"):
            _picker()
    bar[1].markdown(
        f'<div class="wl-cmp-bar {"ready" if can_compare else ""}">{message}</div>',
        unsafe_allow_html=True,
    )
    with bar[2]:
        if st.button("Compare", key="wl-cmp-open",
                     type="primary", use_container_width=True,
                     disabled=not can_compare):
            st.session_state["_wl_compare_active"] = True
            st.rerun()
    with bar[3]:
        if st.button("Clear", key="wl-cmp-clear",
                     use_container_width=True,
                     disabled=count == 0):
            st.session_state["_wl_compare_set"] = set()
            st.rerun()


def _render_picker_dialog(db, username: str) -> None:
    """Content of the 'Pick stocks to compare' dialog.

    Stays open while the user ticks any number of boxes — only `st.rerun()`
    from inside closes it, so Done is the explicit way out.
    """
    selected_set = st.session_state.setdefault("_wl_compare_set", set())
    watchlists = store.list_for_user(db, username)
    has_any = any(wl.get("tickers") for wl in watchlists)

    if not has_any:
        st.info("You have no stocks in your watchlists yet. "
                "Add some tickers to a list first.")
        if st.button("Close", type="primary", use_container_width=True):
            st.rerun()
        return

    # Prefix-only search — only filters once the user types ≥1 character.
    search = st.text_input(
        "Search",
        key="_picker_search",
        placeholder="Type a letter to filter (prefix match, e.g. 'A' → AAPL, AMZN)",
        label_visibility="collapsed",
    )
    prefix = (search or "").strip().upper()

    st.caption(
        f"**{len(selected_set)}** stock(s) selected · "
        "Tick as many as you want, then press Done."
    )

    any_shown = False
    for wl in watchlists:
        tickers = wl.get("tickers", [])
        # Apply strict starts-with filter when the user has typed something.
        if prefix:
            tickers = [t for t in tickers if t.upper().startswith(prefix)]
        if not tickers:
            continue
        any_shown = True
        st.markdown(f'<div class="wl-picker-group">{html.escape(wl["name"])}'
                    f' <span class="wl-picker-count">{len(tickers)}</span></div>',
                    unsafe_allow_html=True)

        grid_cols = st.columns(3)
        for i, t in enumerate(tickers):
            with grid_cols[i % 3]:
                is_sel = t in selected_set
                new_val = st.checkbox(
                    t,
                    value=is_sel,
                    key=f"picker-{wl['_id']}-{t}",
                )
                # Mutate the shared set WITHOUT calling st.rerun() so the
                # dialog stays open across checkbox clicks.
                if new_val and not is_sel:
                    selected_set.add(t)
                elif not new_val and is_sel:
                    selected_set.discard(t)

    if prefix and not any_shown:
        st.caption(f"No tickers in your watchlists start with **{prefix}**.")

    st.markdown("---")
    footer = st.columns([3, 1, 1])
    footer[0].markdown(
        f'<div class="wl-picker-footer">'
        f'<b>{len(selected_set)}</b> selected</div>',
        unsafe_allow_html=True,
    )
    if footer[1].button("Clear all", key="picker-clear-all",
                        use_container_width=True):
        selected_set.clear()
        # No rerun — stays open so user sees everything cleared and can re-pick.
    if footer[2].button("Done", key="picker-done",
                        type="primary", use_container_width=True):
        st.rerun()  # closes the dialog


# --- Inline compare view (replaces list when 2+ stocks selected & Compare clicked) ---


_CMP_PERIODS = {
    "1 month": "1mo", "3 months": "3mo", "6 months": "6mo",
    "1 year": "1y", "5 years": "5y",
}


def _render_compare_view(tickers: list[str]) -> None:
    """Full comparison view — normalized returns, metrics table, correlation,
    rolling volatility. Reuses the shared chart and analytics helpers so there's
    zero duplicated logic."""
    import pandas as pd

    from finance_dashboard.analytics import (
        annualized_volatility,
        log_returns,
        period_pct_change,
        rolling_annualized_volatility,
    )
    from finance_dashboard.charts import (
        fig_correlation_heatmap,
        fig_pct_change,
        fig_rolling_vol,
    )

    top = st.columns([1, 5])
    with top[0]:
        if st.button(":material/arrow_back: Back", key="wl-cmp-back",
                     use_container_width=True):
            st.session_state["_wl_compare_active"] = False
            st.rerun()
    with top[1]:
        st.markdown(
            f'<div class="wl-detail-head">Compare · '
            f'<span class="wl-detail-name">{html.escape(", ".join(tickers))}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    period_label = st.selectbox(
        "Period",
        options=list(_CMP_PERIODS.keys()),
        index=3,  # default: 1 year
        key="wl-cmp-period",
    )
    period = _CMP_PERIODS[period_label]

    with st.spinner("Loading…"):
        df = fetch_history(tuple(tickers), period)
    if df is None or df.empty:
        st.error("No data available for the selected stocks.")
        return

    missing = [t for t in tickers if t not in df.columns]
    if missing:
        st.warning(f"No data for: {', '.join(missing)}")
    valid = [t for t in tickers if t in df.columns]
    if len(valid) < 2:
        st.error("Need at least two stocks with data to compare.")
        return
    df = df[valid]

    # --- 1. Normalized returns (outperformance view) ---
    cum_pct = (df / df.iloc[0] - 1.0) * 100
    st.plotly_chart(
        fig_pct_change(cum_pct, f"Normalized returns · {period_label}"),
        use_container_width=True,
        config={"displayModeBar": False},
        key="wl-compare-chart",
    )

    # --- 2. Summary metrics table ---
    rets = df.pct_change().dropna()
    rows = []
    for t in valid:
        series = df[t]
        vol = annualized_volatility(rets[t]) * 100 if t in rets.columns else float("nan")
        ret = period_pct_change(series)
        # Max drawdown: deepest fall from the running peak over the period.
        drawdown = ((series / series.cummax() - 1.0).min()) * 100
        risk_adj = (ret / vol) if (vol and vol > 0) else None
        rows.append({
            "Ticker": t,
            "Company": (company_name(t) or "")[:28],
            "Start": f"${series.iloc[0]:,.2f}",
            "Current": f"${series.iloc[-1]:,.2f}",
            "Return": f"{ret:+.2f}%",
            "Volatility": f"{vol:.2f}%",
            "Max drawdown": f"{drawdown:.2f}%",
            "Return / vol": f"{risk_adj:.2f}" if risk_adj is not None else "—",
        })
    st.markdown('<div class="wl-cmp-section">Summary</div>',
                unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- 3. Correlation matrix — essential for diversification intuition. ---
    st.markdown('<div class="wl-cmp-section">Correlation of daily log-returns</div>',
                unsafe_allow_html=True)
    try:
        log_rets = log_returns(df)
        if len(log_rets) >= 10:
            corr = log_rets.corr()
            st.plotly_chart(
                fig_correlation_heatmap(corr),
                use_container_width=True,
                config={"displayModeBar": False},
                key="wl-compare-corr",
            )
            st.caption(
                "Values close to +1 → stocks move together (less diversification). "
                "Close to 0 → unrelated. Negative → they tend to move opposite."
            )
        else:
            st.caption("Not enough overlapping days to compute a reliable correlation.")
    except Exception as exc:
        st.caption(f"Correlation unavailable: {exc}")

    # --- 4. Rolling volatility — which stock gets wild when? ---
    st.markdown('<div class="wl-cmp-section">Rolling 21-day volatility</div>',
                unsafe_allow_html=True)
    try:
        roll = rolling_annualized_volatility(rets)
        st.plotly_chart(
            fig_rolling_vol(roll, "Rolling 21-day annualized volatility"),
            use_container_width=True,
            config={"displayModeBar": False},
            key="wl-compare-rollvol",
        )
    except Exception as exc:
        st.caption(f"Rolling volatility unavailable: {exc}")


# --- Inline detail view (replaces list when a ticker is clicked) ----------


def _render_detail_view(ticker: str) -> None:
    """Full stock detail — reuses PriceMetricsDashboard so there's zero duplication."""
    from finance_dashboard.dashboards import PriceMetricsDashboard

    top = st.columns([1, 5])
    with top[0]:
        if st.button(":material/arrow_back: Back",
                     key="wl-back",
                     use_container_width=True):
            st.session_state.pop("_wl_detail_ticker", None)
            st.rerun()
    with top[1]:
        st.markdown(
            f'<div class="wl-detail-head">{html.escape(ticker)} · '
            f'<span class="wl-detail-name">{html.escape(company_name(ticker) or "")}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Use the same dashboard the Prices tab uses — zero duplication.
    # Unique key_prefix so our charts don't collide with the Prices tab's.
    try:
        PriceMetricsDashboard().display([ticker], "1 year", key_prefix="wl-detail-")
    except Exception as exc:
        st.error(f"Could not load details for {ticker}: {exc}")


# --- Watchlist section ---------------------------------------------------


def _render_watchlist(wl: dict, username: str, db) -> None:
    wid = wl["_id"]
    name = wl["name"]
    tickers = wl.get("tickers", [])

    # --- modal dialogs (defined per-render so they close over db/username) ---
    @st.dialog("Rename watchlist")
    def _rename_dialog() -> None:
        new_name = st.text_input("New name", value=name, key=f"dlg-rename-{wid}")
        c1, c2 = st.columns(2)
        if c1.button("Save", key=f"dlg-rsave-{wid}",
                     type="primary", use_container_width=True):
            try:
                store.rename(db, username, wid, new_name)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if c2.button("Cancel", key=f"dlg-rcancel-{wid}", use_container_width=True):
            st.rerun()

    @st.dialog("Delete watchlist?")
    def _delete_dialog() -> None:
        st.markdown(
            f"<p>Are you sure you want to delete <b>{html.escape(name)}</b>?</p>"
            f'<p style="color:#8A9BA8; font-size:0.85rem;">'
            f'This can\'t be undone.</p>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        if c1.button("Delete", key=f"dlg-dok-{wid}",
                     type="primary", use_container_width=True):
            store.delete(db, username, wid)
            st.rerun()
        if c2.button("Cancel", key=f"dlg-dcancel-{wid}", use_container_width=True):
            st.rerun()

    expand_key = f"wl-expanded-{wid}"
    # Default: expanded on first view so user sees their stocks right away.
    expanded = st.session_state.get(expand_key, True)

    # --- header row: chevron + name + edit + delete -----------------------
    hdr = st.columns([0.5, 4.5, 0.6, 0.6], vertical_alignment="center")
    chevron = ":material/expand_more:" if expanded else ":material/chevron_right:"
    if hdr[0].button(chevron, key=f"collapse-{wid}", type="tertiary",
                     help="Collapse" if expanded else "Expand"):
        st.session_state[expand_key] = not expanded
        st.rerun()
    hdr[1].markdown(
        f'<div class="wl-section-title">{html.escape(name)}'
        f' <span class="wl-section-count">{len(tickers)}</span></div>',
        unsafe_allow_html=True,
    )
    if hdr[2].button(":material/edit:", key=f"edit-{wid}",
                     type="tertiary", help="Rename this watchlist"):
        _rename_dialog()
    if hdr[3].button(":material/delete:", key=f"del-{wid}",
                     type="tertiary", help="Delete this watchlist"):
        _delete_dialog()

    # --- rows (only when expanded) ---
    if expanded:
        if tickers:
            for t in tickers:
                _render_stock_row(wid, t, username, db)
        else:
            st.markdown(
                '<div class="wl-list-empty">No tickers yet — add one below.</div>',
                unsafe_allow_html=True,
            )

        # --- add-ticker row: searchable multi-pick dropdown ---
        add_c1, add_c2 = st.columns([5, 1.2])
        add_key = f"wl-add-sel-{wid}"
        with add_c1:
            # Guard against the StreamlitDefaultNotInOptionsError pattern — see
            # ticker_select in widgets.py — by making options tolerant of any
            # stale label the user may still have selected from a prior session.
            opts = list(ticker_labels())
            prior = st.session_state.get(add_key) or []
            for lbl in prior:
                if lbl not in opts:
                    opts.append(lbl)
            picked = st.multiselect(
                "Add a ticker",
                options=opts,
                default=[],
                accept_new_options=True,
                key=add_key,
                label_visibility="collapsed",
                placeholder="Type letters to search — pick one or more, then Add",
            )
        if add_c2.button("Add", key=f"addbtn-{wid}", use_container_width=True):
            errors: list[str] = []
            for label in picked:
                ticker = ticker_from_label(label)
                try:
                    store.add_ticker(db, username, wid, ticker, exists_fn=_ticker_exists)
                except (ValueError, PermissionError) as exc:
                    errors.append(f"{ticker}: {exc}")
            for msg in errors:
                st.error(msg)
            # Clear the picker so the row is ready for the next batch.
            st.session_state.pop(add_key, None)
            st.rerun()

    st.markdown('<hr class="wl-section-sep" />', unsafe_allow_html=True)


# --- Main entry -----------------------------------------------------------


def render_watchlist_tab() -> None:
    db = get_db()
    username = current_user()
    if db is None or not username:
        st.info("Sign in and configure MongoDB to use watchlists.")
        return

    st.markdown(_WATCHLIST_CSS, unsafe_allow_html=True)

    # Compare view takes over the tab when the user hit Compare.
    if st.session_state.get("_wl_compare_active"):
        selected = sorted(st.session_state.get("_wl_compare_set", set()))
        if len(selected) >= 2:
            _render_compare_view(selected)
            return
        # Not enough selected — fall through to the normal view.
        st.session_state["_wl_compare_active"] = False

    # Detail view takes over the tab if a ticker was clicked.
    detail = st.session_state.get("_wl_detail_ticker")
    if detail:
        _render_detail_view(detail)
        return

    # --- Page header ---
    st.markdown('<div class="wl-page-title">Watchlists</div>', unsafe_allow_html=True)
    st.markdown('<div class="wl-page-sub">Track your favorite tickers</div>',
                unsafe_allow_html=True)

    # --- Compare bar: single clean row that explains itself ---
    selected = sorted(st.session_state.get("_wl_compare_set", set()))
    _render_compare_bar(selected, db, username)

    # --- Market overview (fills the empty space usefully) ---
    st.markdown('<div class="wl-section-label">Market overview</div>',
                unsafe_allow_html=True)
    _render_market_overview()

    st.markdown('<hr class="wl-section-sep" />', unsafe_allow_html=True)

    # --- Create new watchlist ---
    st.markdown('<div class="wl-section-label">Your lists</div>',
                unsafe_allow_html=True)
    with st.form("new_watchlist_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 1.2])
        name = c1.text_input(
            "New watchlist name",
            placeholder="New watchlist name (e.g. Tech, AI, Dividends)",
            label_visibility="collapsed",
        )
        if c2.form_submit_button("＋ Create", use_container_width=True, type="primary"):
            try:
                store.create(db, username, name)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    watchlists = store.list_for_user(db, username)
    if not watchlists:
        st.markdown(
            '<div class="wl-empty">'
            '<div class="wl-empty-icon">📋</div>'
            '<div class="wl-empty-title">No watchlists yet</div>'
            '<div class="wl-empty-sub">Create your first list above to start tracking stocks.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    for wl in watchlists:
        _render_watchlist(wl, username, db)


_WATCHLIST_CSS = """
<style>
/* === PAGE HEADER === */
.wl-page-title {
    font-size: 2rem; font-weight: 700; color: #E8ECEF;
    letter-spacing: -0.02em; margin-bottom: 0.15rem;
}
.wl-page-sub {
    color: #6B7280; font-size: 0.85rem; margin-bottom: 1.4rem;
}

/* === SECTION LABELS === */
.wl-section-label {
    color: #6B7280; font-size: 0.72rem; letter-spacing: 0.14em;
    text-transform: uppercase; font-weight: 600;
    margin: 0.4rem 0 0.7rem;
}

/* === MARKET OVERVIEW CARDS === */
.wl-idx-card {
    padding: 0.9rem 1rem;
    background: linear-gradient(135deg, #0D1016 0%, #000 100%);
    border: 1px solid #1A1F28; border-radius: 10px;
}
.wl-idx-label {
    color: #8A9BA8; font-size: 0.7rem;
    text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;
}
.wl-idx-price {
    color: #FFFFFF; font-size: 1.4rem; font-weight: 700;
    letter-spacing: -0.01em; margin-top: 0.3rem;
    font-variant-numeric: tabular-nums;
}
.wl-idx-pct {
    font-size: 0.85rem; font-weight: 600; margin-top: 0.1rem;
    font-variant-numeric: tabular-nums;
}

/* === WATCHLIST SECTION === */
.wl-section-title {
    font-size: 1.15rem; font-weight: 600; color: #E8ECEF;
    letter-spacing: -0.01em;
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.3rem 0;
}
.wl-section-count {
    display: inline-block; padding: 0.05rem 0.5rem;
    background: #131A22; color: #8A9BA8;
    border-radius: 999px; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.03em;
}
.wl-section-sep {
    border: none; border-top: 1px solid #121820;
    margin: 1rem 0;
}
.wl-list-empty {
    color: #6B7280; font-size: 0.85rem; padding: 0.6rem 0 0.8rem;
}

/* === ICON BUTTONS (rename/delete) === */
div[data-testid="stButton"] button[kind="tertiary"]:where([data-testid*="edit-"]),
div[data-testid="stButton"] button[kind="tertiary"]:where([data-testid*="del-"]) {
    color: #8A9BA8 !important;
    padding: 0.3rem !important;
    background: transparent !important;
    border-radius: 6px !important;
    transition: color 120ms ease, background 120ms ease;
}
div[data-testid="stButton"] button[kind="tertiary"]:where([data-testid*="edit-"]):hover {
    color: #3B82F6 !important;
    background: #0E1015 !important;
}
div[data-testid="stButton"] button[kind="tertiary"]:where([data-testid*="del-"]):hover {
    color: #FF3B30 !important;
    background: #0E1015 !important;
}

/* Collapse/expand chevron — subtle, left of the watchlist name. */
div[data-testid="stButton"] button[kind="tertiary"]:where([data-testid*="collapse-"]) {
    color: #6B7280 !important;
    padding: 0.25rem !important;
    background: transparent !important;
    border-radius: 6px !important;
    transition: color 120ms ease, background 120ms ease;
}
div[data-testid="stButton"] button[kind="tertiary"]:where([data-testid*="collapse-"]):hover {
    color: #E8ECEF !important;
    background: #0E1015 !important;
}

/* Per-row range selector — compact dropdown next to the pct block. */
div[data-testid="stSelectbox"]:has(div[data-testid*="wl-range-"]),
div.stSelectbox:has([data-baseweb="select"][aria-label="Range"]) {
    min-width: 0 !important;
}
.wl-row div[data-baseweb="select"] > div {
    min-height: 32px !important;
    padding: 0.1rem 0.4rem !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    border-radius: 6px !important;
    background: #0E1015 !important;
    border: 1px solid #1E2027 !important;
}
.wl-row div[data-baseweb="select"] svg { width: 14px !important; height: 14px !important; }

/* === STOCK ROW === */
.wl-row {
    padding: 0.3rem 0.4rem;
    border-radius: 8px;
    transition: background 150ms ease;
}
.wl-row:hover { background: #0E1015; }
.wl-logo, .wl-logo-fallback {
    width: 40px; height: 40px; border-radius: 50%;
    background: #FFFFFF;
    display: inline-flex; align-items: center; justify-content: center;
    object-fit: contain; padding: 3px;
    border: 1px solid #1E2027;
    overflow: hidden;
}
.wl-logo-fallback {
    background: #131A22; color: #E8ECEF;
    font-weight: 700; font-size: 1.05rem;
    letter-spacing: 0.03em;
}
div[data-testid="stButton"] button[kind="tertiary"]:where([data-testid*="sel-"]) {
    color: #E8ECEF !important;
    font-weight: 700 !important; font-size: 1.15rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.25rem 0 !important;
    background: transparent !important;
    text-align: left !important; justify-content: flex-start !important;
}
div[data-testid="stButton"] button[kind="tertiary"]:where([data-testid*="sel-"]):hover {
    color: #3B82F6 !important; background: transparent !important;
}
div[data-testid="stButton"] button[kind="tertiary"]:where([data-testid*="rm-"]) {
    color: #4A5260 !important; font-size: 0.9rem !important;
    padding: 0 !important; background: transparent !important;
    opacity: 0.5;
    transition: opacity 120ms ease, color 120ms ease;
}
div[data-testid="stButton"] button[kind="tertiary"]:where([data-testid*="rm-"]):hover {
    opacity: 1; color: #FF3B30 !important;
}
.wl-change { font-variant-numeric: tabular-nums; }
.wl-pct-big { font-size: 1rem; font-weight: 600; line-height: 1.2; }
.wl-chg-sml { font-size: 0.85rem; font-weight: 500; line-height: 1.2;
              opacity: 0.85; margin-top: 2px; }
.wl-pos    { color: #00E676; }
.wl-neg    { color: #FF3B30; }
.wl-flat   { color: #6B7280; }
.wl-price-big {
    color: #FFFFFF; font-variant-numeric: tabular-nums;
    font-weight: 700; font-size: 1.2rem;
    text-align: right; letter-spacing: -0.01em;
}

/* === EMPTY STATE === */
.wl-empty {
    text-align: center; padding: 2.5rem 1rem;
    border: 1px dashed #1E2027; border-radius: 10px;
    background: #080A0D; margin-top: 1rem;
}
.wl-empty-icon { font-size: 2.2rem; opacity: 0.6; margin-bottom: 0.4rem; }
.wl-empty-title { color: #E8ECEF; font-weight: 600; font-size: 1.05rem; }
.wl-empty-sub { color: #6B7280; font-size: 0.85rem; margin-top: 0.2rem; }

/* === PICKER DIALOG === */
.wl-picker-group {
    color: #E8ECEF; font-weight: 600; font-size: 1rem;
    margin: 1rem 0 0.4rem;
    letter-spacing: -0.01em;
    display: flex; align-items: center; gap: 0.5rem;
}
.wl-picker-count {
    background: #131A22; color: #8A9BA8;
    padding: 0.05rem 0.5rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.03em;
}
.wl-picker-footer {
    color: #9AA3AD; padding: 0.3rem 0;
}
.wl-picker-footer b { color: #E8ECEF; }

/* Dialog + button — big and blue so users understand it's the primary CTA. */
div[data-testid="stButton"] button:where([data-testid*="wl-cmp-add"]) {
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    padding: 0.3rem 0 !important;
    line-height: 1 !important;
}

/* === COMPARE BAR (single-line, always visible, self-explaining) === */
.wl-cmp-bar {
    background: #0E1015;
    border: 1px solid #1E2027;
    border-radius: 8px;
    padding: 0.65rem 0.9rem;
    color: #9AA3AD;
    font-size: 0.9rem;
    line-height: 1.4;
}
.wl-cmp-bar b { color: #E8ECEF; font-weight: 600; }
.wl-cmp-bar.ready {
    background: linear-gradient(90deg, rgba(59,130,246,0.14), rgba(59,130,246,0.04));
    border-color: rgba(59,130,246,0.35);
    color: #E8ECEF;
}
.wl-cmp-dim { color: #6B7280; }
.wl-cmp-plus-inline {
    display: inline-block;
    background: #3B82F6; color: #FFFFFF;
    width: 20px; height: 20px; line-height: 18px;
    border-radius: 50%;
    text-align: center; font-weight: 700;
    font-size: 0.85rem;
    vertical-align: baseline;
    margin-right: 0.3rem;
}
.wl-cmp-count {
    display: inline-block;
    background: #3B82F6;
    color: #FFFFFF;
    font-weight: 700;
    padding: 0.1rem 0.55rem;
    border-radius: 999px;
    font-size: 0.85rem;
    margin-right: 0.2rem;
}
.wl-cmp-label { color: #8A9BA8; }
.wl-cmp-tickers { font-weight: 600; letter-spacing: 0.02em; }

/* Compare toggle button — big enough to be inviting; always-blue ring
   makes it clear "this is the compare action", distinct from the red
   delete button on the far right. */
div[data-testid="stButton"] button[kind="tertiary"]:where([data-testid*="cmp-OFF-"]) {
    color: #3B82F6 !important;
    background: rgba(59,130,246,0.08) !important;
    border: 1.5px solid rgba(59,130,246,0.35) !important;
    border-radius: 50% !important;
    padding: 0 !important;
    width: 34px !important; height: 34px !important;
    min-height: 0 !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    transition: color 120ms ease, background 120ms ease,
                border-color 120ms ease, transform 120ms ease;
}
div[data-testid="stButton"] button[kind="tertiary"]:where([data-testid*="cmp-OFF-"]):hover {
    color: #FFFFFF !important;
    background: #3B82F6 !important;
    border-color: #3B82F6 !important;
    transform: scale(1.08);
}
div[data-testid="stButton"] button[kind="tertiary"]:where([data-testid*="cmp-ON-"]) {
    color: #FFFFFF !important;
    background: #3B82F6 !important;
    border: 1.5px solid #3B82F6 !important;
    border-radius: 50% !important;
    padding: 0 !important;
    width: 34px !important; height: 34px !important;
    min-height: 0 !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.18);
}
div[data-testid="stButton"] button[kind="tertiary"]:where([data-testid*="cmp-ON-"]):hover {
    background: #2563EB !important;
    border-color: #2563EB !important;
}

/* Compare view section headers */
.wl-cmp-section {
    color: #E8ECEF; font-weight: 600; font-size: 1.05rem;
    letter-spacing: -0.01em;
    margin: 1.5rem 0 0.6rem;
}

/* === INLINE DETAIL VIEW === */
.wl-detail-head {
    font-size: 1.6rem; font-weight: 700; color: #E8ECEF;
    letter-spacing: -0.02em;
    padding: 0.3rem 0;
}
.wl-detail-name {
    color: #8A9BA8; font-weight: 400; font-size: 1rem;
    margin-left: 0.3rem;
}
</style>
"""
