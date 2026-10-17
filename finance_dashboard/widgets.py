"""Shared Streamlit widgets reused across tabs."""

from __future__ import annotations

import streamlit as st

from finance_dashboard.recommender.universe import (
    ETF_DISPLAY_NAMES,
    ETF_FALLBACK_HOLDINGS,
    PREDEFINED_UNIVERSES,
)

# One deduplicated pool of every ticker we know about. Used as the default
# suggestion list for ticker_prefix_autocomplete when the caller doesn't pass
# a scoped list of its own (e.g. a watchlist's own tickers).
POPULAR_TICKERS: list[str] = sorted(
    {t for ts in ETF_FALLBACK_HOLDINGS.values() for t in ts}
    | {t for ts in PREDEFINED_UNIVERSES.values() for t in ts}
)

# Separator between the ticker and its company name in option labels.
# A thin middle dot reads cleaner than a dash — subtle enough to not
# compete with the ticker but clear enough to delimit the fields.
_LABEL_SEP = "  ·  "


@st.cache_data(ttl=3600, show_spinner=False)
def ticker_labels() -> list[str]:
    """`["AAPL — Apple Inc.", "ABBV — AbbVie Inc.", ...]` for every popular ticker.

    Company names come from the yfinance-backed `company_name()` lookup, which
    is itself cached — we add another layer here so building the dropdown is
    always a cache hit on reruns. One hour TTL is fine; company names don't
    churn.
    """
    from finance_dashboard.data import company_name
    labels: list[str] = []
    for ticker in POPULAR_TICKERS:
        try:
            name = company_name(ticker) or ""
        except Exception:
            name = ""
        labels.append(f"{ticker}{_LABEL_SEP}{name}" if name else ticker)
    return labels


def ticker_from_label(label: str) -> str:
    """Extract the ticker symbol from an `AAPL · Apple Inc.` option label.

    Tolerates both the current middle-dot separator and the older em-dash
    one, so session_state values persisted before this change still parse.
    """
    if not label:
        return ""
    for sep in (_LABEL_SEP, " · ", " — "):
        if sep in label:
            return label.split(sep, 1)[0].strip().upper()
    return label.strip().upper()


def label_for_ticker(ticker: str) -> str:
    """Build the display label for a ticker, falling back to the symbol itself."""
    from finance_dashboard.data import company_name
    try:
        name = company_name(ticker) or ""
    except Exception:
        name = ""
    return f"{ticker}{_LABEL_SEP}{name}" if name else ticker


def label_for_etf(etf: str) -> str:
    """`QQQ  ·  Invesco QQQ Trust (Nasdaq 100)` — ETF ticker + full fund name."""
    if not etf or etf == "None":
        return etf
    name = ETF_DISPLAY_NAMES.get(etf, "")
    return f"{etf}{_LABEL_SEP}{name}" if name else etf


def etf_from_label(label: str) -> str | None:
    """Strip the ETF ticker back out of an `XLK  ·  Technology Select...` label."""
    if not label or label == "None":
        return None
    return ticker_from_label(label)


def ticker_select(
    label: str,
    *,
    key: str,
    default: str = "AAPL",
    label_visibility: str = "visible",
) -> str:
    """Single-ticker picker with the Correlation-tab search UX.

    Uses `st.multiselect(max_selections=1, accept_new_options=True)` so the
    user gets the same type-ahead dropdown as the Correlation tab: typing a
    letter filters the options down to tickers that match, and the full
    company name is visible alongside each symbol. Falling back to a plain
    text input when a user types an unknown symbol (e.g. a small-cap not in
    `POPULAR_TICKERS`) still works because `accept_new_options=True`.

    Returns the ticker symbol only — the company-name annotation is stripped.
    """
    options = list(ticker_labels())
    default_label = label_for_ticker(default) if default else ""
    # Streamlit's multiselect VALIDATES that each `default` value is already
    # in `options` — accept_new_options only applies to what the USER types.
    # If the cached ticker_labels() call fell back to a bare ticker (e.g.
    # yfinance was slow when the cache warmed), label_for_ticker's fresh call
    # may now add the company name, giving a mismatch. Prepend when missing
    # so the picker always has a safe default to render.
    if default_label and default_label not in options:
        options = [default_label] + options
    default_list = [default_label] if default_label else []

    selected = st.multiselect(
        label,
        options=options,
        default=default_list,
        max_selections=1,
        accept_new_options=True,
        key=key,
        label_visibility=label_visibility,
        placeholder="Type a letter to search (e.g. A for AAPL, AMZN...)",
    )
    if not selected:
        return ""
    return ticker_from_label(selected[0])


def ticker_prefix_autocomplete(
    label: str,
    key: str,
    *,
    default: str = "",
    options: list[str] | None = None,
    placeholder: str = "",
    max_suggestions: int = 8,
    label_visibility: str = "visible",
) -> str:
    """Text input with prefix-matched ticker suggestions underneath.

    - User sees `label` + a text input.
    - As soon as the user types at least one character, up to `max_suggestions`
      tickers whose ticker STARTS WITH the typed prefix appear as chip buttons.
    - Clicking a chip fills the input via session_state and reruns.

    Returns the current input value.

    `key` is the stable identifier for this widget — the function builds
    `_tx_<key>` for the text input and `_sugg_<key>_<ticker>` for each chip.
    Callers that want to seed a value (e.g. a handoff from another tab)
    should write to `st.session_state[f"_tx_{key}"]` BEFORE calling this.
    """
    opts = options if options is not None else POPULAR_TICKERS
    input_key = f"_tx_{key}"
    # Seed a default value only on the first render.
    if default and input_key not in st.session_state:
        st.session_state[input_key] = default

    typed = st.text_input(
        label,
        key=input_key,
        placeholder=placeholder,
        label_visibility=label_visibility,
    )
    prefix = (typed or "").strip().upper()
    if prefix:
        matches = [o for o in opts if o.upper().startswith(prefix)]
        # Don't bother suggesting what the user already typed exactly.
        matches = [m for m in matches if m.upper() != prefix][:max_suggestions]
        if matches:
            chip_cols = st.columns(min(len(matches), 4))
            for i, match in enumerate(matches):
                with chip_cols[i % 4]:
                    if st.button(
                        match,
                        key=f"_sugg_{key}_{match}",
                        type="tertiary",
                        use_container_width=True,
                    ):
                        st.session_state[input_key] = match
                        st.rerun()
    return typed
