"""Shared Streamlit widgets reused across tabs."""

from __future__ import annotations

import streamlit as st

from finance_dashboard.recommender.universe import (
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
