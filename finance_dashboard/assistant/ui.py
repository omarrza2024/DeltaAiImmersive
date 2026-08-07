"""Streamlit conversational chat UI for the investment assistant."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from finance_dashboard.assistant.assistant import (
    AssistantReply,
    InvestmentAssistant,
    LLMUnavailableError,
)
from finance_dashboard.assistant.financials import (
    FinancialsUnavailableError,
    fetch_company_memo,
    render_memo_markdown,
)
from finance_dashboard.assistant.llm import get_api_key
from finance_dashboard.recommender.models import RiskLabel

_STATE_KEY = "assistant_state"
_HISTORY_KEY = "assistant_history"  # list of (role, content, recommendations|None)

_LABEL_COLORS = {
    RiskLabel.LOW: "🟢",
    RiskLabel.MEDIUM: "🟡",
    RiskLabel.HIGH_POSITIVE: "🟠",
    RiskLabel.HIGH_NEGATIVE: "🔴",
}


def _init_state() -> InvestmentAssistant:
    if _STATE_KEY not in st.session_state:
        assistant = InvestmentAssistant()
        st.session_state[_STATE_KEY] = (assistant, assistant.new_state())
        st.session_state[_HISTORY_KEY] = [
            (
                "assistant",
                "Hi! I can suggest stocks that match your investment goals. "
                "Let's start — what annual return are you targeting (e.g. 8%, 12%)?",
                None,
            )
        ]
    return st.session_state[_STATE_KEY][0]


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_memo_markdown(ticker: str) -> str:
    """Cache the memo markdown per ticker for the session — SEC fetches are slow."""
    return render_memo_markdown(fetch_company_memo(ticker))


def _render_recommendations(reply: AssistantReply, key_prefix: str = "") -> None:
    if not reply.recommendations:
        return
    if reply.no_perfect_match:
        st.warning("No stocks fully match your constraints — showing the closest matches.")
    df = pd.DataFrame(
        [
            {
                "": _LABEL_COLORS.get(r.stock.risk_label, ""),
                "Ticker": r.stock.ticker,
                "Match": f"{r.match_score:.0f}",
                "Ann. return %": f"{r.stock.features.get('returns', 0) * 100:+.1f}",
                "Ann. volatility %": f"{r.stock.features.get('volatility', 0) * 100:.1f}",
                "Risk": r.stock.risk_label.value,
            }
            for r in reply.recommendations
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    for i, rec in enumerate(reply.recommendations):
        ticker = rec.stock.ticker
        with st.expander(f"📄 View SEC financials — {ticker}"):
            with st.spinner(f"Fetching {ticker} from SEC EDGAR…"):
                try:
                    st.markdown(_cached_memo_markdown(ticker))
                except FinancialsUnavailableError as exc:
                    st.error(str(exc))


def render_assistant_tab() -> None:
    st.markdown(
        "Chat with an AI assistant that will ask about your goals, then use "
        "the recommender's real calculations to suggest matching stocks."
    )

    if not get_api_key():
        st.error(
            "**Groq API key missing.** Add `GROQ_API_KEY = \"gsk_...\"` to "
            "`.streamlit/secrets.toml` (or set the env var), then rerun the app. "
            "Chat is disabled until then to prevent hallucinated advice."
        )
        return

    assistant = _init_state()
    _, state = st.session_state[_STATE_KEY]

    for role, content, reply in st.session_state[_HISTORY_KEY]:
        with st.chat_message(role):
            st.markdown(content)
            if reply is not None:
                _render_recommendations(reply)

    prompt = st.chat_input("Type your answer or ask a follow-up…")
    if not prompt:
        return

    st.session_state[_HISTORY_KEY].append(("user", prompt, None))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                reply = assistant.handle_message(state, prompt)
            except LLMUnavailableError as exc:
                message = f"⚠️ The AI service failed: {exc}. Please try again."
                st.error(message)
                st.session_state[_HISTORY_KEY].append(("assistant", message, None))
                return
        st.markdown(reply.message)
        _render_recommendations(reply)
    st.session_state[_HISTORY_KEY].append(("assistant", reply.message, reply))
