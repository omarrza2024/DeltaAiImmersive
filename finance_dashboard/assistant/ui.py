"""Streamlit conversational chat UI for the investment assistant."""

from __future__ import annotations

import html
import textwrap

import plotly.graph_objects as go
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
from finance_dashboard.assistant.models import Recommendation
from finance_dashboard.data import company_name, fetch_history
from finance_dashboard.recommender.models import RiskLabel

_STATE_KEY = "assistant_state"
_HISTORY_KEY = "assistant_history"  # list of (role, content, recommendations|None)

_USER_AVATAR = "🧑"
_BOT_AVATAR = "💠"

_RISK_CHIP_CLASS = {
    RiskLabel.LOW: "chip low",
    RiskLabel.MEDIUM: "chip med",
    RiskLabel.HIGH_POSITIVE: "chip high",
    RiskLabel.HIGH_NEGATIVE: "chip high",
}


@st.cache_data(ttl=3600, show_spinner=False)
def _sparkline_html(ticker: str) -> str | None:
    """Render a small 1y price sparkline as an inline SVG-like plotly figure."""
    try:
        prices = fetch_history((ticker,), "1y")
    except Exception:
        return None
    if prices is None or prices.empty or ticker not in prices.columns:
        return None
    series = prices[ticker].dropna()
    if series.empty:
        return None
    color = "#2DD4BF" if series.iloc[-1] >= series.iloc[0] else "#F87171"
    fig = go.Figure(go.Scatter(x=series.index, y=series.values, mode="lines",
                               line=dict(color=color, width=1.6), hoverinfo="skip"))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=54,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_memo_markdown(ticker: str) -> str:
    return render_memo_markdown(fetch_company_memo(ticker))


def _init_state() -> InvestmentAssistant:
    if _STATE_KEY not in st.session_state:
        assistant = InvestmentAssistant()
        st.session_state[_STATE_KEY] = (assistant, assistant.new_state())
        st.session_state[_HISTORY_KEY] = [
            (
                "assistant",
                "Hi — I'll help you find stocks that fit your goals. "
                "What annual return are you targeting (e.g. 8%, 12%)?",
                None,
            )
        ]
    return st.session_state[_STATE_KEY][0]


def _render_card(rank: int, rec: Recommendation) -> None:
    stock = rec.stock
    ticker = stock.ticker
    name = company_name(ticker) or ""
    returns = stock.features.get("returns", 0) * 100
    volatility = stock.features.get("volatility", 0) * 100
    risk_chip = _RISK_CHIP_CLASS.get(stock.risk_label, "chip")
    ret_class = "chip pos" if returns >= 0 else "chip neg"
    rank_html = '<span class="rec-rank">★ TOP MATCH</span>' if rank == 1 else ""

    card_html = textwrap.dedent(
        f"""\
        <div class="rec-card">
        <div class="rec-header">{rank_html}<span class="rec-ticker">{html.escape(ticker)}</span><span class="rec-name">{html.escape(name)}</span></div>
        <div class="match-bar"><span class="label">Match score</span><span class="track"><span class="fill" style="width:{rec.match_score:.0f}%"></span></span><span class="val">{rec.match_score:.0f}</span></div>
        <div class="chip-row"><span class="{ret_class}">Return {returns:+.1f}%</span><span class="chip">Volatility {volatility:.1f}%</span><span class="{risk_chip}">{html.escape(stock.risk_label.value)}</span></div>
        </div>
        """
    )
    with st.container(border=True):
        st.markdown(card_html, unsafe_allow_html=True)
        fig = _sparkline_html(ticker)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with st.expander(f"📄 SEC financials — {ticker}"):
            with st.spinner(f"Fetching {ticker} from SEC EDGAR…"):
                try:
                    st.markdown(_cached_memo_markdown(ticker))
                except FinancialsUnavailableError as exc:
                    st.error(str(exc))


def _render_recommendations(reply: AssistantReply) -> None:
    if not reply.recommendations:
        return
    if reply.no_perfect_match:
        st.warning("No stocks fully match your constraints — showing the closest matches.")
    for i, rec in enumerate(reply.recommendations, start=1):
        _render_card(i, rec)


def render_assistant_tab() -> None:
    st.markdown(
        "Chat with an AI assistant that asks about your goals, then uses the "
        "recommender's real calculations to suggest matching stocks."
    )

    if not get_api_key():
        st.error(
            "**Groq API key missing.** Add `GROQ_API_KEY = \"gsk_...\"` to "
            "`.streamlit/secrets.toml`, then rerun the app. Chat is disabled "
            "until then to prevent hallucinated advice."
        )
        return

    assistant = _init_state()
    _, state = st.session_state[_STATE_KEY]

    for role, content, reply in st.session_state[_HISTORY_KEY]:
        avatar = _USER_AVATAR if role == "user" else _BOT_AVATAR
        with st.chat_message(role, avatar=avatar):
            st.markdown(content)
            if reply is not None:
                _render_recommendations(reply)

    prompt = st.chat_input("Type your answer or ask a follow-up…")
    if not prompt:
        return

    st.session_state[_HISTORY_KEY].append(("user", prompt, None))
    with st.chat_message("user", avatar=_USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=_BOT_AVATAR):
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
