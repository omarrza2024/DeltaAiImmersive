"""End-to-end tests for InvestmentAssistant with mocked LLM and synthetic data."""

import json

import numpy as np
import pandas as pd
import pytest

from finance_dashboard.assistant.assistant import (
    AssistantReply,
    InvestmentAssistant,
    LLMUnavailableError,
)
from finance_dashboard.assistant.models import InvestmentGoal
from finance_dashboard.recommender.recommender import StockRecommender
from finance_dashboard.recommender.universe import UniverseProvider


def synthetic_prices(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    """One high-vol, one low-vol, both positive drift — deterministic."""
    n = 260
    idx = pd.bdate_range("2024-01-02", periods=n)
    rng = np.random.default_rng(42)
    out = {}
    for i, t in enumerate(tickers):
        vol = 0.005 if t.startswith("CALM") else 0.030
        out[t] = 100 * np.exp(np.cumsum(rng.normal(0.0004, vol, n)))
    return pd.DataFrame(out, index=idx)


class ScriptedLLM:
    """chat_completion callable that returns queued responses in order."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[str, list[dict], bool]] = []

    def __call__(self, system: str, messages: list[dict], *, json_mode: bool = False) -> str:
        self.calls.append((system, messages, json_mode))
        if not self._responses:
            raise AssertionError("LLM called more times than scripted responses")
        return self._responses.pop(0)


def make_assistant(llm_responses: list[str]) -> tuple[InvestmentAssistant, ScriptedLLM]:
    llm = ScriptedLLM(llm_responses)
    provider = UniverseProvider(
        holdings_fetcher=lambda etf: ["CALM1", "CALM2", "WILD1", "WILD2", "WILD3"]
    )
    recommender = StockRecommender(fetcher=synthetic_prices, universe_provider=provider)
    assistant = InvestmentAssistant(chat_completion=llm, recommender=recommender)
    return assistant, llm


def _goal_json(**overrides) -> str:
    goal = {
        "target_annual_return_pct": None,
        "max_volatility_pct": None,
        "sectors": None,
        "max_recommendations": None,
    }
    goal.update(overrides)
    return json.dumps(goal)


def test_asks_next_missing_field_when_goal_incomplete():
    # 1st LLM call: parser JSON. 2nd: LLM-composed warm follow-up.
    composer_response = "Great, 10% is a solid target. What's your comfort with risk?"
    assistant, llm = make_assistant([
        _goal_json(target_annual_return_pct=10.0),
        composer_response,
    ])
    state = assistant.new_state()
    reply = assistant.handle_message(state, "I want about 10% return")
    assert isinstance(reply, AssistantReply)
    assert state.goal.target_annual_return_pct == 10.0
    # The LLM composes the response; here we scripted one that mentions risk.
    assert ("volatility" in reply.message.lower() or "risk" in reply.message.lower())
    assert reply.recommendations == []


def test_gathering_response_composed_by_llm_not_hardcoded():
    """The follow-up prompt should be an LLM call, not a hardcoded string."""
    composer_response = "Sounds great — let's talk about risk next."
    assistant, llm = make_assistant([
        _goal_json(target_annual_return_pct=10.0),
        composer_response,
    ])
    state = assistant.new_state()
    reply = assistant.handle_message(state, "10% return please")

    # Two LLM calls: parser (JSON) + composer (natural language).
    assert len(llm.calls) == 2
    _, _, json_mode_first = llm.calls[0]
    _, _, json_mode_second = llm.calls[1]
    assert json_mode_first is True   # parser
    assert json_mode_second is False # composer
    # And the reply is the composer's output verbatim.
    assert composer_response in reply.message


def test_runs_pipeline_when_goal_complete_and_returns_top_n():
    goal_json = _goal_json(
        target_annual_return_pct=8.0,
        max_volatility_pct=50.0,
        sectors=["Tech"],
        max_recommendations=2,
    )
    explanation = "Here are your top picks based on your goal."
    assistant, llm = make_assistant([goal_json, explanation])
    state = assistant.new_state()
    # Force custom tickers so we don't depend on the Tech universe roster.
    state.custom_tickers = ("CALM1", "CALM2", "WILD1", "WILD2", "WILD3")

    reply = assistant.handle_message(state, "10% return, moderate risk, tech, top 2")

    assert len(reply.recommendations) == 2
    scores = [r.match_score for r in reply.recommendations]
    assert scores == sorted(scores, reverse=True)  # sorted best-first
    assert "historical" in reply.message.lower()  # disclaimer required


def test_no_stocks_pass_constraint_reports_closest_matches():
    goal_json = _goal_json(
        target_annual_return_pct=50.0,   # unrealistically high
        max_volatility_pct=1.0,          # unrealistically strict
        sectors=["Tech"],
        max_recommendations=3,
    )
    pushback = "Your constraints are unusually strict; showing closest matches."
    assistant, llm = make_assistant([goal_json, pushback])
    state = assistant.new_state()
    state.custom_tickers = ("CALM1", "CALM2", "WILD1", "WILD2", "WILD3")

    reply = assistant.handle_message(state, "50% return, near-zero vol, tech, 3 picks")

    assert reply.recommendations, "should still return closest matches, not empty"
    assert reply.no_perfect_match is True
    assert "closest" in reply.message.lower() or "strict" in reply.message.lower()


def test_llm_failure_surfaces_error_not_hallucination():
    def broken_llm(system, messages, *, json_mode=False):
        raise RuntimeError("groq: 503 service unavailable")

    provider = UniverseProvider(holdings_fetcher=lambda etf: ["CALM1", "CALM2", "WILD1"])
    recommender = StockRecommender(fetcher=synthetic_prices, universe_provider=provider)
    assistant = InvestmentAssistant(chat_completion=broken_llm, recommender=recommender)
    state = assistant.new_state()

    with pytest.raises(LLMUnavailableError, match="groq"):
        assistant.handle_message(state, "10% return")


def test_data_fetch_failure_surfaces_error():
    def empty_fetcher(tickers, period):
        return pd.DataFrame()  # nothing returned

    goal_json = _goal_json(
        target_annual_return_pct=8.0,
        max_volatility_pct=50.0,
        sectors=["Tech"],
        max_recommendations=2,
    )
    llm = ScriptedLLM([goal_json])
    provider = UniverseProvider(holdings_fetcher=lambda etf: ["A", "B", "C"])
    recommender = StockRecommender(fetcher=empty_fetcher, universe_provider=provider)
    assistant = InvestmentAssistant(chat_completion=llm, recommender=recommender)
    state = assistant.new_state()

    reply = assistant.handle_message(state, "10% return, low risk, tech, 3 picks")
    assert reply.recommendations == []
    assert "data" in reply.message.lower() or "unavailable" in reply.message.lower()


def test_followup_chat_does_not_rerun_recommender():
    """After recommendations, chit-chat must NOT re-trigger the pipeline."""
    goal_json = _goal_json(
        target_annual_return_pct=8.0,
        max_volatility_pct=50.0,
        sectors=["Tech"],
        max_recommendations=2,
    )
    initial_explain = "Here are your top picks."
    chat_reply = "Sure, happy to just chat — the picks stand for now."
    assistant, llm = make_assistant([goal_json, initial_explain, goal_json, chat_reply])
    state = assistant.new_state()
    state.custom_tickers = ("CALM1", "CALM2", "WILD1", "WILD2", "WILD3")

    first = assistant.handle_message(state, "10% return, moderate, tech, top 2")
    assert first.recommendations, "first turn should produce recommendations"

    second = assistant.handle_message(state, "just chat with me, no more picks")
    assert second.recommendations == [], "follow-up chat must not re-list picks"
    # Only 4 LLM calls: parse + explain + parse + chat. Not another explain call.
    assert len(llm.calls) == 4


def test_followup_with_changed_constraint_reruns_recommender():
    """When the user changes a constraint, we DO re-recommend."""
    goal_json = _goal_json(
        target_annual_return_pct=8.0,
        max_volatility_pct=50.0,
        sectors=["Tech"],
        max_recommendations=2,
    )
    goal_json_higher = _goal_json(
        target_annual_return_pct=20.0,
        max_volatility_pct=50.0,
        sectors=["Tech"],
        max_recommendations=2,
    )
    assistant, llm = make_assistant(
        [goal_json, "picks A", goal_json_higher, "picks B"]
    )
    state = assistant.new_state()
    state.custom_tickers = ("CALM1", "CALM2", "WILD1", "WILD2", "WILD3")

    assistant.handle_message(state, "10% return, moderate, tech, top 2")
    second = assistant.handle_message(state, "actually make it 20%")
    assert second.recommendations, "changed target return must trigger a re-run"


def test_ticker_mention_injects_sec_context_into_chat(monkeypatch):
    """When a user asks about a specific recommended stock, its SEC summary
    is injected into the LLM's system context — but only for that ticker."""
    import finance_dashboard.assistant.assistant as asst_mod

    monkeypatch.setattr(
        asst_mod,
        "financials_summary",
        lambda t: f"[SEC-CONTEXT for {t}: revenue up, cash strong]",
    )

    goal_json = _goal_json(
        target_annual_return_pct=8.0,
        max_volatility_pct=50.0,
        sectors=["Tech"],
        max_recommendations=2,
    )
    assistant, llm = make_assistant([goal_json, "picks", goal_json, "reply"])
    state = assistant.new_state()
    state.custom_tickers = ("CALM1", "CALM2", "WILD1", "WILD2", "WILD3")

    first = assistant.handle_message(state, "10% return, moderate, tech, top 2")
    top_ticker    = first.recommendations[0].stock.ticker
    other_ticker  = first.recommendations[1].stock.ticker
    assistant.handle_message(state, f"tell me about {top_ticker}'s fundamentals")

    # The most recent LLM call is the chat reply — its system+messages must
    # include the SEC context for the ticker the user mentioned, and NOT the other.
    system, messages, _ = llm.calls[-1]
    context_dump = system + " " + " ".join(m["content"] for m in messages)
    assert f"SEC-CONTEXT for {top_ticker}" in context_dump
    assert f"SEC-CONTEXT for {other_ticker}" not in context_dump


def test_only_missing_count_defaults_to_five_and_runs_recommender():
    """When return/risk/sectors are set and only count is missing, don't stall —
    default to 5 and run the pipeline. This fixes the "hangs asking how many"
    UX bug that led the LLM to invent stock names instead."""
    goal_json = _goal_json(
        target_annual_return_pct=10.0,
        max_volatility_pct=25.0,
        sectors=["Medical"],
        max_recommendations=None,  # user didn't specify — should default
    )
    assistant, llm = make_assistant([goal_json, "here are your picks"])
    state = assistant.new_state()
    state.custom_tickers = ("CALM1", "CALM2", "WILD1", "WILD2", "WILD3")

    reply = assistant.handle_message(state, "give me all that fit my criteria")
    assert reply.recommendations, "should have produced recommendations (not asked again)"
    assert state.goal.max_recommendations == 5  # auto-defaulted


def test_portfolio_context_is_injected_into_chat(monkeypatch):
    """When the user has holdings, the chat call must include a portfolio summary."""
    import finance_dashboard.assistant.assistant as asst_mod

    # Fake portfolio provider — returns a signal-string we can grep for.
    monkeypatch.setattr(
        asst_mod,
        "get_portfolio_summary",
        lambda username: "[PORTFOLIO-SUMMARY: AAPL 60% Tech, JNJ 40% Healthcare]",
    )

    goal_json = _goal_json(
        target_annual_return_pct=8.0,
        max_volatility_pct=50.0,
        sectors=["Tech"],
        max_recommendations=2,
    )
    assistant, llm = make_assistant([goal_json, "picks", goal_json, "reply"])
    state = assistant.new_state()
    state.custom_tickers = ("CALM1", "CALM2", "WILD1", "WILD2", "WILD3")
    state.username = "alice"

    assistant.handle_message(state, "10% return, moderate, tech, top 2")
    assistant.handle_message(state, "chat with me")

    # Chat call is the last one — its context must include the portfolio summary.
    system, messages, _ = llm.calls[-1]
    context_dump = system + " " + " ".join(m["content"] for m in messages)
    assert "PORTFOLIO-SUMMARY" in context_dump


def test_unrelated_chat_does_not_fetch_sec(monkeypatch):
    """If the user isn't asking about fundamentals, don't spend an SEC fetch."""
    import finance_dashboard.assistant.assistant as asst_mod

    calls: list[str] = []
    monkeypatch.setattr(
        asst_mod,
        "financials_summary",
        lambda t: calls.append(t) or f"[SEC for {t}]",
    )

    goal_json = _goal_json(
        target_annual_return_pct=8.0,
        max_volatility_pct=50.0,
        sectors=["Tech"],
        max_recommendations=2,
    )
    assistant, llm = make_assistant([goal_json, "picks", goal_json, "chat back"])
    state = assistant.new_state()
    state.custom_tickers = ("CALM1", "CALM2", "WILD1", "WILD2", "WILD3")

    assistant.handle_message(state, "10% return, moderate, tech, top 2")
    assistant.handle_message(state, "hi, just saying hello")

    assert calls == []  # no SEC lookup for a generic hello
