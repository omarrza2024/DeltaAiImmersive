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
    assistant, llm = make_assistant([_goal_json(target_annual_return_pct=10.0)])
    state = assistant.new_state()
    reply = assistant.handle_message(state, "I want about 10% return")
    assert isinstance(reply, AssistantReply)
    # State absorbed the return; next question should be about risk/volatility.
    assert state.goal.target_annual_return_pct == 10.0
    assert ("volatility" in reply.message.lower() or "risk" in reply.message.lower())
    assert reply.recommendations == []


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
