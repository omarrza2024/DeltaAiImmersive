"""Tests for GoalParser — LLM extraction of InvestmentGoal from chat messages."""

import json

import pytest

from finance_dashboard.assistant.goal_parser import GoalParser, next_missing_question
from finance_dashboard.assistant.models import InvestmentGoal
from finance_dashboard.models import UserInputError


def fake_llm(response_json: dict):
    """Build a chat_completion callable that always returns the given JSON."""
    def _call(system: str, messages: list[dict], *, json_mode: bool = False) -> str:
        return json.dumps(response_json)
    return _call


def test_parser_extracts_all_fields_from_one_message():
    parser = GoalParser(
        chat_completion=fake_llm(
            {
                "target_annual_return_pct": 12.0,
                "max_volatility_pct": 25.0,
                "sectors": ["Tech"],
                "max_recommendations": 3,
            }
        )
    )
    goal = parser.update(InvestmentGoal(), [{"role": "user", "content": "12% return, moderate risk, tech, top 3"}])
    assert goal.target_annual_return_pct == 12.0
    assert goal.max_volatility_pct == 25.0
    assert goal.sectors == ["Tech"]
    assert goal.max_recommendations == 3
    assert goal.is_complete()


def test_parser_leaves_existing_fields_alone_when_llm_returns_null():
    goal = InvestmentGoal(target_annual_return_pct=10.0, max_volatility_pct=None, sectors=[], max_recommendations=None)
    parser = GoalParser(
        chat_completion=fake_llm(
            {
                "target_annual_return_pct": None,
                "max_volatility_pct": 20.0,
                "sectors": None,
                "max_recommendations": None,
            }
        )
    )
    updated = parser.update(goal, [{"role": "user", "content": "no more than 20% vol"}])
    assert updated.target_annual_return_pct == 10.0
    assert updated.max_volatility_pct == 20.0
    assert updated.sectors == []


def test_parser_rejects_unknown_sector():
    parser = GoalParser(
        chat_completion=fake_llm(
            {
                "target_annual_return_pct": 10.0,
                "max_volatility_pct": 20.0,
                "sectors": ["Crypto"],
                "max_recommendations": 3,
            }
        )
    )
    with pytest.raises(UserInputError, match="sector"):
        parser.update(InvestmentGoal(), [{"role": "user", "content": "crypto stuff"}])


def test_missing_field_question_covers_each_slot_in_order():
    # A brand-new goal is missing everything — first question is about return.
    q = next_missing_question(InvestmentGoal())
    assert q is not None and "return" in q.lower()

    # After return is set, next missing is volatility.
    q = next_missing_question(InvestmentGoal(target_annual_return_pct=10.0))
    assert q is not None and ("volatility" in q.lower() or "risk" in q.lower())

    # After return + volatility, next is sectors.
    q = next_missing_question(
        InvestmentGoal(target_annual_return_pct=10.0, max_volatility_pct=20.0)
    )
    assert q is not None and "sector" in q.lower()

    # After all three, next is count.
    q = next_missing_question(
        InvestmentGoal(
            target_annual_return_pct=10.0, max_volatility_pct=20.0, sectors=["Tech"]
        )
    )
    assert q is not None and ("recommendation" in q.lower() or "how many" in q.lower())

    # Fully specified goal has no more questions.
    q = next_missing_question(
        InvestmentGoal(
            target_annual_return_pct=10.0,
            max_volatility_pct=20.0,
            sectors=["Tech"],
            max_recommendations=3,
        )
    )
    assert q is None
