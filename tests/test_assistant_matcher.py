"""Tests for pure-Python matching of stocks to InvestmentGoals."""

import pytest

from finance_dashboard.assistant.matcher import score_match
from finance_dashboard.assistant.models import InvestmentGoal
from finance_dashboard.recommender.models import RiskLabel, StockRecommendation


def stock(ticker: str, *, vol: float, ret: float, label: RiskLabel = RiskLabel.MEDIUM) -> StockRecommendation:
    return StockRecommendation(
        ticker=ticker,
        risk_label=label,
        risk_score=50.0,
        group_id=0,
        features={"volatility": vol, "returns": ret},
    )


def base_goal(**overrides) -> InvestmentGoal:
    defaults = dict(
        target_annual_return_pct=10.0,
        max_volatility_pct=25.0,
        sectors=["Tech"],
        max_recommendations=5,
    )
    defaults.update(overrides)
    return InvestmentGoal(**defaults)


def test_perfect_match_scores_100():
    goal = base_goal(target_annual_return_pct=10.0, max_volatility_pct=25.0)
    # Historical annualized return of 10%, volatility exactly at cap.
    s = stock("A", vol=0.25, ret=0.10)
    assert score_match(s, goal) == pytest.approx(100.0)


def test_score_bounded_zero_to_hundred():
    goal = base_goal()
    for vol, ret in [(0.01, -0.99), (5.0, 5.0), (0.25, 0.10)]:
        score = score_match(stock("A", vol=vol, ret=ret), goal)
        assert 0.0 <= score <= 100.0


def test_lower_volatility_than_cap_does_not_penalize():
    """Under-cap volatility is not a defect — the cap is a maximum, not a target."""
    goal = base_goal(max_volatility_pct=30.0, target_annual_return_pct=10.0)
    at_cap = score_match(stock("A", vol=0.30, ret=0.10), goal)
    below_cap = score_match(stock("B", vol=0.15, ret=0.10), goal)
    assert below_cap >= at_cap


def test_over_volatility_cap_reduces_score():
    goal = base_goal(max_volatility_pct=20.0, target_annual_return_pct=10.0)
    within = score_match(stock("A", vol=0.20, ret=0.10), goal)
    over = score_match(stock("B", vol=0.40, ret=0.10), goal)
    assert over < within


def test_return_below_target_reduces_score():
    goal = base_goal(target_annual_return_pct=15.0, max_volatility_pct=30.0)
    on_target = score_match(stock("A", vol=0.20, ret=0.15), goal)
    below = score_match(stock("B", vol=0.20, ret=0.05), goal)
    assert below < on_target


def test_return_above_target_does_not_penalize():
    goal = base_goal(target_annual_return_pct=10.0, max_volatility_pct=30.0)
    on_target = score_match(stock("A", vol=0.20, ret=0.10), goal)
    exceeds = score_match(stock("B", vol=0.20, ret=0.20), goal)
    assert exceeds >= on_target


def test_missing_features_score_zero_not_crash():
    goal = base_goal()
    empty = StockRecommendation(
        ticker="X", risk_label=RiskLabel.LOW, risk_score=0.0, group_id=0, features={}
    )
    assert score_match(empty, goal) == pytest.approx(0.0)
