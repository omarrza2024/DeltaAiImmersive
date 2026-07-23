"""End-to-end tests for StockRecommender with a fake market data fetcher."""

import numpy as np
import pandas as pd
import pytest

from finance_dashboard.models import UserInputError
from finance_dashboard.recommender.models import (
    FeatureName,
    RecommendationRequest,
    RiskLabel,
    StrategyName,
)
from finance_dashboard.recommender.recommender import StockRecommender
from finance_dashboard.recommender.universe import UniverseProvider


def synthetic_prices(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    """Deterministic fake fetcher: vol/drift keyed by ticker prefix; UNKNOWN missing."""
    n = 200
    idx = pd.bdate_range("2024-01-02", periods=n)
    out = {}
    for i, t in enumerate(tickers):
        if t.startswith("UNKNOWN"):
            continue  # simulates a symbol yfinance can't resolve
        rng = np.random.default_rng(100 + i)
        if t.startswith("LOW"):
            vol, drift = 0.002, 0.0004
        elif t.startswith("MID"):
            vol, drift = 0.015, 0.0004
        elif t.startswith("HIGHNEG"):
            vol, drift = 0.05, -0.004
        else:  # HIGH*
            vol, drift = 0.05, 0.004
        out[t] = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    return pd.DataFrame(out, index=idx)


UNIVERSE = ("LOW1", "LOW2", "MID1", "MID2", "HIGHPOS1", "HIGHNEG1")


def make_recommender() -> StockRecommender:
    return StockRecommender(fetcher=synthetic_prices)


def base_request(**overrides) -> RecommendationRequest:
    defaults = dict(
        custom_tickers=UNIVERSE,
        features=(FeatureName.VOLATILITY, FeatureName.RETURNS),
        strategy=StrategyName.KMEANS,
    )
    defaults.update(overrides)
    return RecommendationRequest(**defaults)


def test_every_resolved_ticker_gets_exactly_one_label():
    result = make_recommender().recommend(base_request())
    assert sorted(r.ticker for r in result.recommendations) == sorted(UNIVERSE)
    for rec in result.recommendations:
        assert isinstance(rec.risk_label, RiskLabel)
        assert 0 <= rec.risk_score <= 100


def test_risk_tiers_match_engineered_volatility():
    # Cluster on volatility alone; returns still drive the High +/- split.
    result = make_recommender().recommend(base_request(features=(FeatureName.VOLATILITY,)))
    by_ticker = {r.ticker: r.risk_label for r in result.recommendations}
    assert by_ticker["LOW1"] == RiskLabel.LOW
    assert by_ticker["LOW2"] == RiskLabel.LOW
    assert by_ticker["MID1"] == RiskLabel.MEDIUM
    assert by_ticker["HIGHPOS1"] == RiskLabel.HIGH_POSITIVE
    assert by_ticker["HIGHNEG1"] == RiskLabel.HIGH_NEGATIVE


def test_unfetchable_ticker_is_skipped_and_reported():
    request = base_request(custom_tickers=UNIVERSE + ("UNKNOWN1",))
    result = make_recommender().recommend(request)
    assert "UNKNOWN1" in result.skipped
    assert sorted(r.ticker for r in result.recommendations) == sorted(UNIVERSE)


def test_results_sorted_high_to_low_risk():
    """Output order (score desc) must walk the buckets High -> Medium -> Low."""
    result = make_recommender().recommend(base_request(features=(FeatureName.VOLATILITY,)))
    tier = {
        RiskLabel.LOW: 0,
        RiskLabel.MEDIUM: 1,
        RiskLabel.HIGH_POSITIVE: 2,
        RiskLabel.HIGH_NEGATIVE: 2,
    }
    tiers = [tier[r.risk_label] for r in result.recommendations]
    assert tiers == sorted(tiers, reverse=True)


def test_deterministic_across_runs():
    a = make_recommender().recommend(base_request())
    b = make_recommender().recommend(base_request())
    assert [(r.ticker, r.risk_label, r.risk_score) for r in a.recommendations] == [
        (r.ticker, r.risk_label, r.risk_score) for r in b.recommendations
    ]


def test_rules_strategy_end_to_end():
    result = make_recommender().recommend(base_request(strategy=StrategyName.RULES))
    by_ticker = {r.ticker: r.risk_label for r in result.recommendations}
    assert by_ticker["LOW1"] == RiskLabel.LOW
    assert by_ticker["HIGHNEG1"] == RiskLabel.HIGH_NEGATIVE


def test_clustering_requires_at_least_three_stocks():
    request = base_request(custom_tickers=("LOW1", "HIGHPOS1"))
    with pytest.raises(UserInputError, match="at least 3"):
        make_recommender().recommend(request)


def test_request_requires_at_least_one_feature():
    with pytest.raises(UserInputError, match="feature"):
        base_request(features=()).validate()


def test_request_rejects_unknown_strategy_and_feature_names():
    with pytest.raises(UserInputError, match="strategy"):
        RecommendationRequest.from_user_input(
            custom_tickers_text="AAPL, MSFT, GOOG",
            feature_names=["volatility"],
            strategy_name="quantum",
        )
    with pytest.raises(UserInputError, match="feature"):
        RecommendationRequest.from_user_input(
            custom_tickers_text="AAPL, MSFT, GOOG",
            feature_names=["astrology"],
            strategy_name="kmeans",
        )


def test_portfolio_profile_summarizes_buckets():
    result = make_recommender().recommend(base_request(features=(FeatureName.VOLATILITY,)))
    profile = result.profile
    assert profile.counts[RiskLabel.LOW] == 2
    assert profile.counts[RiskLabel.MEDIUM] == 2
    assert profile.counts[RiskLabel.HIGH_POSITIVE] == 1
    assert profile.counts[RiskLabel.HIGH_NEGATIVE] == 1
    assert profile.total == 6


def test_recommendation_risk_label_is_always_enum():
    """Regression: pandas may coerce str-Enum values to plain str during the
    DataFrame roundtrip inside RiskLabeler, breaking `r.risk_label.value`."""
    from finance_dashboard.recommender.models import StockRecommendation

    # Direct construction with a plain string simulates what pandas returns on
    # Streamlit Cloud (pandas future.infer_string coerces our str-Enum).
    rec = StockRecommendation(
        ticker="AAPL", risk_label="Low Risk", risk_score=42.0, group_id=0
    )
    assert isinstance(rec.risk_label, RiskLabel)
    assert rec.risk_label is RiskLabel.LOW
    assert rec.risk_label.value == "Low Risk"

    # End-to-end: every recommendation must have an enum, not a str.
    result = make_recommender().recommend(base_request())
    for r in result.recommendations:
        assert isinstance(r.risk_label, RiskLabel), f"{r.ticker} label is {type(r.risk_label)}"
        _ = r.risk_label.value  # would AttributeError if str slipped through


def test_recommender_uses_universe_provider_for_etf():
    provider = UniverseProvider(holdings_fetcher=lambda etf: list(UNIVERSE))
    recommender = StockRecommender(fetcher=synthetic_prices, universe_provider=provider)
    result = recommender.recommend(base_request(custom_tickers=(), etf="XLK"))
    assert sorted(r.ticker for r in result.recommendations) == sorted(UNIVERSE)
