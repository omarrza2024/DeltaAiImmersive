"""Tests for universe resolution: ETFs, predefined universes, custom tickers."""

import pytest

from finance_dashboard.models import UserInputError
from finance_dashboard.recommender.models import RecommendationRequest
from finance_dashboard.recommender.universe import (
    ETF_FALLBACK_HOLDINGS,
    PREDEFINED_UNIVERSES,
    UniverseProvider,
)


def request_with(**kwargs) -> RecommendationRequest:
    return RecommendationRequest(**kwargs)


def test_predefined_universes_exist():
    assert set(PREDEFINED_UNIVERSES) == {"Tech", "Medical", "Finance"}
    for name, tickers in PREDEFINED_UNIVERSES.items():
        assert tickers, f"universe {name} is empty"


def test_universe_selectable_without_etf():
    provider = UniverseProvider()
    tickers = provider.resolve(request_with(universes=("Tech",)))
    assert tickers == list(dict.fromkeys(PREDEFINED_UNIVERSES["Tech"]))


def test_multiple_universes_union_without_etf():
    provider = UniverseProvider()
    tickers = provider.resolve(request_with(universes=("Tech", "Finance")))
    for t in PREDEFINED_UNIVERSES["Tech"]:
        assert t in tickers
    for t in PREDEFINED_UNIVERSES["Finance"]:
        assert t in tickers


def test_etf_uses_live_holdings_when_available():
    provider = UniverseProvider(holdings_fetcher=lambda etf: ["AAA", "BBB"])
    assert provider.resolve(request_with(etf="XLK")) == ["AAA", "BBB"]


def test_etf_falls_back_to_static_holdings_when_fetch_fails():
    def broken(etf):
        raise RuntimeError("network down")

    provider = UniverseProvider(holdings_fetcher=broken)
    assert provider.resolve(request_with(etf="XLK")) == ETF_FALLBACK_HOLDINGS["XLK"]


def test_etf_falls_back_when_fetch_returns_empty():
    provider = UniverseProvider(holdings_fetcher=lambda etf: [])
    assert provider.resolve(request_with(etf="XLV")) == ETF_FALLBACK_HOLDINGS["XLV"]


def test_unknown_etf_rejected():
    provider = UniverseProvider(holdings_fetcher=lambda etf: ["AAA"])
    with pytest.raises(UserInputError, match="Unknown ETF"):
        provider.resolve(request_with(etf="NOPE"))


def test_unknown_universe_rejected():
    provider = UniverseProvider()
    with pytest.raises(UserInputError, match="Unknown universe"):
        provider.resolve(request_with(universes=("Crypto",)))


def test_custom_tickers_merged_and_deduped():
    provider = UniverseProvider(holdings_fetcher=lambda etf: ["AAA", "BBB"])
    tickers = provider.resolve(
        request_with(etf="XLK", custom_tickers=("bbb", " ccc ", "CCC"))
    )
    assert tickers == ["AAA", "BBB", "CCC"]


def test_empty_selection_rejected():
    provider = UniverseProvider()
    with pytest.raises(UserInputError, match="at least one"):
        provider.resolve(request_with())
