"""Tests for portfolio enrichment: current value, P&L, weights, sector rollup, LLM summary."""

import pytest

from finance_dashboard.portfolio import analytics


def _positions():
    return [
        {"ticker": "AAPL", "shares": 10, "avg_cost": 150.0},
        {"ticker": "MSFT", "shares": 5,  "avg_cost": 300.0},
        {"ticker": "JNJ",  "shares": 20, "avg_cost": 100.0},
    ]


def _prices():
    # Current prices — deterministic
    return {"AAPL": 200.0, "MSFT": 400.0, "JNJ": 90.0}


def _sectors():
    return {"AAPL": "Technology", "MSFT": "Technology", "JNJ": "Healthcare"}


def test_enrich_computes_market_value_and_pnl():
    enriched = analytics.enrich(_positions(), _prices(), _sectors())
    by_t = {p["ticker"]: p for p in enriched["positions"]}

    # AAPL: 10 * 200 = 2000; cost 10 * 150 = 1500; PnL 500, +33.33%
    assert by_t["AAPL"]["market_value"] == pytest.approx(2000.0)
    assert by_t["AAPL"]["cost_basis"]   == pytest.approx(1500.0)
    assert by_t["AAPL"]["pnl"]          == pytest.approx(500.0)
    assert by_t["AAPL"]["pnl_pct"]      == pytest.approx(33.333333, rel=1e-4)

    # JNJ: 20 * 90 = 1800; cost 2000; PnL -200, -10%
    assert by_t["JNJ"]["pnl"]     == pytest.approx(-200.0)
    assert by_t["JNJ"]["pnl_pct"] == pytest.approx(-10.0)


def test_enrich_computes_totals():
    e = analytics.enrich(_positions(), _prices(), _sectors())
    # Total mv: 2000 + 2000 + 1800 = 5800
    # Total cost: 1500 + 1500 + 2000 = 5000
    # Total pnl: +800, +16%
    assert e["total_value"]   == pytest.approx(5800.0)
    assert e["total_cost"]    == pytest.approx(5000.0)
    assert e["total_pnl"]     == pytest.approx(800.0)
    assert e["total_pnl_pct"] == pytest.approx(16.0)


def test_enrich_computes_weights_summing_to_100():
    e = analytics.enrich(_positions(), _prices(), _sectors())
    weights = [p["weight_pct"] for p in e["positions"]]
    assert sum(weights) == pytest.approx(100.0)


def test_sector_allocation():
    e = analytics.enrich(_positions(), _prices(), _sectors())
    # Tech: 2000 + 2000 = 4000 → 68.97%; Healthcare: 1800 → 31.03%
    alloc = e["sector_allocation"]
    assert alloc["Technology"] == pytest.approx(68.9655, rel=1e-3)
    assert alloc["Healthcare"] == pytest.approx(31.0344, rel=1e-3)


def test_missing_price_gracefully_zeroes_position():
    positions = [{"ticker": "GHOST", "shares": 5, "avg_cost": 100.0}]
    e = analytics.enrich(positions, prices={}, sectors={})
    pos = e["positions"][0]
    assert pos["market_value"] == 0.0
    assert pos["current_price"] is None


def test_summary_for_llm_lists_tickers_and_sectors():
    e = analytics.enrich(_positions(), _prices(), _sectors())
    text = analytics.summary_for_llm(e)
    assert "AAPL" in text and "MSFT" in text and "JNJ" in text
    assert "Technology" in text and "Healthcare" in text
    assert "$5,800" in text or "5800" in text  # total value shown


def test_summary_for_empty_portfolio():
    e = analytics.enrich([], prices={}, sectors={})
    text = analytics.summary_for_llm(e)
    assert "empty" in text.lower() or "no position" in text.lower()
