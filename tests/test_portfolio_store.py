"""Tests for portfolio CRUD + strict per-user scoping."""

import pytest
from mongomock import MongoClient

from finance_dashboard.portfolio import store


@pytest.fixture
def db():
    return MongoClient()["tradematrix_test"]


def test_empty_portfolio_returns_empty_positions(db):
    p = store.get_or_create(db, "alice")
    assert p["username"] == "alice"
    assert p["positions"] == []


def test_add_position_stores_correctly(db):
    store.add_position(db, "alice", ticker="AAPL", shares=10, avg_cost=150.0)
    p = store.get_or_create(db, "alice")
    assert len(p["positions"]) == 1
    pos = p["positions"][0]
    assert pos["ticker"] == "AAPL"
    assert pos["shares"] == 10
    assert pos["avg_cost"] == 150.0


def test_ticker_normalized_to_uppercase(db):
    store.add_position(db, "alice", ticker=" aapl ", shares=5, avg_cost=100.0)
    p = store.get_or_create(db, "alice")
    assert p["positions"][0]["ticker"] == "AAPL"


def test_adding_same_ticker_twice_averages_cost_and_sums_shares(db):
    store.add_position(db, "alice", ticker="AAPL", shares=10, avg_cost=100.0)
    store.add_position(db, "alice", ticker="AAPL", shares=10, avg_cost=200.0)
    p = store.get_or_create(db, "alice")
    assert len(p["positions"]) == 1
    pos = p["positions"][0]
    assert pos["shares"] == 20
    assert pos["avg_cost"] == pytest.approx(150.0)  # weighted average


def test_remove_position(db):
    store.add_position(db, "alice", ticker="AAPL", shares=10, avg_cost=150.0)
    store.add_position(db, "alice", ticker="MSFT", shares=5, avg_cost=300.0)
    store.remove_position(db, "alice", "AAPL")
    p = store.get_or_create(db, "alice")
    assert [pos["ticker"] for pos in p["positions"]] == ["MSFT"]


def test_users_cannot_see_each_others_portfolios(db):
    store.add_position(db, "alice", ticker="AAPL", shares=10, avg_cost=150.0)
    store.add_position(db, "bob",   ticker="TSLA", shares=1,  avg_cost=200.0)
    alice_p = store.get_or_create(db, "alice")
    bob_p   = store.get_or_create(db, "bob")
    assert [pos["ticker"] for pos in alice_p["positions"]] == ["AAPL"]
    assert [pos["ticker"] for pos in bob_p["positions"]]   == ["TSLA"]


def test_rejects_zero_or_negative_shares(db):
    with pytest.raises(ValueError):
        store.add_position(db, "alice", ticker="AAPL", shares=0, avg_cost=150.0)
    with pytest.raises(ValueError):
        store.add_position(db, "alice", ticker="AAPL", shares=-5, avg_cost=150.0)


def test_rejects_blank_ticker(db):
    with pytest.raises(ValueError):
        store.add_position(db, "alice", ticker="", shares=10, avg_cost=150.0)
