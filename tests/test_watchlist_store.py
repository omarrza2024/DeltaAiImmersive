"""Tests for watchlist CRUD, scoping, and duplicate prevention."""

import pytest
from mongomock import MongoClient

from finance_dashboard.watchlist import store


@pytest.fixture
def db():
    return MongoClient()["tradematrix_test"]


# --- create ---

def test_create_watchlist(db):
    wid = store.create(db, "alice", "Tech stocks")
    doc = db["watchlists"].find_one({"_id": wid})
    assert doc["username"] == "alice"
    assert doc["name"] == "Tech stocks"
    assert doc["tickers"] == []


def test_empty_name_rejected(db):
    with pytest.raises(ValueError):
        store.create(db, "alice", "")
    with pytest.raises(ValueError):
        store.create(db, "alice", "   ")


def test_duplicate_name_per_user_rejected(db):
    store.create(db, "alice", "Tech stocks")
    with pytest.raises(ValueError, match="already"):
        store.create(db, "alice", "Tech stocks")
    with pytest.raises(ValueError):
        store.create(db, "alice", "  Tech stocks  ")  # trimmed matches


def test_different_users_can_have_same_watchlist_name(db):
    store.create(db, "alice", "Tech stocks")
    bid = store.create(db, "bob", "Tech stocks")
    assert db["watchlists"].find_one({"_id": bid, "username": "bob"}) is not None


# --- list ---

def test_list_only_returns_own_watchlists(db):
    a1 = store.create(db, "alice", "Tech")
    a2 = store.create(db, "alice", "Medical")
    store.create(db, "bob", "Finance")
    alice_ids = [w["_id"] for w in store.list_for_user(db, "alice")]
    assert set(alice_ids) == {a1, a2}


# --- add ticker ---

def test_add_ticker(db):
    wid = store.create(db, "alice", "Tech")
    store.add_ticker(db, "alice", wid, "aapl",
                     exists_fn=lambda t: True)
    doc = db["watchlists"].find_one({"_id": wid})
    assert doc["tickers"] == ["AAPL"]  # normalized to uppercase


def test_duplicate_ticker_rejected(db):
    wid = store.create(db, "alice", "Tech")
    store.add_ticker(db, "alice", wid, "AAPL", exists_fn=lambda t: True)
    with pytest.raises(ValueError, match="already"):
        store.add_ticker(db, "alice", wid, "aapl", exists_fn=lambda t: True)


def test_nonexistent_ticker_rejected(db):
    wid = store.create(db, "alice", "Tech")
    with pytest.raises(ValueError, match="not found"):
        store.add_ticker(db, "alice", wid, "ZZZZ", exists_fn=lambda t: False)


def test_cannot_add_to_other_users_watchlist(db):
    wid = store.create(db, "alice", "Tech")
    with pytest.raises(PermissionError):
        store.add_ticker(db, "bob", wid, "AAPL", exists_fn=lambda t: True)


# --- remove ticker ---

def test_remove_ticker(db):
    wid = store.create(db, "alice", "Tech")
    store.add_ticker(db, "alice", wid, "AAPL", exists_fn=lambda t: True)
    store.add_ticker(db, "alice", wid, "MSFT", exists_fn=lambda t: True)
    store.remove_ticker(db, "alice", wid, "AAPL")
    doc = db["watchlists"].find_one({"_id": wid})
    assert doc["tickers"] == ["MSFT"]


def test_cannot_remove_from_other_users_watchlist(db):
    wid = store.create(db, "alice", "Tech")
    store.add_ticker(db, "alice", wid, "AAPL", exists_fn=lambda t: True)
    with pytest.raises(PermissionError):
        store.remove_ticker(db, "bob", wid, "AAPL")


# --- rename ---

def test_rename(db):
    wid = store.create(db, "alice", "Tech")
    store.rename(db, "alice", wid, "Tech growth")
    doc = db["watchlists"].find_one({"_id": wid})
    assert doc["name"] == "Tech growth"


def test_rename_rejects_duplicate(db):
    store.create(db, "alice", "Tech")
    med = store.create(db, "alice", "Medical")
    with pytest.raises(ValueError, match="already"):
        store.rename(db, "alice", med, "Tech")


def test_rename_rejects_empty(db):
    wid = store.create(db, "alice", "Tech")
    with pytest.raises(ValueError):
        store.rename(db, "alice", wid, "")


def test_cannot_rename_other_users_watchlist(db):
    wid = store.create(db, "alice", "Tech")
    with pytest.raises(PermissionError):
        store.rename(db, "bob", wid, "pwned")


# --- delete ---

def test_delete(db):
    wid = store.create(db, "alice", "Tech")
    store.delete(db, "alice", wid)
    assert db["watchlists"].find_one({"_id": wid}) is None


def test_cannot_delete_other_users_watchlist(db):
    wid = store.create(db, "alice", "Tech")
    with pytest.raises(PermissionError):
        store.delete(db, "bob", wid)
    # Still there
    assert db["watchlists"].find_one({"_id": wid}) is not None
