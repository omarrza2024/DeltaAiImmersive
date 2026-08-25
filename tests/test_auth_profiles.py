"""Tests for user profiles: get, update, default-on-missing."""

import pytest
from mongomock import MongoClient

from finance_dashboard.auth import profiles


@pytest.fixture
def db():
    return MongoClient()["tradematrix_test"]


def test_missing_profile_returns_defaults(db):
    p = profiles.get_or_create(db, "alice")
    assert p["username"] == "alice"
    assert p["gain_color"] == "#00E676"
    assert p["loss_color"] == "#FF3B30"
    assert p["preferences"] == {
        "target_annual_return_pct": None,
        "max_volatility_pct": None,
        "sectors": [],
        "max_recommendations": None,
        "risk_preference": None,
    }


def test_get_or_create_persists_default(db):
    profiles.get_or_create(db, "alice")
    # Second call should return the same doc (not overwrite).
    p2 = profiles.get_or_create(db, "alice")
    assert p2["_id"] == db["profiles"].find_one({"username": "alice"})["_id"]


def test_update_merges_fields(db):
    profiles.get_or_create(db, "alice")
    profiles.update(db, "alice", {"gain_color": "#22C55E"})
    p = profiles.get_or_create(db, "alice")
    assert p["gain_color"] == "#22C55E"
    # Untouched fields survive.
    assert p["loss_color"] == "#FF3B30"


def test_update_preferences_scoped_to_user(db):
    profiles.get_or_create(db, "alice")
    profiles.get_or_create(db, "bob")
    profiles.update(db, "alice", {"preferences.target_annual_return_pct": 10.0})
    assert profiles.get_or_create(db, "alice")["preferences"]["target_annual_return_pct"] == 10.0
    assert profiles.get_or_create(db, "bob")["preferences"]["target_annual_return_pct"] is None
