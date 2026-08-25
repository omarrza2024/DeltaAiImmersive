"""User profiles: investment preferences + UI appearance."""

from __future__ import annotations

from datetime import datetime, timezone


def _default_profile(username: str) -> dict:
    return {
        "username": username,
        "gain_color": "#00E676",
        "loss_color": "#FF3B30",
        "preferences": {
            "target_annual_return_pct": None,
            "max_volatility_pct": None,
            "sectors": [],
            "max_recommendations": None,
            "risk_preference": None,
        },
        "updated_at": datetime.now(timezone.utc),
    }


def get_or_create(db, username: str) -> dict:
    """Return the user's profile, creating a default one if it doesn't exist."""
    doc = db["profiles"].find_one({"username": username})
    if doc is not None:
        return doc
    doc = _default_profile(username)
    db["profiles"].insert_one(doc)
    return doc


def update(db, username: str, updates: dict) -> None:
    """Merge dotted-path updates into the user's profile. No cross-user leakage."""
    updates = {**updates, "updated_at": datetime.now(timezone.utc)}
    db["profiles"].update_one(
        {"username": username},
        {"$set": updates},
        upsert=True,
    )
