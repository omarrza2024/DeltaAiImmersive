"""MongoDB persistence for watchlists. All ops strictly scoped per user."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable


def _norm_name(name: str) -> str:
    return (name or "").strip()


def _norm_ticker(t: str) -> str:
    return (t or "").strip().upper()


def _owner_or_raise(db, username: str, watchlist_id) -> dict:
    doc = db["watchlists"].find_one({"_id": watchlist_id})
    if doc is None:
        raise LookupError(f"Watchlist {watchlist_id} not found")
    if doc.get("username") != username:
        raise PermissionError(f"Watchlist {watchlist_id} is not owned by {username}")
    return doc


def _name_taken(db, username: str, name: str, except_id=None) -> bool:
    """Case-insensitive uniqueness check — matches rename+create behavior."""
    q = {"username": username, "name": {"$regex": f"^{name}$", "$options": "i"}}
    if except_id is not None:
        q["_id"] = {"$ne": except_id}
    return db["watchlists"].find_one(q) is not None


def create(db, username: str, name: str):
    """Create a new watchlist for `username`. Returns the inserted _id."""
    name = _norm_name(name)
    if not name:
        raise ValueError("Watchlist name cannot be empty.")
    if _name_taken(db, username, name):
        raise ValueError(f"You already have a watchlist named '{name}'.")
    now = datetime.now(timezone.utc)
    result = db["watchlists"].insert_one({
        "username": username,
        "name": name,
        "tickers": [],
        "created_at": now,
        "updated_at": now,
    })
    return result.inserted_id


def list_for_user(db, username: str) -> list[dict]:
    """Return all of `username`'s watchlists, newest-updated first."""
    return list(
        db["watchlists"]
        .find({"username": username})
        .sort("updated_at", -1)
    )


def get(db, username: str, watchlist_id) -> dict:
    """Return the full watchlist doc if owned by `username`."""
    return _owner_or_raise(db, username, watchlist_id)


def add_ticker(db, username: str, watchlist_id, ticker: str,
               *, exists_fn: Callable[[str], bool]) -> None:
    """Add a validated ticker. `exists_fn(ticker)` returns True → accept."""
    doc = _owner_or_raise(db, username, watchlist_id)
    t = _norm_ticker(ticker)
    if not t:
        raise ValueError("Ticker is required.")
    if t in doc.get("tickers", []):
        raise ValueError(f"{t} is already in this watchlist.")
    if not exists_fn(t):
        raise ValueError(f"Ticker '{t}' not found. Check the symbol and try again.")
    db["watchlists"].update_one(
        {"_id": watchlist_id, "username": username},
        {"$push": {"tickers": t}, "$set": {"updated_at": datetime.now(timezone.utc)}},
    )


def remove_ticker(db, username: str, watchlist_id, ticker: str) -> None:
    _owner_or_raise(db, username, watchlist_id)
    t = _norm_ticker(ticker)
    db["watchlists"].update_one(
        {"_id": watchlist_id, "username": username},
        {"$pull": {"tickers": t}, "$set": {"updated_at": datetime.now(timezone.utc)}},
    )


def rename(db, username: str, watchlist_id, new_name: str) -> None:
    _owner_or_raise(db, username, watchlist_id)
    new_name = _norm_name(new_name)
    if not new_name:
        raise ValueError("Watchlist name cannot be empty.")
    if _name_taken(db, username, new_name, except_id=watchlist_id):
        raise ValueError(f"You already have a watchlist named '{new_name}'.")
    db["watchlists"].update_one(
        {"_id": watchlist_id, "username": username},
        {"$set": {"name": new_name, "updated_at": datetime.now(timezone.utc)}},
    )


def delete(db, username: str, watchlist_id) -> None:
    _owner_or_raise(db, username, watchlist_id)
    db["watchlists"].delete_one({"_id": watchlist_id, "username": username})
