"""MongoDB persistence for user portfolios. Strictly per-user scoping."""

from __future__ import annotations

from datetime import datetime, timezone


def _normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper()


def get_or_create(db, username: str) -> dict:
    """Return the user's portfolio; create an empty one if none exists."""
    doc = db["portfolios"].find_one({"username": username})
    if doc is not None:
        return doc
    doc = {
        "username": username,
        "positions": [],
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    db["portfolios"].insert_one(doc)
    return doc


def add_position(db, username: str, *, ticker: str, shares: float, avg_cost: float) -> None:
    """Add or increase a position. Same ticker → weighted-average the cost basis."""
    t = _normalize_ticker(ticker)
    if not t:
        raise ValueError("Ticker is required.")
    if shares <= 0:
        raise ValueError("Shares must be positive.")
    if avg_cost < 0:
        raise ValueError("Average cost cannot be negative.")

    portfolio = get_or_create(db, username)
    positions = portfolio.get("positions", [])
    existing = next((p for p in positions if p["ticker"] == t), None)

    if existing is not None:
        # Weighted-average: (old_shares*old_cost + new_shares*new_cost) / total
        old_shares = existing["shares"]
        old_cost = existing["avg_cost"]
        total_shares = old_shares + shares
        new_avg = (old_shares * old_cost + shares * avg_cost) / total_shares
        existing["shares"] = total_shares
        existing["avg_cost"] = new_avg
    else:
        positions.append({
            "ticker": t,
            "shares": shares,
            "avg_cost": avg_cost,
            "added_at": datetime.now(timezone.utc),
        })

    db["portfolios"].update_one(
        {"username": username},
        {"$set": {"positions": positions, "updated_at": datetime.now(timezone.utc)}},
    )


def remove_position(db, username: str, ticker: str) -> None:
    t = _normalize_ticker(ticker)
    db["portfolios"].update_one(
        {"username": username},
        {"$pull": {"positions": {"ticker": t}},
         "$set": {"updated_at": datetime.now(timezone.utc)}},
    )
