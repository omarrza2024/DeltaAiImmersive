"""Signup / login / bcrypt password hashing."""

from __future__ import annotations

from datetime import datetime, timezone

import bcrypt

from finance_dashboard.models import UserInputError


def hash_password(password: str) -> bytes:
    """Bcrypt-hash a password. Per-password salt is embedded in the returned hash."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())


def verify_password(password: str, stored_hash) -> bool:
    """Return True if `password` matches `stored_hash`."""
    if stored_hash is None:
        return False
    if isinstance(stored_hash, str):
        stored_hash = stored_hash.encode("utf-8")
    try:
        return bcrypt.checkpw(password.encode("utf-8"), stored_hash)
    except Exception:
        return False


def signup(db, username: str, password: str) -> str:
    """Create a new user. Raises UserInputError if invalid or duplicate."""
    username = (username or "").strip()
    if not username:
        raise UserInputError("Username is required.")
    if not password:
        raise UserInputError("Password is required.")
    if len(password) < 6:
        raise UserInputError("Password must be at least 6 characters.")

    if db["users"].find_one({"username": username}) is not None:
        raise UserInputError(f"Username '{username}' is already taken.")

    db["users"].insert_one(
        {
            "username": username,
            "password_hash": hash_password(password),
            "created_at": datetime.now(timezone.utc),
        }
    )
    return username


def login(db, username: str, password: str) -> str:
    """Return `username` on success. Raises UserInputError on failure."""
    username = (username or "").strip()
    doc = db["users"].find_one({"username": username})
    if not doc or not verify_password(password, doc["password_hash"]):
        raise UserInputError("Invalid username or password.")
    return username
