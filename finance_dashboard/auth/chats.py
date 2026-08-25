"""Chat persistence with strict per-user scoping.

Every read/write takes a `username` and refuses to touch chats owned by anyone
else — the rule "user A cannot see user B's chats" is enforced at this layer,
not left to callers.
"""

from __future__ import annotations

from datetime import datetime, timezone


def create(db, username: str, title: str = "New chat") -> object:
    """Create a new empty chat for `username`. Returns the chat's `_id`."""
    now = datetime.now(timezone.utc)
    result = db["chats"].insert_one(
        {
            "username": username,
            "title": title,
            "messages": [],
            "created_at": now,
            "updated_at": now,
        }
    )
    return result.inserted_id


def _owner_or_raise(db, username: str, chat_id) -> dict:
    doc = db["chats"].find_one({"_id": chat_id})
    if doc is None:
        raise LookupError(f"Chat {chat_id} not found")
    if doc.get("username") != username:
        raise PermissionError(f"Chat {chat_id} is not owned by {username}")
    return doc


def append_message(db, username: str, chat_id, role: str, content: str) -> None:
    """Append one message to a chat. Refuses if the chat isn't the user's."""
    _owner_or_raise(db, username, chat_id)
    db["chats"].update_one(
        {"_id": chat_id, "username": username},
        {
            "$push": {"messages": {"role": role, "content": content,
                                   "timestamp": datetime.now(timezone.utc)}},
            "$set":  {"updated_at": datetime.now(timezone.utc)},
        },
    )


def load(db, username: str, chat_id) -> dict:
    """Return the full chat document if `username` owns it."""
    return _owner_or_raise(db, username, chat_id)


def list_for_user(db, username: str) -> list[dict]:
    """List all chats owned by `username`, newest first (title + timestamps only)."""
    cursor = (
        db["chats"]
        .find(
            {"username": username},
            {"title": 1, "created_at": 1, "updated_at": 1},
        )
        .sort("updated_at", -1)
    )
    return list(cursor)


def set_title(db, username: str, chat_id, title: str) -> None:
    _owner_or_raise(db, username, chat_id)
    db["chats"].update_one(
        {"_id": chat_id, "username": username},
        {"$set": {"title": title, "updated_at": datetime.now(timezone.utc)}},
    )


def delete(db, username: str, chat_id) -> None:
    _owner_or_raise(db, username, chat_id)
    db["chats"].delete_one({"_id": chat_id, "username": username})
