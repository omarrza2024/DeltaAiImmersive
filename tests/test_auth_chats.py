"""Tests for chat persistence + strict per-user scoping."""

import pytest
from mongomock import MongoClient

from finance_dashboard.auth import chats


@pytest.fixture
def db():
    return MongoClient()["tradematrix_test"]


def test_new_chat_returns_id_and_records_owner(db):
    chat_id = chats.create(db, "alice", title="10% tech")
    doc = db["chats"].find_one({"_id": chat_id})
    assert doc["username"] == "alice"
    assert doc["title"] == "10% tech"
    assert doc["messages"] == []


def test_append_message_stores_role_and_content(db):
    chat_id = chats.create(db, "alice", title="chat 1")
    chats.append_message(db, "alice", chat_id, role="user", content="hello")
    chats.append_message(db, "alice", chat_id, role="assistant", content="hi back")
    doc = db["chats"].find_one({"_id": chat_id})
    assert [m["role"] for m in doc["messages"]] == ["user", "assistant"]
    assert doc["messages"][0]["content"] == "hello"


def test_user_cannot_write_to_other_users_chat(db):
    chat_id = chats.create(db, "alice", title="alice's chat")
    with pytest.raises(PermissionError):
        chats.append_message(db, "bob", chat_id, role="user", content="steal")


def test_list_only_returns_own_chats(db):
    a1 = chats.create(db, "alice", title="alice 1")
    a2 = chats.create(db, "alice", title="alice 2")
    chats.create(db, "bob", title="bob's chat")
    alice_ids = [c["_id"] for c in chats.list_for_user(db, "alice")]
    assert set(alice_ids) == {a1, a2}


def test_load_rejects_wrong_owner(db):
    chat_id = chats.create(db, "alice", title="private")
    with pytest.raises(PermissionError):
        chats.load(db, "bob", chat_id)


def test_load_returns_full_chat_for_owner(db):
    chat_id = chats.create(db, "alice", title="private")
    chats.append_message(db, "alice", chat_id, role="user", content="mine")
    doc = chats.load(db, "alice", chat_id)
    assert doc["title"] == "private"
    assert doc["messages"][0]["content"] == "mine"


def test_chats_listed_newest_first(db):
    import time
    a = chats.create(db, "alice", title="old")
    time.sleep(0.01)
    b = chats.create(db, "alice", title="new")
    listed = chats.list_for_user(db, "alice")
    assert [c["_id"] for c in listed] == [b, a]
