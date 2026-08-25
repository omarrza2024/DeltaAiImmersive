"""Tests for signup / login / password hashing."""

import pytest
from mongomock import MongoClient

from finance_dashboard.auth import users
from finance_dashboard.models import UserInputError


@pytest.fixture
def db():
    client = MongoClient()
    return client["tradematrix_test"]


def test_password_hash_roundtrip():
    hashed = users.hash_password("hunter2")
    assert hashed != "hunter2"                          # never stored plaintext
    assert hashed.startswith(b"$2b$") or hashed.startswith(b"$2a$")  # bcrypt marker
    assert users.verify_password("hunter2", hashed)
    assert not users.verify_password("wrong", hashed)


def test_signup_creates_user_with_hashed_password(db):
    users.signup(db, "alice", "hunter2")
    doc = db["users"].find_one({"username": "alice"})
    assert doc is not None
    assert doc["password_hash"] != "hunter2"
    assert users.verify_password("hunter2", doc["password_hash"])


def test_signup_rejects_duplicate_username(db):
    users.signup(db, "alice", "hunter2")
    with pytest.raises(UserInputError, match="taken"):
        users.signup(db, "alice", "different_password")


def test_signup_rejects_blank_credentials(db):
    with pytest.raises(UserInputError):
        users.signup(db, "", "hunter2")
    with pytest.raises(UserInputError):
        users.signup(db, "alice", "")


def test_login_returns_username_on_success(db):
    users.signup(db, "alice", "hunter2")
    assert users.login(db, "alice", "hunter2") == "alice"


def test_login_wrong_password_rejected(db):
    users.signup(db, "alice", "hunter2")
    with pytest.raises(UserInputError, match="Invalid"):
        users.login(db, "alice", "wrong_password")


def test_login_unknown_user_rejected(db):
    with pytest.raises(UserInputError, match="Invalid"):
        users.login(db, "ghost", "any")
