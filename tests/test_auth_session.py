"""Tests for JWT session token encoding / decoding."""

import time

import pytest

from finance_dashboard.auth import session


def test_roundtrip_encode_decode():
    token = session.encode_token("alice", secret="test-secret", max_age_days=7)
    assert token is not None
    assert session.decode_token(token, secret="test-secret") == "alice"


def test_wrong_secret_rejected():
    token = session.encode_token("alice", secret="test-secret", max_age_days=7)
    assert session.decode_token(token, secret="different-secret") is None


def test_tampered_token_rejected():
    token = session.encode_token("alice", secret="test-secret", max_age_days=7)
    tampered = token[:-1] + ("A" if token[-1] != "A" else "B")
    assert session.decode_token(tampered, secret="test-secret") is None


def test_expired_token_rejected():
    # 1s expiry — sleep a bit then check.
    token = session.encode_token("alice", secret="test-secret", max_age_days=0,
                                 override_exp_seconds=1)
    assert session.decode_token(token, secret="test-secret") == "alice"
    time.sleep(1.5)
    assert session.decode_token(token, secret="test-secret") is None


def test_encode_returns_none_when_secret_missing(monkeypatch):
    """Isolate every source of secret material so we exercise the 'no secret' path."""
    monkeypatch.delenv("SESSION_SECRET", raising=False)
    monkeypatch.delenv("MONGO_URI", raising=False)
    # Also block Streamlit-secrets fallback in the pytest context.
    monkeypatch.setattr(session, "_resolve_secret", lambda explicit=None: explicit)
    assert session.encode_token("alice", secret=None, max_age_days=7) is None


def test_decode_handles_garbage_input():
    assert session.decode_token("not-a-token", secret="test-secret") is None
    assert session.decode_token("", secret="test-secret") is None
    assert session.decode_token(None, secret="test-secret") is None
