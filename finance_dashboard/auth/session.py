"""Signed session tokens — used in a browser cookie so login survives refresh."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta, timezone

_ALG = "HS256"


def _resolve_secret(explicit: str | None = None) -> str | None:
    """Resolve the signing secret.

    Priority:
      1. explicit argument (used in tests)
      2. SESSION_SECRET in Streamlit secrets or env var
      3. Derive one from MONGO_URI (already required, high-entropy) so users
         don't need to add another secret manually. Stable per-machine.
    """
    if explicit is not None:
        return explicit
    try:
        import streamlit as st
        v = st.secrets.get("SESSION_SECRET")
        if v:
            return v
    except Exception:
        pass
    v = os.getenv("SESSION_SECRET")
    if v:
        return v
    # Fallback: derive from MONGO_URI. Still keyed to something private, so
    # session tokens can't be forged without also having the Mongo URI.
    try:
        import streamlit as st
        mongo = st.secrets.get("MONGO_URI") or os.getenv("MONGO_URI")
    except Exception:
        mongo = os.getenv("MONGO_URI")
    if mongo:
        return hashlib.sha256(("tradematrix-session::" + mongo).encode()).hexdigest()
    return None


def encode_token(
    username: str,
    *,
    secret: str | None = None,
    max_age_days: int = 7,
    override_exp_seconds: int | None = None,
) -> str | None:
    """Return a JWT identifying `username`, or None if no secret is available."""
    resolved = _resolve_secret(secret)
    if not resolved or not username:
        return None
    import jwt

    now = datetime.now(timezone.utc)
    if override_exp_seconds is not None:
        exp = now + timedelta(seconds=override_exp_seconds)
    else:
        exp = now + timedelta(days=max_age_days)
    payload = {"sub": username, "iat": now, "exp": exp}
    token = jwt.encode(payload, resolved, algorithm=_ALG)
    return token if isinstance(token, str) else token.decode()


def decode_token(token: str | None, *, secret: str | None = None) -> str | None:
    """Return the username encoded in `token`, or None if invalid/expired/missing."""
    if not token:
        return None
    resolved = _resolve_secret(secret)
    if not resolved:
        return None
    import jwt
    try:
        payload = jwt.decode(token, resolved, algorithms=[_ALG])
        sub = payload.get("sub")
        return sub if isinstance(sub, str) and sub else None
    except Exception:
        return None
