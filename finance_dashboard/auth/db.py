"""MongoDB connection factory. Reads MONGO_URI from Streamlit secrets or env.

Guests (not signed in) transparently share an in-memory mongomock DB held
in session_state — so Watchlist and chat history features work without an
account. The guest DB survives Streamlit reruns within one browser tab but
not hard refreshes, which is the trial nature of guest mode.
"""

from __future__ import annotations

import os

DB_NAME = "tradematrix"
_GUEST_DB_KEY = "_guest_mongomock_db"


def get_mongo_uri() -> str | None:
    try:
        import streamlit as st
        return st.secrets.get("MONGO_URI") or os.getenv("MONGO_URI")
    except Exception:
        return os.getenv("MONGO_URI")


def _mongo_client_kwargs() -> dict:
    """MongoClient kwargs — includes certifi's CA bundle so Atlas TLS verifies on macOS."""
    kwargs = {"serverSelectionTimeoutMS": 5000}
    try:
        import certifi
        kwargs["tlsCAFile"] = certifi.where()
    except ImportError:
        pass  # SRV URIs on Linux typically have system CAs available.
    return kwargs


def _get_guest_db():
    """In-memory mongomock DB scoped to the current browser session."""
    import streamlit as st
    if _GUEST_DB_KEY not in st.session_state:
        import mongomock
        st.session_state[_GUEST_DB_KEY] = mongomock.MongoClient()[DB_NAME]
    return st.session_state[_GUEST_DB_KEY]


def get_db():
    """Return the Mongo Database handle for the current session.

    - Signed-in user → real MongoDB (or None when unconfigured).
    - Guest         → session-scoped mongomock DB (so Watchlist works).

    The auth.ui import is deferred to avoid a circular import.
    """
    try:
        from finance_dashboard.auth.ui import is_guest
        if is_guest():
            return _get_guest_db()
    except Exception:
        pass  # auth.ui not importable yet — fall through to real Mongo

    uri = get_mongo_uri()
    if not uri:
        return None
    try:
        import streamlit as st

        @st.cache_resource(show_spinner=False)
        def _client(u: str):
            from pymongo import MongoClient
            return MongoClient(u, **_mongo_client_kwargs())

        return _client(uri)[DB_NAME]
    except Exception:
        from pymongo import MongoClient
        return MongoClient(uri, **_mongo_client_kwargs())[DB_NAME]
