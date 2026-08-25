"""MongoDB connection factory. Reads MONGO_URI from Streamlit secrets or env."""

from __future__ import annotations

import os

DB_NAME = "tradematrix"


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


def get_db():
    """Return the shared MongoDB Database handle, or None if unconfigured."""
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
