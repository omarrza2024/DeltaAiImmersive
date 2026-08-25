"""Groq LLM wrapper. Reads the key from Streamlit secrets or env var."""

from __future__ import annotations

import os
from typing import Callable

MODEL = "openai/gpt-oss-120b"

ChatCompletion = Callable[..., str]  # (system, messages, *, json_mode) -> content


def get_api_key() -> str | None:
    """Streamlit secrets first (production/UI), env var second (tests, CLI)."""
    try:
        import streamlit as st

        return st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    except Exception:
        return os.getenv("GROQ_API_KEY")


def groq_chat_completion(
    system: str,
    messages: list[dict],
    *,
    json_mode: bool = False,
) -> str:
    """Single-shot chat completion via Groq."""
    from groq import Groq

    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    client = Groq(api_key=api_key)
    kwargs: dict = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}, *messages],
        "temperature": 0.2,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""
