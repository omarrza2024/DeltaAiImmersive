"""Marketaux-backed news sentiment for a ticker.

Pulls the latest news articles for a symbol from marketaux, averages the
entity-level sentiment scores published alongside each article, and reports
the result plus a human label (Positive / Negative / Neutral).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import requests
import streamlit as st

MARKETAUX_URL = "https://api.marketaux.com/v1/news/all"
NEWS_LIMIT = 10
HTTP_TIMEOUT = 10.0

# Thresholds for the human-readable label. Marketaux scores are in [-1, 1];
# a narrow neutral band avoids calling 0.02 "positive".
_POSITIVE_CUTOFF = 0.05
_NEGATIVE_CUTOFF = -0.05


class SentimentUnavailableError(RuntimeError):
    """Raised when the sentiment service cannot deliver a result."""


@dataclass(frozen=True)
class NewsArticle:
    """One article contributing to the sentiment average."""
    title: str
    url: str
    source: str
    published_at: str
    score: float
    description: str = ""


@dataclass(frozen=True)
class SentimentResult:
    ticker: str
    avg_score: float          # mean of entity sentiment_score across articles, [-1, 1]
    article_count: int        # how many articles contributed a score
    label: str                # "Positive" | "Negative" | "Neutral"
    articles: tuple[NewsArticle, ...] = field(default_factory=tuple)


def classify(score: float) -> str:
    if score > _POSITIVE_CUTOFF:
        return "Positive"
    if score < _NEGATIVE_CUTOFF:
        return "Negative"
    return "Neutral"


def _get_api_key() -> str:
    try:
        key = st.secrets["MARKETAUX_API_KEY"]
    except Exception as exc:
        raise SentimentUnavailableError(
            "MARKETAUX_API_KEY missing from .streamlit/secrets.toml"
        ) from exc
    if not key:
        raise SentimentUnavailableError(
            "MARKETAUX_API_KEY is empty — add your key to .streamlit/secrets.toml"
        )
    return key


def _extract_scores(payload: dict[str, Any], ticker: str) -> list[float]:
    """Pull every sentiment_score from entities that match the ticker."""
    return [article.score for article in _extract_articles(payload, ticker)]


def _extract_articles(payload: dict[str, Any], ticker: str) -> list[NewsArticle]:
    """Pull per-article metadata + sentiment for entities matching `ticker`.

    An article may list several entities; we take the FIRST entity matching
    the ticker (its sentiment_score represents the article's slant on that
    specific symbol).
    """
    out: list[NewsArticle] = []
    ticker_upper = ticker.upper()
    for article in payload.get("data", []) or []:
        for entity in article.get("entities", []) or []:
            if (entity.get("symbol", "") or "").upper() != ticker_upper:
                continue
            raw = entity.get("sentiment_score")
            if raw is None:
                break
            try:
                score = float(raw)
            except (TypeError, ValueError):
                break
            out.append(
                NewsArticle(
                    title=str(article.get("title") or "").strip() or "(no title)",
                    url=str(article.get("url") or ""),
                    source=str(article.get("source") or ""),
                    published_at=str(article.get("published_at") or ""),
                    score=score,
                    description=str(article.get("description") or "").strip(),
                )
            )
            break  # one entity-match per article
    return out


def _fetch_payload(ticker: str, api_key: str) -> dict[str, Any]:
    """One HTTP call to marketaux — raises SentimentUnavailableError on any failure."""
    try:
        resp = requests.get(
            MARKETAUX_URL,
            params={
                "symbols": ticker,
                "limit": NEWS_LIMIT,
                "filter_entities": "true",
                "language": "en",
                "api_token": api_key,
            },
            timeout=HTTP_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise SentimentUnavailableError(f"Network error contacting marketaux: {exc}") from exc

    if resp.status_code != 200:
        raise SentimentUnavailableError(
            f"Marketaux returned HTTP {resp.status_code}: {resp.text[:160]}"
        )
    try:
        return resp.json()
    except ValueError as exc:
        raise SentimentUnavailableError("Marketaux returned non-JSON response") from exc


def compute_sentiment(ticker: str, payload: dict[str, Any]) -> SentimentResult:
    """Pure function: average the entity sentiment scores from a marketaux payload."""
    articles = _extract_articles(payload, ticker)
    if not articles:
        return SentimentResult(
            ticker=ticker, avg_score=0.0, article_count=0, label="Neutral", articles=()
        )
    avg = sum(a.score for a in articles) / len(articles)
    return SentimentResult(
        ticker=ticker,
        avg_score=round(avg, 4),
        article_count=len(articles),
        label=classify(avg),
        articles=tuple(articles),
    )


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_sentiment(ticker: str) -> SentimentResult:
    """Fetch latest news for `ticker` and compute average entity sentiment."""
    api_key = _get_api_key()
    payload = _fetch_payload(ticker, api_key)
    return compute_sentiment(ticker, payload)


def make_sentiment_lookup() -> Callable[[str], float]:
    """A lookup for FeatureProvider — returns 0.0 on any failure so clustering
    still runs even if marketaux is unreachable or the key is missing."""
    def lookup(ticker: str) -> float:
        try:
            return fetch_sentiment(ticker).avg_score
        except SentimentUnavailableError:
            return 0.0
    return lookup
