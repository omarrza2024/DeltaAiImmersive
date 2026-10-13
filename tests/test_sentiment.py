"""Tests for the marketaux-backed sentiment module.

Only the pure logic (compute_sentiment, classify, _extract_scores) is exercised
here. The network path is covered by a monkey-patched requests.get.
"""

from __future__ import annotations

import pytest

from finance_dashboard import sentiment
from finance_dashboard.sentiment import (
    NewsArticle,
    SentimentResult,
    SentimentUnavailableError,
    _extract_articles,
    _extract_scores,
    _fetch_payload,
    classify,
    compute_sentiment,
)


def test_classify_thresholds():
    assert classify(0.1) == "Positive"
    assert classify(-0.1) == "Negative"
    assert classify(0.0) == "Neutral"
    assert classify(0.04) == "Neutral"   # within neutral band
    assert classify(-0.04) == "Neutral"


def test_extract_scores_matches_ticker_case_insensitive():
    payload = {
        "data": [
            {"entities": [{"symbol": "aapl", "sentiment_score": 0.4}]},
            {"entities": [{"symbol": "AAPL", "sentiment_score": -0.2}]},
            {"entities": [{"symbol": "MSFT", "sentiment_score": 0.9}]},  # ignored
        ]
    }
    scores = _extract_scores(payload, "AAPL")
    assert scores == [0.4, -0.2]


def test_extract_scores_skips_null_and_malformed():
    payload = {
        "data": [
            {"entities": [{"symbol": "AAPL", "sentiment_score": None}]},
            {"entities": [{"symbol": "AAPL", "sentiment_score": "bogus"}]},
            {"entities": [{"symbol": "AAPL", "sentiment_score": 0.3}]},
        ]
    }
    assert _extract_scores(payload, "AAPL") == [0.3]


def test_compute_sentiment_averages_and_labels_positive():
    payload = {
        "data": [
            {"entities": [{"symbol": "AAPL", "sentiment_score": 0.5}]},
            {"entities": [{"symbol": "AAPL", "sentiment_score": 0.3}]},
            {"entities": [{"symbol": "AAPL", "sentiment_score": 0.1}]},
        ]
    }
    result = compute_sentiment("AAPL", payload)
    assert result.ticker == "AAPL"
    assert result.article_count == 3
    assert result.avg_score == pytest.approx(0.3, abs=1e-4)
    assert result.label == "Positive"


def test_compute_sentiment_averages_and_labels_negative():
    payload = {
        "data": [
            {"entities": [{"symbol": "TSLA", "sentiment_score": -0.4}]},
            {"entities": [{"symbol": "TSLA", "sentiment_score": -0.2}]},
        ]
    }
    result = compute_sentiment("TSLA", payload)
    assert result.label == "Negative"
    assert result.avg_score < 0
    assert result.article_count == 2


def test_compute_sentiment_empty_payload_returns_neutral_zero():
    result = compute_sentiment("AAPL", {"data": []})
    assert result == SentimentResult(
        ticker="AAPL", avg_score=0.0, article_count=0, label="Neutral"
    )


def test_compute_sentiment_handles_missing_data_key():
    result = compute_sentiment("AAPL", {})
    assert result.article_count == 0
    assert result.label == "Neutral"


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.text = text

    def json(self):
        return self._json


def test_fetch_payload_raises_on_non_200(monkeypatch):
    monkeypatch.setattr(
        sentiment.requests, "get",
        lambda *a, **kw: _FakeResponse(status_code=429, text="rate limited"),
    )
    with pytest.raises(SentimentUnavailableError, match="HTTP 429"):
        _fetch_payload("AAPL", api_key="k")


def test_fetch_payload_raises_on_network_error(monkeypatch):
    def boom(*a, **kw):
        raise sentiment.requests.ConnectionError("DNS")
    monkeypatch.setattr(sentiment.requests, "get", boom)
    with pytest.raises(SentimentUnavailableError, match="Network error"):
        _fetch_payload("AAPL", api_key="k")


def test_fetch_payload_returns_parsed_json(monkeypatch):
    monkeypatch.setattr(
        sentiment.requests, "get",
        lambda *a, **kw: _FakeResponse(200, {"data": [{"foo": 1}]}),
    )
    payload = _fetch_payload("AAPL", api_key="k")
    assert payload == {"data": [{"foo": 1}]}


def test_extract_articles_captures_metadata():
    payload = {
        "data": [
            {
                "title": "Big news!",
                "url": "https://example.com/a",
                "source": "Example News",
                "published_at": "2026-10-10T12:00:00.000000Z",
                "description": "A great summary.",
                "entities": [{"symbol": "AAPL", "sentiment_score": 0.42}],
            },
            {
                "title": "Also happening",
                "url": "https://example.com/b",
                "source": "News Co",
                "published_at": "2026-10-09T09:00:00.000000Z",
                "entities": [
                    {"symbol": "MSFT", "sentiment_score": 0.5},   # not AAPL
                    {"symbol": "AAPL", "sentiment_score": -0.3},
                ],
            },
        ]
    }
    articles = _extract_articles(payload, "AAPL")
    assert len(articles) == 2
    assert articles[0] == NewsArticle(
        title="Big news!", url="https://example.com/a", source="Example News",
        published_at="2026-10-10T12:00:00.000000Z", score=0.42,
        description="A great summary.",
    )
    # Second article: the first matching AAPL entity's score is used.
    assert articles[1].score == pytest.approx(-0.3)
    assert articles[1].title == "Also happening"


def test_compute_sentiment_populates_articles_tuple():
    payload = {
        "data": [
            {
                "title": "Headline",
                "url": "https://x.test/1",
                "entities": [{"symbol": "NVDA", "sentiment_score": 0.6}],
            },
        ]
    }
    result = compute_sentiment("NVDA", payload)
    assert result.article_count == 1
    assert len(result.articles) == 1
    assert result.articles[0].title == "Headline"
    assert result.articles[0].url == "https://x.test/1"
