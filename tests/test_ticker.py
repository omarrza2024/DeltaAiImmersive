"""Tests for ticker parsing and validation."""

import pytest

from finance_dashboard.models import parse_ticker_list, normalize_ticker, dedupe_tickers


def test_normalize_ticker_strips_and_uppercases():
    assert normalize_ticker("  aapl  ") == "AAPL"
    assert normalize_ticker("ms ft") == "MSFT"


def test_parse_ticker_list_splits_on_commas_and_newlines():
    assert parse_ticker_list("AAPL, MSFT\nGOOGL") == ["AAPL", "MSFT", "GOOGL"]
    assert parse_ticker_list("aapl;msft") == ["AAPL", "MSFT"]


def test_parse_ticker_list_empty_returns_empty():
    assert parse_ticker_list("") == []
    assert parse_ticker_list("   ,  ; ") == []


def test_dedupe_tickers_preserves_order():
    assert dedupe_tickers(["AAPL", "MSFT", "AAPL", "GOOGL"]) == ["AAPL", "MSFT", "GOOGL"]
