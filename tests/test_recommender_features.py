"""Tests for per-stock scalar feature computation on synthetic price series."""

import numpy as np
import pandas as pd
import pytest

from finance_dashboard.recommender.features import (
    FeatureProvider,
    macd_histogram,
    sma_ratio,
)
from finance_dashboard.recommender.models import FeatureName


def make_prices(n: int = 120, **series: np.ndarray) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame({k: v for k, v in series.items()}, index=idx)


def flat(n: int = 120, level: float = 100.0) -> np.ndarray:
    return np.full(n, level)


def trending(n: int = 120, start: float = 100.0, daily: float = 0.01) -> np.ndarray:
    return start * (1 + daily) ** np.arange(n)


def noisy(n: int = 120, scale: float = 0.05, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0, scale, n)))


def test_sma_ratio_positive_in_uptrend_zero_when_flat():
    up = pd.Series(trending())
    fl = pd.Series(flat())
    assert sma_ratio(up, window=20) > 0
    assert sma_ratio(fl, window=20) == pytest.approx(0.0)


def test_sma_ratio_nan_when_history_shorter_than_window():
    short = pd.Series(flat(5))
    assert np.isnan(sma_ratio(short, window=20))


def test_macd_histogram_flat_series_is_zero():
    assert macd_histogram(pd.Series(flat())) == pytest.approx(0.0, abs=1e-12)


def test_macd_histogram_uptrend_positive():
    assert macd_histogram(pd.Series(trending())) > 0


def test_feature_matrix_always_includes_volatility_and_returns():
    prices = make_prices(low=noisy(scale=0.005, seed=1), high=noisy(scale=0.05, seed=2))
    provider = FeatureProvider()
    matrix, skipped = provider.compute(prices, features=(FeatureName.SMA,))
    assert {"volatility", "returns", "sma_ratio"} <= set(matrix.columns)
    assert skipped == {}


def test_volatility_and_returns_values():
    # Constant 1% daily growth: zero volatility, known total return.
    n = 120
    prices = make_prices(n=n, steady=trending(n=n, daily=0.01))
    provider = FeatureProvider()
    matrix, _ = provider.compute(prices, features=())
    assert matrix.loc["steady", "volatility"] == pytest.approx(0.0, abs=1e-9)
    expected_return = 1.01 ** (n - 1) - 1
    assert matrix.loc["steady", "returns"] == pytest.approx(expected_return, rel=1e-9)


def test_higher_noise_means_higher_volatility():
    prices = make_prices(calm=noisy(scale=0.005, seed=3), wild=noisy(scale=0.05, seed=4))
    matrix, _ = FeatureProvider().compute(prices, features=())
    assert matrix.loc["wild", "volatility"] > matrix.loc["calm", "volatility"]


def test_ticker_with_insufficient_data_is_skipped_not_fatal():
    n = 120
    good = noisy(seed=5)
    bad = np.full(n, np.nan)
    bad[:2] = 100.0
    prices = make_prices(GOOD=good, BAD=bad)
    matrix, skipped = FeatureProvider().compute(prices, features=(FeatureName.SMA,))
    assert list(matrix.index) == ["GOOD"]
    assert "BAD" in skipped


def test_fundamentals_uses_injected_lookup():
    prices = make_prices(A=noisy(seed=6))
    provider = FeatureProvider(fundamentals_lookup=lambda t: {"beta": 1.5})
    matrix, _ = provider.compute(prices, features=(FeatureName.FUNDAMENTALS,))
    assert matrix.loc["A", "beta"] == pytest.approx(1.5)


def test_sentiment_stub_is_neutral():
    prices = make_prices(A=noisy(seed=8))
    matrix, _ = FeatureProvider().compute(prices, features=(FeatureName.SENTIMENT,))
    assert matrix.loc["A", "sentiment"] == pytest.approx(0.0)
