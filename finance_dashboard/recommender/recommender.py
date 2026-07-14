"""StockRecommender — orchestrates universe, data, features, grouping, and labeling."""

from __future__ import annotations

from typing import Callable

import pandas as pd

from finance_dashboard.models import UserInputError
from finance_dashboard.recommender.features import FeatureProvider
from finance_dashboard.recommender.models import (
    FeatureName,
    PortfolioProfile,
    RecommendationRequest,
    RecommendationResult,
    StockRecommendation,
    StrategyName,
)
from finance_dashboard.recommender.risk import RiskLabeler
from finance_dashboard.recommender.strategies import make_strategy
from finance_dashboard.recommender.universe import UniverseProvider

MIN_STOCKS_FOR_CLUSTERING = 3

# Feature-matrix columns produced for each user-selectable feature.
_FEATURE_COLUMNS: dict[FeatureName, list[str]] = {
    FeatureName.VOLATILITY: ["volatility"],
    FeatureName.RETURNS: ["returns"],
    FeatureName.SMA: ["sma_ratio"],
    FeatureName.MACD: ["macd_hist"],
    FeatureName.SENTIMENT: ["sentiment"],
    FeatureName.FUNDAMENTALS: ["beta", "trailing_pe"],
}

MarketDataFetcher = Callable[[tuple[str, ...], str], pd.DataFrame]


def _default_fetcher(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    from finance_dashboard.data import fetch_history

    return fetch_history(tickers, period)


class StockRecommender:
    """Turns a RecommendationRequest into risk-labeled stocks."""

    def __init__(
        self,
        universe_provider: UniverseProvider | None = None,
        fetcher: MarketDataFetcher | None = None,
        feature_provider: FeatureProvider | None = None,
        labeler: RiskLabeler | None = None,
    ):
        self._universe = universe_provider or UniverseProvider()
        self._fetch = fetcher or _default_fetcher
        self._features = feature_provider or FeatureProvider()
        self._labeler = labeler or RiskLabeler()

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        request.validate()
        tickers = self._universe.resolve(request)

        prices = self._fetch(tuple(tickers), request.period)
        skipped = {t: "no market data" for t in tickers if t not in prices.columns}
        available = [t for t in tickers if t in prices.columns]
        if not available:
            raise UserInputError("No market data for any selected ticker.")

        matrix, feature_skipped = self._features.compute(prices[available], request.features)
        skipped.update(feature_skipped)
        if len(matrix) < MIN_STOCKS_FOR_CLUSTERING:
            raise UserInputError(
                f"Need at least {MIN_STOCKS_FOR_CLUSTERING} stocks with usable data "
                f"to group; only {len(matrix)} available."
            )

        # Cluster on the user-selected features only; volatility/returns stay in
        # the matrix regardless because risk labeling depends on them. The rules
        # strategy always thresholds volatility, so it gets the full matrix.
        if request.strategy == StrategyName.RULES:
            cluster_matrix = matrix
        else:
            cols = [
                c
                for feature in request.features
                for c in _FEATURE_COLUMNS[feature]
                if c in matrix.columns
            ]
            cluster_matrix = matrix[cols]

        groups = make_strategy(request.strategy).assign_groups(cluster_matrix)
        labeled = self._labeler.label(matrix, groups)

        feature_cols = [c for c in matrix.columns]
        recommendations = [
            StockRecommendation(
                ticker=str(t),
                risk_label=row["risk_label"],
                risk_score=float(row["risk_score"]),
                group_id=int(row["group_id"]),
                features={c: float(row[c]) for c in feature_cols},
            )
            for t, row in labeled.iterrows()
        ]
        recommendations.sort(key=lambda r: r.risk_score, reverse=True)

        return RecommendationResult(
            recommendations=recommendations,
            skipped=skipped,
            profile=PortfolioProfile.from_recommendations(recommendations),
        )
