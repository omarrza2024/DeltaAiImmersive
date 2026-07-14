"""Grouping strategies: K-Means, DBSCAN, and hard-coded volatility rules."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from finance_dashboard.recommender.models import StrategyName

KMEANS_SEED = 42
RULES_LOW_VOL = 0.20
RULES_HIGH_VOL = 0.40


def _zscore(matrix: pd.DataFrame) -> np.ndarray:
    """Standardize columns so no single feature's scale dominates distances."""
    values = matrix.to_numpy(dtype=float)
    std = values.std(axis=0)
    std[std == 0] = 1.0
    return (values - values.mean(axis=0)) / std


class ClusteringStrategy(ABC):
    """Assigns each ticker an integer group id; -1 marks noise/outliers."""

    @abstractmethod
    def assign_groups(self, matrix: pd.DataFrame) -> pd.Series: ...


class KMeansStrategy(ClusteringStrategy):
    def __init__(self, n_clusters: int = 3, random_state: int = KMEANS_SEED):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def assign_groups(self, matrix: pd.DataFrame) -> pd.Series:
        from sklearn.cluster import KMeans

        n_clusters = min(self.n_clusters, len(matrix))
        model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = model.fit_predict(_zscore(matrix))
        return pd.Series(labels, index=matrix.index)


class DBSCANStrategy(ClusteringStrategy):
    def __init__(self, eps: float = 0.9, min_samples: int = 2):
        self.eps = eps
        self.min_samples = min_samples

    def assign_groups(self, matrix: pd.DataFrame) -> pd.Series:
        from sklearn.cluster import DBSCAN

        model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = model.fit_predict(_zscore(matrix))
        return pd.Series(labels, index=matrix.index)


class HardCodedRulesStrategy(ClusteringStrategy):
    """Threshold annualized volatility into three fixed buckets (0=low, 1=mid, 2=high)."""

    def __init__(self, low_vol: float = RULES_LOW_VOL, high_vol: float = RULES_HIGH_VOL):
        self.low_vol = low_vol
        self.high_vol = high_vol

    def assign_groups(self, matrix: pd.DataFrame) -> pd.Series:
        vol = matrix["volatility"]
        groups = pd.Series(1, index=matrix.index)
        groups[vol < self.low_vol] = 0
        groups[vol >= self.high_vol] = 2
        return groups


def make_strategy(name: StrategyName) -> ClusteringStrategy:
    strategies: dict[StrategyName, type[ClusteringStrategy]] = {
        StrategyName.KMEANS: KMeansStrategy,
        StrategyName.DBSCAN: DBSCANStrategy,
        StrategyName.RULES: HardCodedRulesStrategy,
    }
    return strategies[name]()
