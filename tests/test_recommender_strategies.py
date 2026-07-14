"""Tests for clustering strategies and risk labeling."""

import numpy as np
import pandas as pd
import pytest

from finance_dashboard.models import UserInputError
from finance_dashboard.recommender.models import RiskLabel, StrategyName
from finance_dashboard.recommender.risk import RiskLabeler
from finance_dashboard.recommender.strategies import (
    DBSCANStrategy,
    HardCodedRulesStrategy,
    KMeansStrategy,
    make_strategy,
)


def three_tier_matrix(h3_return: float = -0.30) -> pd.DataFrame:
    """Nine stocks in three well-separated volatility/return tiers."""
    data = {
        # low volatility, modest positive returns
        "L1": (0.08, 0.05), "L2": (0.09, 0.06), "L3": (0.10, 0.04),
        # medium volatility
        "M1": (0.28, 0.10), "M2": (0.30, 0.12), "M3": (0.29, 0.08),
        # high volatility — H3's return is configurable so labeler tests can
        # exercise the positive/negative split on data k-means never sees.
        "H1": (0.60, 0.50), "H2": (0.62, 0.40), "H3": (0.58, h3_return),
    }
    return pd.DataFrame(
        {"volatility": [v for v, _ in data.values()], "returns": [r for _, r in data.values()]},
        index=list(data),
    )


def groups_as_partition(groups: pd.Series) -> set[frozenset]:
    return {frozenset(groups.index[groups == g]) for g in groups.unique()}


def test_kmeans_recovers_three_tiers():
    # Tiers separable in both dimensions — k-means should recover the partition.
    matrix = three_tier_matrix(h3_return=0.45)
    groups = KMeansStrategy(n_clusters=3).assign_groups(matrix)
    assert groups_as_partition(groups) == {
        frozenset({"L1", "L2", "L3"}),
        frozenset({"M1", "M2", "M3"}),
        frozenset({"H1", "H2", "H3"}),
    }


def test_kmeans_is_deterministic():
    matrix = three_tier_matrix()
    a = KMeansStrategy(n_clusters=3).assign_groups(matrix)
    b = KMeansStrategy(n_clusters=3).assign_groups(matrix)
    assert a.equals(b)


def test_dbscan_marks_far_outlier_as_noise():
    matrix = three_tier_matrix()
    matrix.loc["OUT"] = [5.0, -3.0]  # far from every tier
    groups = DBSCANStrategy().assign_groups(matrix)
    assert groups.loc["OUT"] == -1


def test_hard_coded_rules_use_volatility_thresholds():
    matrix = three_tier_matrix()
    strategy = HardCodedRulesStrategy(low_vol=0.20, high_vol=0.40)
    groups = strategy.assign_groups(matrix)
    assert groups_as_partition(groups) == {
        frozenset({"L1", "L2", "L3"}),
        frozenset({"M1", "M2", "M3"}),
        frozenset({"H1", "H2", "H3"}),
    }


def test_make_strategy_maps_names():
    assert isinstance(make_strategy(StrategyName.KMEANS), KMeansStrategy)
    assert isinstance(make_strategy(StrategyName.DBSCAN), DBSCANStrategy)
    assert isinstance(make_strategy(StrategyName.RULES), HardCodedRulesStrategy)


# --- RiskLabeler ---


def test_labeler_ranks_groups_by_volatility():
    matrix = three_tier_matrix()
    groups = pd.Series([0] * 3 + [1] * 3 + [2] * 3, index=matrix.index)
    labeled = RiskLabeler().label(matrix, groups)

    for t in ("L1", "L2", "L3"):
        assert labeled.loc[t, "risk_label"] == RiskLabel.LOW
    for t in ("M1", "M2", "M3"):
        assert labeled.loc[t, "risk_label"] == RiskLabel.MEDIUM
    # High-risk bucket splits on the sign of each stock's return.
    assert labeled.loc["H1", "risk_label"] == RiskLabel.HIGH_POSITIVE
    assert labeled.loc["H2", "risk_label"] == RiskLabel.HIGH_POSITIVE
    assert labeled.loc["H3", "risk_label"] == RiskLabel.HIGH_NEGATIVE


def test_labeler_treats_dbscan_noise_as_high_risk():
    matrix = three_tier_matrix()
    groups = pd.Series([0] * 3 + [1] * 3 + [2] * 3, index=matrix.index)
    matrix.loc["OUT"] = [5.0, -3.0]
    groups.loc["OUT"] = -1
    labeled = RiskLabeler().label(matrix, groups)
    assert labeled.loc["OUT", "risk_label"] == RiskLabel.HIGH_NEGATIVE


def test_labeler_with_two_groups_uses_low_and_high():
    matrix = three_tier_matrix().loc[["L1", "L2", "L3", "H1", "H2", "H3"]]
    groups = pd.Series([0, 0, 0, 1, 1, 1], index=matrix.index)
    labeled = RiskLabeler().label(matrix, groups)
    assert labeled.loc["L1", "risk_label"] == RiskLabel.LOW
    assert labeled.loc["H1", "risk_label"] == RiskLabel.HIGH_POSITIVE


def test_labeler_with_single_group_uses_medium():
    matrix = three_tier_matrix().loc[["M1", "M2", "M3"]]
    groups = pd.Series([0, 0, 0], index=matrix.index)
    labeled = RiskLabeler().label(matrix, groups)
    assert (labeled["risk_label"] == RiskLabel.MEDIUM).all()


def test_risk_score_sorts_by_bucket_then_volatility():
    """Sorting by score must never rank a lower bucket above a higher one."""
    matrix = three_tier_matrix()
    groups = pd.Series([0] * 3 + [1] * 3 + [2] * 3, index=matrix.index)
    labeled = RiskLabeler().label(matrix, groups)

    assert labeled["risk_score"].between(0, 100).all()
    low = labeled.loc[["L1", "L2", "L3"], "risk_score"]
    med = labeled.loc[["M1", "M2", "M3"], "risk_score"]
    high = labeled.loc[["H1", "H2", "H3"], "risk_score"]
    assert low.max() < med.min() < med.max() < high.min()
    # Within a bucket, more volatile means a higher score.
    assert labeled.loc["H2", "risk_score"] == pytest.approx(100.0)  # highest vol overall
    assert labeled.loc["H3", "risk_score"] < labeled.loc["H1", "risk_score"]
    assert labeled.loc["L1", "risk_score"] < labeled.loc["L3", "risk_score"]


def test_labeler_requires_volatility_and_returns_columns():
    matrix = pd.DataFrame({"sma_ratio": [0.1, 0.2]}, index=["A", "B"])
    groups = pd.Series([0, 1], index=matrix.index)
    with pytest.raises(UserInputError, match="volatility"):
        RiskLabeler().label(matrix, groups)
