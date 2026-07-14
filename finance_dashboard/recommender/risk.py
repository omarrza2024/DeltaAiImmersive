"""Maps similarity groups to Low / Medium / High risk labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from finance_dashboard.models import UserInputError
from finance_dashboard.recommender.models import RiskLabel

# Tier index -> label for the non-noise groups, keyed by how many tiers exist.
_TIER_LABELS: dict[int, list[str]] = {
    1: ["medium"],
    2: ["low", "high"],
    3: ["low", "medium", "high"],
}


class RiskLabeler:
    """
    Clustering finds similarity, not risk — this adds the ordering.

    Groups are ranked by median volatility and mapped onto Low/Medium/High tiers.
    DBSCAN noise points (-1) are treated as High risk: outlier behavior is risk.
    High-risk stocks split into Positive/Negative on the sign of their own return.
    """

    def label(self, matrix: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
        missing = {"volatility", "returns"} - set(matrix.columns)
        if missing:
            raise UserInputError(f"Risk labeling requires columns: {', '.join(sorted(missing))}")

        out = matrix.copy()
        out["group_id"] = groups

        real_groups = sorted(
            (g for g in groups.unique() if g != -1),
            key=lambda g: matrix.loc[groups == g, "volatility"].median(),
        )
        tier_names = _TIER_LABELS.get(len(real_groups))
        if tier_names is None:
            # More than 3 groups: split the volatility-ordered list into 3 tiers.
            tiers = np.array_split(real_groups, 3)
        else:
            # Map each ordered group onto its tier name's position.
            tiers = [
                [g for g, name_ in zip(real_groups, tier_names) if name_ == name]
                for name in ("low", "medium", "high")
            ]

        group_tier: dict[int, str] = {}
        for name, groups_in_tier in zip(("low", "medium", "high"), tiers):
            for g in groups_in_tier:
                group_tier[int(g)] = name

        tiers = pd.Series(
            {
                t: "high" if int(groups.loc[t]) == -1 else group_tier[int(groups.loc[t])]
                for t in out.index
            }
        )

        def label_for(ticker: str) -> RiskLabel:
            tier = tiers.loc[ticker]
            if tier == "low":
                return RiskLabel.LOW
            if tier == "medium":
                return RiskLabel.MEDIUM
            if matrix.loc[ticker, "returns"] >= 0:
                return RiskLabel.HIGH_POSITIVE
            return RiskLabel.HIGH_NEGATIVE

        out["risk_label"] = [label_for(t) for t in out.index]
        # Score = bucket band + volatility rank within the bucket, so sorting by
        # score can never place a lower-risk bucket above a higher one:
        # Low (0, 33.3], Medium (33.3, 66.7], High (66.7, 100].
        tier_index = tiers.map({"low": 0, "medium": 1, "high": 2})
        within_tier_rank = matrix["volatility"].groupby(tiers).rank(pct=True)
        out["risk_score"] = (tier_index + within_tier_rank) / 3.0 * 100.0
        return out
