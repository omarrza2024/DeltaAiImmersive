"""Domain models for the investment assistant."""

from __future__ import annotations

from dataclasses import dataclass, field

from finance_dashboard.recommender.models import StockRecommendation


@dataclass
class InvestmentGoal:
    """Structured constraints extracted from the conversation."""

    target_annual_return_pct: float | None = None
    max_volatility_pct: float | None = None
    sectors: list[str] = field(default_factory=list)
    max_recommendations: int | None = None

    def is_complete(self) -> bool:
        return (
            self.target_annual_return_pct is not None
            and self.max_volatility_pct is not None
            and bool(self.sectors)
            and self.max_recommendations is not None
        )


@dataclass(frozen=True)
class Recommendation:
    """A scored stock produced by the assistant."""

    stock: StockRecommendation
    match_score: float          # 0-100, higher = closer to goal
    explanation: str            # LLM-authored justification
