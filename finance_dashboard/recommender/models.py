"""Recommender domain models: request, result, enums."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from finance_dashboard.models import UserInputError, dedupe_tickers, parse_ticker_list


class FeatureName(str, Enum):
    SMA = "sma"
    VOLATILITY = "volatility"
    RETURNS = "returns"
    MACD = "macd"
    FUNDAMENTALS = "fundamentals"
    SENTIMENT = "sentiment"


class StrategyName(str, Enum):
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    RULES = "rules"


class RiskLabel(str, Enum):
    LOW = "Low Risk"
    MEDIUM = "Medium Risk"
    HIGH_POSITIVE = "High Risk (Positive)"
    HIGH_NEGATIVE = "High Risk (Negative)"


@dataclass(frozen=True)
class RecommendationRequest:
    """Everything the user selects: universe sources, features, and strategy."""

    etf: str | None = None
    universes: tuple[str, ...] = ()
    custom_tickers: tuple[str, ...] = ()
    features: tuple[FeatureName, ...] = (FeatureName.VOLATILITY, FeatureName.RETURNS)
    strategy: StrategyName = StrategyName.KMEANS
    period: str = "1y"

    def validate(self) -> None:
        if not self.etf and not self.universes and not self.custom_tickers:
            raise UserInputError(
                "Select at least one source: an ETF, a predefined universe, or custom tickers."
            )
        if not self.features:
            raise UserInputError("Select at least one feature.")

    @classmethod
    def from_user_input(
        cls,
        *,
        etf: str | None = None,
        universe_names: list[str] | None = None,
        custom_tickers_text: str = "",
        feature_names: list[str] | None = None,
        strategy_name: str = StrategyName.KMEANS.value,
        period: str = "1y",
    ) -> RecommendationRequest:
        """Build a validated request from raw UI values."""
        features: list[FeatureName] = []
        for name in feature_names or []:
            try:
                features.append(FeatureName(name))
            except ValueError as exc:
                raise UserInputError(f"Unknown feature: {name}") from exc

        try:
            strategy = StrategyName(strategy_name)
        except ValueError as exc:
            raise UserInputError(f"Unknown strategy: {strategy_name}") from exc

        request = cls(
            etf=etf or None,
            universes=tuple(universe_names or ()),
            custom_tickers=tuple(dedupe_tickers(parse_ticker_list(custom_tickers_text))),
            features=tuple(features),
            strategy=strategy,
            period=period,
        )
        request.validate()
        return request


@dataclass(frozen=True)
class StockRecommendation:
    """One labeled stock in the result."""

    ticker: str
    risk_label: RiskLabel
    risk_score: float
    group_id: int
    features: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # RiskLabel is a str-Enum, so pandas (with future.infer_string, as on
        # Streamlit Cloud) can coerce it to plain str during a DataFrame
        # roundtrip. Coerce back so `.value` and isinstance checks work.
        if not isinstance(self.risk_label, RiskLabel):
            object.__setattr__(self, "risk_label", RiskLabel(self.risk_label))


@dataclass(frozen=True)
class PortfolioProfile:
    """Distribution of the resulting universe across risk buckets."""

    counts: dict[RiskLabel, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @classmethod
    def from_recommendations(
        cls, recommendations: list[StockRecommendation]
    ) -> PortfolioProfile:
        counts = {label: 0 for label in RiskLabel}
        for rec in recommendations:
            counts[rec.risk_label] += 1
        return cls(counts=counts)


@dataclass(frozen=True)
class RecommendationResult:
    """Labeled stocks (highest risk score first) plus skipped tickers with reasons."""

    recommendations: list[StockRecommendation]
    skipped: dict[str, str]
    profile: PortfolioProfile
