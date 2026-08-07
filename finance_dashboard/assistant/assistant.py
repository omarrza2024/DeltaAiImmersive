"""InvestmentAssistant: conversational orchestrator reusing the recommender."""

from __future__ import annotations

from dataclasses import dataclass, field

from finance_dashboard.assistant.goal_parser import GoalParser, next_missing_question
from finance_dashboard.assistant.llm import ChatCompletion, groq_chat_completion
from finance_dashboard.assistant.matcher import score_match
from finance_dashboard.assistant.models import InvestmentGoal, Recommendation
from finance_dashboard.models import UserInputError
from finance_dashboard.recommender.models import (
    FeatureName,
    RecommendationRequest,
    StrategyName,
)
from finance_dashboard.recommender.recommender import StockRecommender

DISCLAIMER = "Historical data does not guarantee future returns."

_EXPLAIN_SYSTEM = """You are an investment assistant explaining stock recommendations.
You are given a user's stated goal and a shortlist of stocks (ticker, annual volatility %,
historical return %, match score 0-100) that Python already scored. Do NOT recompute anything.

Write a brief, plain-language response that:
- Names each stock and why it matches (or nearly matches) the goal.
- Pushes back if the user's constraints are unusually strict or contradictory.
- Always ends with the disclaimer: "Historical data does not guarantee future returns."
Keep it under 150 words.
"""


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM call fails — surfaces the underlying reason."""


@dataclass
class AssistantState:
    """Per-user session state."""

    goal: InvestmentGoal = field(default_factory=InvestmentGoal)
    messages: list[dict] = field(default_factory=list)
    custom_tickers: tuple[str, ...] = ()  # optional override, mainly for tests


@dataclass
class AssistantReply:
    message: str
    recommendations: list[Recommendation] = field(default_factory=list)
    no_perfect_match: bool = False


class InvestmentAssistant:
    def __init__(
        self,
        chat_completion: ChatCompletion | None = None,
        recommender: StockRecommender | None = None,
    ):
        self._llm = chat_completion or groq_chat_completion
        self._recommender = recommender or StockRecommender()
        self._parser = GoalParser(chat_completion=self._llm)

    def new_state(self) -> AssistantState:
        return AssistantState()

    def handle_message(self, state: AssistantState, user_message: str) -> AssistantReply:
        state.messages.append({"role": "user", "content": user_message})

        try:
            state.goal = self._parser.update(state.goal, state.messages)
        except UserInputError as exc:
            return self._record(state, str(exc))
        except Exception as exc:
            raise LLMUnavailableError(f"Goal parser failed: {exc}") from exc

        question = next_missing_question(state.goal)
        if question is not None:
            return self._record(state, question)

        return self._recommend(state)

    def _recommend(self, state: AssistantState) -> AssistantReply:
        goal = state.goal
        try:
            request = RecommendationRequest(
                universes=tuple(goal.sectors),
                custom_tickers=state.custom_tickers,
                features=(FeatureName.VOLATILITY, FeatureName.RETURNS),
                strategy=StrategyName.KMEANS,
                period="1y",
            )
            result = self._recommender.recommend(request)
        except UserInputError as exc:
            return self._record(state, f"Couldn't produce recommendations: {exc}")
        except Exception as exc:
            return self._record(
                state, f"Market data unavailable right now ({exc}). Please try again."
            )

        scored = sorted(
            [(s, score_match(s, goal)) for s in result.recommendations],
            key=lambda p: p[1],
            reverse=True,
        )
        top = scored[: goal.max_recommendations or 5]
        no_perfect_match = not any(score >= 80.0 for _, score in top)

        try:
            explanation = self._explain(goal, top, no_perfect_match)
        except Exception as exc:
            raise LLMUnavailableError(f"Explanation LLM failed: {exc}") from exc

        # Ensure the disclaimer is always present, regardless of what the LLM said.
        if DISCLAIMER.lower() not in explanation.lower():
            explanation = f"{explanation.rstrip()}\n\n_{DISCLAIMER}_"

        recommendations = [
            Recommendation(stock=s, match_score=round(score, 1), explanation=explanation)
            for s, score in top
        ]
        state.messages.append({"role": "assistant", "content": explanation})
        return AssistantReply(
            message=explanation,
            recommendations=recommendations,
            no_perfect_match=no_perfect_match,
        )

    def _explain(
        self,
        goal: InvestmentGoal,
        scored: list[tuple],
        no_perfect_match: bool,
    ) -> str:
        shortlist = "\n".join(
            f"- {s.ticker}: vol={s.features.get('volatility', 0) * 100:.1f}%, "
            f"return={s.features.get('returns', 0) * 100:.1f}%, match={score:.0f}"
            for s, score in scored
        )
        warning = (
            "\nIMPORTANT: none of these stocks meet the user's constraints closely; "
            "show them as CLOSEST matches and note the mismatch.\n"
            if no_perfect_match
            else ""
        )
        user_msg = (
            f"Goal: target return {goal.target_annual_return_pct}%, "
            f"max volatility {goal.max_volatility_pct}%, "
            f"sectors {goal.sectors}, top {goal.max_recommendations}.\n"
            f"Shortlist:\n{shortlist}{warning}"
        )
        return self._llm(_EXPLAIN_SYSTEM, [{"role": "user", "content": user_msg}])

    def _record(self, state: AssistantState, message: str) -> AssistantReply:
        state.messages.append({"role": "assistant", "content": message})
        return AssistantReply(message=message)
