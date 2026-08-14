"""GoalParser: LLM extracts InvestmentGoal fields from the chat transcript."""

from __future__ import annotations

import json

from finance_dashboard.assistant.llm import ChatCompletion, groq_chat_completion
from finance_dashboard.assistant.models import InvestmentGoal
from finance_dashboard.models import UserInputError
from finance_dashboard.recommender.universe import PREDEFINED_UNIVERSES

_SYSTEM_PROMPT = f"""You extract structured investment constraints from a user's chat messages.
Return ONLY a JSON object with exactly these keys:
  target_annual_return_pct  (number, e.g. 10.0 for 10%; null if not mentioned)
  max_volatility_pct        (number; ONLY extract if the user gives an explicit % — do NOT convert risk words. Also set from a risk word as a safety cap: low=20, medium=35, high=60)
  risk_preference           (string "low" | "medium" | "high"; ONLY set if the user used a risk word like low/conservative/safe (=low), moderate/medium/balanced (=medium), high/aggressive/growth (=high). null if they gave only a number.)
  sectors                   (array of strings from {sorted(PREDEFINED_UNIVERSES)}; null if not mentioned)
  max_recommendations       (integer; null if not mentioned)

Rules:
- Never invent a value the user did not provide, however weakly.
- If the user changes their mind, return the newest value.
- If they say something incoherent (e.g. negative return target), still extract literally.
"""

_QUESTIONS = [
    ("target_annual_return_pct", "What annual return are you targeting (e.g. 8%, 12%)?"),
    ("max_volatility_pct", "What's the maximum volatility/risk you're willing to accept? (say low, moderate, high, or a % like 20%)"),
    ("sectors", f"Which sectors interest you? Options: {', '.join(sorted(PREDEFINED_UNIVERSES))}."),
    ("max_recommendations", "How many recommendations would you like (e.g. 3, 5, 10)?"),
]


def next_missing_question(goal: InvestmentGoal) -> str | None:
    """Return the next predefined question for a field the user hasn't answered."""
    for field, question in _QUESTIONS:
        value = getattr(goal, field)
        if value is None or (isinstance(value, list) and not value):
            return question
    return None


class GoalParser:
    def __init__(self, chat_completion: ChatCompletion | None = None):
        self._llm = chat_completion or groq_chat_completion

    def update(self, goal: InvestmentGoal, messages: list[dict]) -> InvestmentGoal:
        """Merge any newly-extracted fields into the existing goal."""
        raw = self._llm(_SYSTEM_PROMPT, messages, json_mode=True)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise UserInputError(f"LLM returned invalid JSON: {raw[:120]}") from exc

        sectors = data.get("sectors")
        if sectors:
            unknown = [s for s in sectors if s not in PREDEFINED_UNIVERSES]
            if unknown:
                raise UserInputError(
                    f"Unknown sector(s): {', '.join(unknown)}. "
                    f"Supported: {', '.join(sorted(PREDEFINED_UNIVERSES))}."
                )
            goal.sectors = sectors

        if data.get("target_annual_return_pct") is not None:
            goal.target_annual_return_pct = float(data["target_annual_return_pct"])
        if data.get("max_volatility_pct") is not None:
            goal.max_volatility_pct = float(data["max_volatility_pct"])
        if data.get("max_recommendations") is not None:
            goal.max_recommendations = int(data["max_recommendations"])
        pref = data.get("risk_preference")
        if pref in ("low", "medium", "high"):
            goal.risk_preference = pref
        return goal
