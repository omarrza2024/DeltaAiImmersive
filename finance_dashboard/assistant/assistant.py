"""InvestmentAssistant: conversational orchestrator reusing the recommender."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from finance_dashboard.assistant.financials import financials_summary
from finance_dashboard.assistant.goal_parser import GoalParser, next_missing_question
from finance_dashboard.assistant.llm import ChatCompletion, groq_chat_completion
from finance_dashboard.assistant.matcher import score_match
from finance_dashboard.assistant.models import InvestmentGoal, Recommendation


def get_portfolio_summary(username: str) -> str:
    """Return a compact plain-text portfolio summary for the LLM's context.

    Best-effort — returns empty string if Mongo isn't configured, the user isn't
    signed in, or the user's portfolio is empty/unavailable.
    """
    if not username:
        return ""
    try:
        from finance_dashboard.auth.db import get_db
        from finance_dashboard.portfolio import analytics, store
        db = get_db()
        if db is None:
            return ""
        portfolio = store.get_or_create(db, username)
        if not portfolio.get("positions"):
            return ""
        # Lazy-import price/sector lookups to avoid Streamlit-context requirements
        # when this helper is called from tests without streamlit initialised.
        from finance_dashboard.portfolio.ui import _current_price, _sector
        positions = portfolio["positions"]
        prices = {p["ticker"]: _current_price(p["ticker"]) for p in positions}
        sectors = {p["ticker"]: _sector(p["ticker"]) for p in positions}
        enriched = analytics.enrich(positions, prices, sectors)
        return analytics.summary_for_llm(enriched)
    except Exception:
        return ""
from finance_dashboard.models import UserInputError
from finance_dashboard.recommender.models import (
    FeatureName,
    RecommendationRequest,
    StrategyName,
)
from finance_dashboard.recommender.recommender import StockRecommender

DISCLAIMER = "Historical data does not guarantee future returns."

_GATHERING_SYSTEM = """You are the user's warm, personable financial advisor — think
"knowledgeable friend who happens to help with investing," NOT a survey robot. Your job on
this turn is to acknowledge what the user just said and, if you still need info to build
their portfolio, ask for it naturally.

You will be given:
- The full recent chat
- A brief summary of what you already know about their goals
- What's still missing to make solid recommendations

Rules for THIS response:
- Acknowledge what they told you (validate, don't parrot). E.g., "10% is a reasonable target"
  or "Tech is a fun space to be in right now" — a short human beat.
- If they asked YOU a question, answer briefly BEFORE steering to the next thing you need.
- If their constraint is unrealistic (e.g. "20% return with zero risk", "get me rich next
  month"), push back gently and honestly — set expectations without being preachy.
- Then, in ONE natural sentence, ask about whatever piece is missing next. Don't list all
  the missing fields. Don't say "next question is". Don't sound like a form.
- Warm but efficient: 1-3 sentences total. Use conjunctions, contractions, occasional "we".
- Do NOT invent numbers. Do NOT re-list past choices. Do NOT thank them for "sharing".

CRITICAL — you are NOT allowed to:
- Name any specific stock, ticker, ETF, fund, or bond (no "VHT", "JNJ", "healthcare ETF",
  "blue-chip pharma", "dividend fund", etc.). Recommendations come from a separate
  calculation engine AFTER we have enough info. If the user is pushing for picks right now,
  briefly confirm the last missing detail so we can run the engine.
- Suggest categories that sound like recommendations ("consider a bond fund", "look at
  large-cap tech") — same rule: only the engine names things.
"""


_EXPLAIN_SYSTEM = """You are an investment assistant explaining stock recommendations.
You are given a user's stated goal and a shortlist of stocks (ticker, annual volatility %,
historical return %, match score 0-100) that Python already scored. Do NOT recompute anything.

Write a brief, plain-language response that:
- Names each stock and why it matches (or nearly matches) the goal.
- Pushes back if the user's constraints are unusually strict or contradictory.
- Always ends with the disclaimer: "Historical data does not guarantee future returns."
Keep it under 150 words.
"""

_CHAT_SYSTEM = """You are TradeMatrix's investment assistant, talking to a user who does
NOT have finance expertise. You have ALREADY given the user stock recommendations in this
conversation. Respond conversationally to their latest message — DO NOT re-list the
recommendations unless they explicitly ask you to.

If the message includes a `[SEC context: ...]` block, you have real SEC-filed data for one
of the recommended stocks. Use it to do fundamental analysis IN PLAIN ENGLISH:
- No jargon (no "EBITDA", "ROE", "P/E ratio") without immediately explaining it
- Use everyday analogies ("Revenue is the money the company brought in from selling stuff.
  Net income is what's left after paying all their bills.")
- Focus on 1–2 clear takeaways ("Revenue is growing, and they're generating more cash than
  they need to run the business — that's a healthy sign") rather than reciting every number
- Highlight the SIGNAL, not the raw numbers

If they want new/different recommendations, tell them to name the constraint they want
changed (e.g. "higher return target", "different sector") — do not invent numbers.
If they're just chatting, chat naturally and briefly. Push back on hallucinations.

Do NOT invent price, return, or volatility figures beyond what's in the context.
Under 180 words. End with: "Historical data does not guarantee future returns."
"""

# Words that suggest a fundamental-analysis question (used to decide whether
# to include SEC context for the top pick when no specific ticker is named).
_FUNDAMENTAL_KEYWORDS = (
    "fundamental", "financial", "revenue", "profit", "earnings", "cash flow",
    "balance sheet", "debt", "margin", "10-k", "10-q", "sec", "filing",
    "healthy", "growing", "grow", "sales", "book value",
)


# Phrases that indicate "just proceed — I've told you enough" intent.
_PROCEED_KEYWORDS = (
    "as many as", "as many that", "all that fit", "all the stocks", "all stocks",
    "give me all", "give me the stocks", "give me the picks", "just give",
    "just do it", "just show", "proceed", "show me", "put as many",
    "any that fit", "any stocks", "whatever fits", "let's see", "lets see",
    "go ahead", "do it", "bring them", "hit me",
)


def _user_wants_to_proceed(message: str) -> bool:
    """Detect intent to skip further questions and just get recommendations."""
    low = (message or "").lower()
    return any(kw in low for kw in _PROCEED_KEYWORDS)


def _describe_known(goal: InvestmentGoal) -> str:
    """One-line summary of constraints the user has already told us."""
    bits: list[str] = []
    if goal.target_annual_return_pct is not None:
        bits.append(f"targets ~{goal.target_annual_return_pct:g}% annual return")
    if goal.risk_preference:
        bits.append(f"{goal.risk_preference} risk tolerance")
    elif goal.max_volatility_pct is not None:
        bits.append(f"volatility cap ~{goal.max_volatility_pct:g}%")
    if goal.sectors:
        bits.append(f"interested in {', '.join(goal.sectors)}")
    if goal.max_recommendations:
        bits.append(f"wants {goal.max_recommendations} pick(s)")
    return "; ".join(bits) if bits else "nothing yet"


def _describe_missing(goal: InvestmentGoal) -> str:
    """Plain-English list of what still needs to be gathered."""
    missing: list[str] = []
    if goal.target_annual_return_pct is None:
        missing.append("their target annual return")
    if goal.max_volatility_pct is None and not goal.risk_preference:
        missing.append("their comfort with risk / volatility")
    if not goal.sectors:
        missing.append("which sectors interest them (Tech, Medical, Finance)")
    if goal.max_recommendations is None:
        missing.append("how many recommendations they'd like")
    return "; ".join(missing) if missing else "nothing — you have enough to make picks"


def _tickers_for_sec_context(user_message: str, recs: list[Recommendation]) -> list[str]:
    """Pick which recommended tickers deserve SEC context on this turn.

    Explicit mention in the message > fundamental-analysis keyword hits top pick.
    Neither present → no SEC lookup (keeps chat cheap and context tight).
    """
    if not recs:
        return []
    rec_tickers = [r.stock.ticker for r in recs]
    upper = user_message.upper()
    mentioned = [t for t in rec_tickers if t in upper]
    if mentioned:
        return mentioned[:3]  # cap to avoid context bloat
    lower = user_message.lower()
    if any(kw in lower for kw in _FUNDAMENTAL_KEYWORDS):
        return rec_tickers[:1]
    return []


def _goal_materially_changed(before: InvestmentGoal, after: InvestmentGoal) -> bool:
    """True when a re-recommendation would produce different results."""
    return (
        before.target_annual_return_pct != after.target_annual_return_pct
        or before.max_volatility_pct   != after.max_volatility_pct
        or before.sectors              != after.sectors
        or before.max_recommendations  != after.max_recommendations
        or before.risk_preference      != after.risk_preference
    )


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM call fails — surfaces the underlying reason."""


@dataclass
class AssistantState:
    """Per-user session state."""

    goal: InvestmentGoal = field(default_factory=InvestmentGoal)
    messages: list[dict] = field(default_factory=list)
    custom_tickers: tuple[str, ...] = ()  # optional override, mainly for tests
    # Set once we've delivered recommendations — after that, further user
    # messages go through chat/refine mode instead of re-running the pipeline.
    has_recommended: bool = False
    last_recommendations: list[Recommendation] = field(default_factory=list)
    # Username of the signed-in user — set by the UI when a session starts, so
    # the assistant can look up their portfolio for context. Empty in tests.
    username: str = ""


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

        prev_goal = replace(state.goal)
        try:
            state.goal = self._parser.update(state.goal, state.messages)
        except UserInputError as exc:
            return self._record(state, str(exc))
        except Exception as exc:
            raise LLMUnavailableError(f"Goal parser failed: {exc}") from exc

        # Follow-up mode: we've already given a shortlist. Only re-run the
        # pipeline if the user *actually* changed a constraint — otherwise
        # treat it as chat so "just chat with me" doesn't spam the same stocks.
        if state.has_recommended:
            if _goal_materially_changed(prev_goal, state.goal):
                return self._recommend(state)
            return self._chat(state)

        # Auto-defaults — apply so the conversation doesn't stall.
        g = state.goal

        # (a) If user says "as many as fit my criteria" or similar "proceed"
        #     intent, and we already know return + sectors, fill in reasonable
        #     defaults for anything still missing (risk defaults to medium,
        #     count defaults to 5). Prevents the "just do it" case where the
        #     goal never completes.
        if (
            _user_wants_to_proceed(user_message)
            and g.target_annual_return_pct is not None
            and g.sectors
        ):
            if g.max_volatility_pct is None and not g.risk_preference:
                g.risk_preference = "medium"
                g.max_volatility_pct = 35.0
            if g.max_recommendations is None:
                g.max_recommendations = 5

        # (b) If only the count is missing and we have return + risk + sectors,
        #     default to 5 without asking.
        if (
            g.max_recommendations is None
            and g.target_annual_return_pct is not None
            and (g.max_volatility_pct is not None or g.risk_preference)
            and g.sectors
        ):
            g.max_recommendations = 5

        if state.goal.is_complete():
            return self._recommend(state)

        # Still gathering — compose a warm, conversational response instead of
        # reading a hardcoded question off the list. Falls back to the scripted
        # question only if the LLM composer fails (so the flow never stalls).
        return self._compose_gathering_response(state)

    def _compose_gathering_response(self, state: AssistantState) -> AssistantReply:
        context = (
            f"What you already know: {_describe_known(state.goal)}.\n"
            f"Still missing: {_describe_missing(state.goal)}."
        )
        try:
            reply = self._llm(
                _GATHERING_SYSTEM,
                [
                    {"role": "system", "content": context},
                    *state.messages[-8:],
                ],
            )
            if not reply.strip():
                raise RuntimeError("empty response")
            return self._record(state, reply.strip())
        except Exception:
            # Composer failed — fall back to the scripted next question so we
            # never leave the user with nothing.
            fallback = next_missing_question(state.goal) or "What else should I know?"
            return self._record(state, fallback)

    def _chat(self, state: AssistantState) -> AssistantReply:
        """Conversational follow-up after recommendations exist — no re-recommend."""
        goal = state.goal
        summary = (
            f"User's goal: target return {goal.target_annual_return_pct}%, "
            f"max volatility {goal.max_volatility_pct}%, sectors {goal.sectors}, "
            f"top {goal.max_recommendations}.\n"
            f"Already recommended: "
            + ", ".join(
                f"{r.stock.ticker} "
                f"(match {r.match_score:.0f}, "
                f"return {r.stock.features.get('returns', 0) * 100:+.1f}%, "
                f"vol {r.stock.features.get('volatility', 0) * 100:.1f}%)"
                for r in state.last_recommendations
            )
            + "."
        )

        # If the user's latest turn is about fundamentals (or names a specific
        # stock), fetch SEC data for the relevant ticker(s) so the LLM can do
        # real fundamental analysis instead of guessing from memory.
        last_user_msg = next(
            (m["content"] for m in reversed(state.messages) if m["role"] == "user"),
            "",
        )
        sec_tickers = _tickers_for_sec_context(last_user_msg, state.last_recommendations)
        sec_blocks = [financials_summary(t) for t in sec_tickers]
        sec_context = ""
        if sec_blocks:
            sec_context = "[SEC context — use this for fundamental analysis]\n" + "\n\n".join(sec_blocks)

        portfolio_summary = get_portfolio_summary(state.username)

        try:
            reply = self._llm(
                _CHAT_SYSTEM,
                [
                    {"role": "system", "content": summary},
                    *([{"role": "system", "content":
                        "[User's current portfolio — reference when relevant]\n" + portfolio_summary}]
                      if portfolio_summary else []),
                    *([{"role": "system", "content": sec_context}] if sec_context else []),
                    *state.messages[-6:],
                ],
            )
        except Exception as exc:
            raise LLMUnavailableError(f"Chat LLM failed: {exc}") from exc

        if DISCLAIMER.lower() not in reply.lower():
            reply = f"{reply.rstrip()}\n\n_{DISCLAIMER}_"
        state.messages.append({"role": "assistant", "content": reply})
        return AssistantReply(message=reply)

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
        state.has_recommended = True
        state.last_recommendations = recommendations
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
