"""InvestmentAssistant: conversational orchestrator reusing the recommender."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from finance_dashboard.assistant.financials import financials_summary
from finance_dashboard.assistant.goal_parser import GoalParser, next_missing_question
from finance_dashboard.assistant.llm import ChatCompletion, groq_chat_completion
from finance_dashboard.assistant.matcher import score_match
from finance_dashboard.assistant.models import InvestmentGoal, Recommendation


def get_watchlists_summary(username: str) -> str:
    """Return a compact plain-text summary of the user's watchlists for LLM context."""
    if not username:
        return ""
    try:
        from finance_dashboard.auth.db import get_db
        from finance_dashboard.watchlist import store
        db = get_db()
        if db is None:
            return ""
        watchlists = store.list_for_user(db, username)
        if not watchlists:
            return ""
        lines = [f"User's watchlists ({len(watchlists)}):"]
        for wl in watchlists:
            tickers = wl.get("tickers", [])
            if tickers:
                lines.append(f"- {wl['name']}: {', '.join(tickers)}")
            else:
                lines.append(f"- {wl['name']}: (empty)")
        return "\n".join(lines)
    except Exception:
        return ""


from finance_dashboard.models import UserInputError
from finance_dashboard.recommender.models import (
    FeatureName,
    RecommendationRequest,
    StrategyName,
)
from finance_dashboard.recommender.recommender import StockRecommender
from finance_dashboard.recommender.universe import PREDEFINED_UNIVERSES
from finance_dashboard.sentiment import SentimentResult, fetch_sentiment

DISCLAIMER = "Historical data does not guarantee future returns."

# A full tour of how the TradeMatrix app computes everything. Injected into
# every system prompt so the LLM can answer questions like "how is risk
# calculated?" or "what's the rolling window?" with the real numbers and
# formulas the Python code uses — never a hallucination.
APP_KNOWLEDGE = """You are the AI assistant inside TradeMatrix, a Streamlit-based
finance app that groups stocks into risk buckets and lets users explore price,
volatility, correlation, watchlists, and news sentiment. You have deep
knowledge of EXACTLY how the app computes everything. When the user asks
about the system, answer with these specifics — do NOT make up alternate
formulas:

RISK SCORE (0–100, per stock):
- Stocks are first clustered by selected features (K-Means by default,
  DBSCAN or hard-coded rules optional).
- Clusters are ordered by their MEDIAN VOLATILITY and mapped onto three
  tiers: Low, Medium, High.
- High-risk is split into HIGH POSITIVE / HIGH NEGATIVE by the sign of the
  stock's own period return.
- Final formula: `risk_score = (tier_index + within_tier_rank) / 3 * 100`,
  where tier_index is 0 for Low, 1 for Medium, 2 for High, and
  within_tier_rank is that stock's volatility percentile INSIDE its tier.
- Bands are non-overlapping: Low 0–33.3, Medium 33.3–66.7, High 66.7–100.
  A Low-Risk stock can never score higher than a Medium one.

VOLATILITY:
- Annualized from daily log returns, multiplied by √TRADING_DAYS (252).
- Rolling volatility uses a 21-day rolling window (one trading month).

PRICE / RETURN features:
- `returns` is total period return from the first to the last price
  (price_end / price_start − 1).
- SMA uses a 20-day simple moving average; `sma_ratio = price / SMA − 1`.
- MACD uses fast=12, slow=26, signal=9 EMAs; the histogram is normalized by
  the current price.

FUNDAMENTALS (feature labeled "Fundamentals (SEC DATA REPORTS)"):
- Sourced directly from SEC EDGAR filings via EdgarTools.
- Computes two numbers per stock:
  - revenue_growth: YoY revenue growth from the issuer's income statement
  - profit_margin:  net income / revenue, latest fiscal period

SENTIMENT (feature labeled "Sentiment"):
- Sourced live from marketaux news API.
- Averages the entity sentiment_score across up to 3 recent articles per
  ticker (free marketaux tier is capped at 3). Score range is [-1, +1].
- Labels: > +0.05 = Positive, < -0.05 = Negative, otherwise Neutral.

UNIVERSES & SOURCES:
- Users can pick any ETF (SPY, QQQ, DIA, IWM, VOO, VTI, VGT, VHT, VFH, and
  Select Sector SPDRs: XLK, XLV, XLF, XLE, XLY, XLP, XLI, XLU, XLC, XLRE,
  XLB), one or more predefined universes (Tech, Medical, Finance, Energy,
  Consumer, Industrials, Communications, Utilities, Real Estate, Materials),
  and / or their own custom tickers.
- Live ETF holdings come from yfinance; the app has fallback top-holdings
  lists if yfinance returns nothing.

OTHER TABS:
- Compare: side-by-side annualized + rolling volatility for two tickers.
- Correlation: pairwise log-return correlation matrix + heatmap.
- Watchlist: per-user saved lists with inline detail view, time-range
  switching per stock, and a Compare picker that drives normalized returns +
  correlation views.

GENERAL FINANCE QUESTIONS:
- You can also chat about general finance concepts (what volatility means,
  how diversification works, what a P/E ratio is, etc.) — explain in plain
  English using everyday analogies. Keep answers under 200 words unless the
  user asks for depth.
- NEVER invent specific stock prices, returns, volatility, or sentiment
  figures that aren't provided in your context — tell the user to run the
  Recommender for live numbers instead.
"""

_GATHERING_SYSTEM = """You are the user's warm, personable financial advisor — think
"knowledgeable friend who happens to help with investing," NOT a survey robot. Your job on
this turn is to acknowledge what the user just said and move the conversation forward.

You will be given:
- The full recent chat
- A brief summary of what you already know about their goals
- What's still open

CRITICAL — what the engine actually needs:
- NOTHING is required. The user can skip every question and we'll still run.
- If they name a sector (Tech / Medical / Finance), great — we'll scope the search there.
- If they skip the sector question — saying "any" / "idk" / "whatever" / "just pick for
  me based on SEC data" — we'll run across EVERY sector automatically. Do NOT keep
  asking. Confirm the fallback and tell them we're pulling picks.
- Target return %, risk comfort, how many picks — all optional. Offer, never demand.

Rules for THIS response:
- Acknowledge what they told you (validate, don't parrot). A short human beat.
- If they asked YOU a question — ABOUT THE APP (risk score, volatility, rolling
  window, SEC data, sentiment, any feature) or ABOUT FINANCE IN GENERAL (what
  is P/E, how does diversification work, what's a bull market) — answer it
  CONCRETELY using the APP KNOWLEDGE context. For app questions, cite the
  real formula / window / source. For general-finance questions, explain in
  plain English with everyday analogies. Then move on to goal-gathering only
  if that still makes sense in context.
- If their constraint is unrealistic (e.g. "20% return with zero risk"), push back gently.
- If sectors are still missing: ask once in ONE natural sentence AND mention they can
  say "any" or "just pick for me" to skip it.
- If sectors ARE set and the user hasn't shown interest in specifying return/risk/count:
  do NOT demand those. Confirm you're ready to pull picks and mention they can
  optionally share a return target or risk preference if they want finer control.
- Warm but efficient: 1-3 sentences total. Use conjunctions, contractions, occasional "we".
- Do NOT invent numbers. Do NOT re-list past choices. Do NOT thank them for "sharing".

CRITICAL — you are NOT allowed to:
- Name any specific stock, ticker, ETF, fund, or bond. Recommendations come from a
  separate calculation engine. If the user is pushing for picks right now and sectors
  are known, confirm you'll run the engine now.
- Suggest categories that sound like recommendations ("consider a bond fund", "look at
  large-cap tech") — same rule: only the engine names things.
"""


_EXPLAIN_SYSTEM = """You are an investment assistant explaining stock recommendations.
You are given a user's stated goal and a shortlist of stocks (ticker, annual volatility %,
historical return %, match score 0-100) that Python already scored. Some lines may also
include news sentiment (positive/negative/neutral with an average score and article count)
pulled from recent news. Do NOT recompute anything.

Write a brief, plain-language response that:
- Names each stock and why it matches (or nearly matches) the goal.
- When news sentiment is provided, briefly note the slant ("news is reading positive on
  AAPL right now across 3 articles") — a one-line tie-breaker, not a lecture.
- Pushes back if the user's constraints are unusually strict or contradictory.
- Always ends with the disclaimer: "Historical data does not guarantee future returns."
Keep it under 180 words.
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

If the message includes a `[News sentiment — EXACT numbers, cite these when the user asks]`
block, those are real marketaux-sourced scores — USE THEM VERBATIM:
- When the user asks about market sentiment, QUOTE the exact average score for each
  relevant ticker (e.g. "AAPL is sitting at +0.234, positive across 3 articles") — don't
  round aggressively, don't invent, don't say "roughly" when you have the number.
- Say Positive / Negative / Neutral using the label the context gives you.
- You do NOT need to list the article URLs in your prose — the UI shows them below your
  message in an expander — but you may name one headline if it illustrates a point.

If they ask ABOUT THE APP (how risk score is calculated, what rolling window
means, where SEC data comes from, which ETFs are supported, etc.) answer
PRECISELY using the APP KNOWLEDGE context — cite the real formulas, windows,
sources. If they ask a GENERAL FINANCE question (what is beta, how do
options work, what's a bull market, what does P/E mean), explain in plain
English with everyday analogies, within 200 words. The assistant is a
full-service finance chat, not only a recommendation explainer.

If they want new/different recommendations, tell them to name the constraint they want
changed (e.g. "higher return target", "different sector") — do not invent numbers.
If they're just chatting, chat naturally and briefly. Push back on hallucinations.

Do NOT invent price, return, volatility, or sentiment figures beyond what's in the context.
Under 200 words. End with: "Historical data does not guarantee future returns."
"""

# Words that suggest a fundamental-analysis question (used to decide whether
# to include SEC context for the top pick when no specific ticker is named).
_FUNDAMENTAL_KEYWORDS = (
    "fundamental", "financial", "revenue", "profit", "earnings", "cash flow",
    "balance sheet", "debt", "margin", "10-k", "10-q", "sec", "filing",
    "healthy", "growing", "grow", "sales", "book value",
)

# Words that suggest a news / sentiment question. When these appear, we fetch
# marketaux sentiment for every recommended ticker and inject the exact scores
# plus the contributing article URLs into the LLM's system context.
_SENTIMENT_KEYWORDS = (
    "sentiment", "news", "article", "articles", "headline", "headlines",
    "buzz", "coverage", "chatter", "press", "media", "reaction", "mood",
    "feeling", "vibe", "hype", "bullish", "bearish",
)


# Phrases that mean "I don't care which sector" or "just pick for me based on
# fundamentals / SEC data". When the user says one of these and we still have
# no sectors, we fall back to the full universe so the engine can run instead
# of grilling them for a sector pick.
_NO_PREFERENCE_PHRASES = (
    "any sector", "all sector", "any sectors", "all sectors", "every sector",
    "any", "all", "whatever", "doesn't matter", "doesnt matter",
    "no preference", "idk", "i don't know", "i dont know", "not sure",
    "just pick", "pick for me", "you pick", "surprise me", "pick anything",
    "based on sec", "sec only", "sec data", "just fundamentals", "fundamentals only",
    "proceed", "go ahead", "just go", "whatever you think", "whatever fits",
)


def _user_wants_any_sector(message: str) -> bool:
    """True when the user signals they don't want to pick a sector."""
    low = (message or "").lower()
    return any(phrase in low for phrase in _NO_PREFERENCE_PHRASES)


def _fetch_sentiment_for_shortlist(
    shortlist: list[tuple],
) -> dict[str, SentimentResult]:
    """Best-effort sentiment lookup for the recommended tickers.

    Marketaux may be unavailable (missing key, rate-limited, network down) or
    this call may be running outside Streamlit (tests). Any failure is caught
    so sentiment is a bonus signal, never a blocker for recommendations.
    """
    out: dict[str, SentimentResult] = {}
    for stock, _ in shortlist:
        try:
            out[stock.ticker] = fetch_sentiment(stock.ticker)
        except Exception:
            continue
    return out


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
    """Plain-English list of what still needs to be gathered.

    Everything is optional — sector too. When the user says "any" / "idk" /
    "pick based on SEC data" we fall back to the full universe and run, so
    the LLM must OFFER these questions without DEMANDING answers.
    """
    if not goal.sectors:
        return (
            "which sectors interest them (Tech, Medical, Finance) — optional; "
            "if they say 'any', 'idk', 'all', or 'just pick based on SEC data' "
            "we'll run across every sector"
        )
    optional: list[str] = []
    if goal.target_annual_return_pct is None:
        optional.append("target annual return (optional)")
    if goal.max_volatility_pct is None and not goal.risk_preference:
        optional.append("risk comfort (optional)")
    if goal.max_recommendations is None:
        optional.append("how many picks (optional, defaults to 5)")
    if optional:
        return (
            "; ".join(optional)
            + " — if the user doesn't want to specify these, proceed with "
              "fundamentals-driven defaults"
        )
    return "nothing — you have everything"


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


def _user_asked_about_sentiment(message: str) -> bool:
    """True when the user is asking about news / market sentiment."""
    low = (message or "").lower()
    return any(kw in low for kw in _SENTIMENT_KEYWORDS)


def _tickers_for_sentiment_context(
    user_message: str, recs: list[Recommendation]
) -> list[str]:
    """Which recommended tickers get a fresh sentiment pull this turn.

    - If the user named specific tickers → those (cap 5).
    - Else if the message is sentiment-flavored → every recommended ticker.
    - Else nothing (keeps the API quota and context compact).
    """
    if not recs:
        return []
    rec_tickers = [r.stock.ticker for r in recs]
    upper = user_message.upper()
    mentioned = [t for t in rec_tickers if t in upper]
    if mentioned:
        return mentioned[:5]
    if _user_asked_about_sentiment(user_message):
        return rec_tickers[:5]
    return []


def _format_sentiment_context(ticker_results: dict[str, SentimentResult]) -> str:
    """Pack precise marketaux numbers + article URLs into the LLM system block."""
    if not ticker_results:
        return ""
    lines = ["[News sentiment — EXACT numbers, cite these when the user asks]"]
    for ticker, result in ticker_results.items():
        if result.article_count == 0:
            lines.append(f"- {ticker}: no articles returned (score 0.000, neutral).")
            continue
        lines.append(
            f"- {ticker}: avg sentiment {result.avg_score:+.3f} "
            f"({result.label}) across {result.article_count} article"
            f"{'s' if result.article_count != 1 else ''}:"
        )
        for article in getattr(result, "articles", ()) or ():
            date = (article.published_at or "")[:10]
            lines.append(
                f"    • {article.title} ({article.source or 'unknown source'}, "
                f"{date}, score {article.score:+.2f}) {article.url}"
            )
    return "\n".join(lines)


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
    # Per-ticker marketaux results from this turn — used by the UI to render
    # an "articles used" expander. Empty on non-sentiment turns.
    sentiment: dict[str, SentimentResult] = field(default_factory=dict)


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
        except UserInputError:
            # LLM returned non-JSON — usually means the user asked a general
            # finance question, not a goal update. Keep goal as-is and let
            # the gathering/chat response answer the question naturally.
            pass
        except Exception as exc:
            # Groq 400 ("Failed to generate JSON") is the same scenario — the
            # model declined to produce JSON because the user wasn't talking
            # about their goal. Skip quietly; the chat path will handle it.
            if "json" in str(exc).lower() or "400" in str(exc):
                pass
            else:
                raise LLMUnavailableError(f"Goal parser failed: {exc}") from exc

        # Follow-up mode: we've already given a shortlist. Only re-run the
        # pipeline if the user *actually* changed a constraint — otherwise
        # treat it as chat so "just chat with me" doesn't spam the same stocks.
        if state.has_recommended:
            if _goal_materially_changed(prev_goal, state.goal):
                return self._recommend(state)
            return self._chat(state)

        # No-preference fallback: when the user says "any" / "idk" / "just pick"
        # / "sec data" and we still have no sectors, use the FULL universe so
        # the engine can run. The user still saw the sector question — they
        # chose not to answer, which is a valid answer.
        if not state.goal.sectors and _user_wants_any_sector(user_message):
            state.goal.sectors = sorted(PREDEFINED_UNIVERSES.keys())

        # Count defaults to 5 whenever the user hasn't picked one — matches the
        # old UX where users who skip "how many?" still get a shortlist. Done
        # pre-recommend so state.goal reflects it for the UI/tests.
        if state.goal.sectors and state.goal.max_recommendations is None:
            state.goal.max_recommendations = 5

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
                    {"role": "system", "content": APP_KNOWLEDGE},
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

        # Fresh sentiment pull when the user asks about news / sentiment.
        sent_tickers = _tickers_for_sentiment_context(last_user_msg, state.last_recommendations)
        sentiment_results: dict[str, SentimentResult] = {}
        for ticker in sent_tickers:
            try:
                sentiment_results[ticker] = fetch_sentiment(ticker)
            except Exception:
                continue
        sentiment_context = _format_sentiment_context(sentiment_results)

        watchlists_summary = get_watchlists_summary(state.username)

        try:
            reply = self._llm(
                _CHAT_SYSTEM,
                [
                    {"role": "system", "content": APP_KNOWLEDGE},
                    {"role": "system", "content": summary},
                    *([{"role": "system", "content":
                        "[User's watchlists — tickers they're tracking]\n" + watchlists_summary}]
                      if watchlists_summary else []),
                    *([{"role": "system", "content": sec_context}] if sec_context else []),
                    *([{"role": "system", "content": sentiment_context}] if sentiment_context else []),
                    *state.messages[-6:],
                ],
            )
        except Exception as exc:
            raise LLMUnavailableError(f"Chat LLM failed: {exc}") from exc

        if DISCLAIMER.lower() not in reply.lower():
            reply = f"{reply.rstrip()}\n\n_{DISCLAIMER}_"
        state.messages.append({"role": "assistant", "content": reply})
        return AssistantReply(message=reply, sentiment=sentiment_results)

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

        # Primary sort: match score DESC (best fit to the user's goal first).
        # Tiebreaker: historical return DESC. When the user gave no explicit
        # return/risk targets, every match score is equal — without a tiebreaker
        # the stable sort would fall back to the recommender's own risk-score
        # order (highest risk first), which surfaces the WORST picks at the top.
        # Highest-return-first is the intuitive "best pick" for an un-constrained
        # user, and respects match score when they DID set constraints.
        scored = sorted(
            [(s, score_match(s, goal)) for s in result.recommendations],
            key=lambda p: (-p[1], -p[0].features.get("returns", 0.0)),
        )
        top = scored[: goal.max_recommendations or 5]
        no_perfect_match = not any(score >= 80.0 for _, score in top)

        # Best-effort: pull marketaux sentiment for the shortlist so the LLM
        # can mention how the news currently reads on each pick. Falls back
        # silently when marketaux is unavailable (tests, missing key, 429).
        sentiment_by_ticker = _fetch_sentiment_for_shortlist(top)

        try:
            explanation = self._explain(goal, top, no_perfect_match, sentiment_by_ticker)
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
        sentiment_by_ticker: dict[str, SentimentResult] | None = None,
    ) -> str:
        sentiment_by_ticker = sentiment_by_ticker or {}

        def _sentiment_fragment(ticker: str) -> str:
            s = sentiment_by_ticker.get(ticker)
            if not s or s.article_count == 0:
                return ""
            return f", news {s.label.lower()} ({s.avg_score:+.2f} across {s.article_count} article{'s' if s.article_count != 1 else ''})"

        shortlist = "\n".join(
            f"- {s.ticker}: vol={s.features.get('volatility', 0) * 100:.1f}%, "
            f"return={s.features.get('returns', 0) * 100:.1f}%, match={score:.0f}"
            f"{_sentiment_fragment(s.ticker)}"
            for s, score in scored
        )
        warning = (
            "\nIMPORTANT: none of these stocks meet the user's constraints closely; "
            "show them as CLOSEST matches and note the mismatch.\n"
            if no_perfect_match
            else ""
        )
        sentiment_hint = (
            "\nWhen a stock has news sentiment info, briefly mention whether the "
            "recent news reads positive or negative — it's a useful tie-breaker.\n"
            if any(s.article_count > 0 for s in sentiment_by_ticker.values())
            else ""
        )
        user_msg = (
            f"Goal: target return {goal.target_annual_return_pct}%, "
            f"max volatility {goal.max_volatility_pct}%, "
            f"sectors {goal.sectors}, top {goal.max_recommendations}.\n"
            f"Shortlist:\n{shortlist}{warning}{sentiment_hint}"
        )
        return self._llm(_EXPLAIN_SYSTEM, [{"role": "user", "content": user_msg}])

    def _record(self, state: AssistantState, message: str) -> AssistantReply:
        state.messages.append({"role": "assistant", "content": message})
        return AssistantReply(message=message)
