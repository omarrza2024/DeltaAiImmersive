"""Pure-Python scoring of stocks against an InvestmentGoal.

The LLM never computes match scores — this module does. Two components,
each in [0, 1], averaged and rescaled to 0–100:

  * volatility fit — 1.0 at or below the cap, decaying linearly above it
  * return fit     — 1.0 at or above the target, decaying linearly below it
"""

from __future__ import annotations

from finance_dashboard.assistant.models import InvestmentGoal
from finance_dashboard.recommender.models import RiskLabel, StockRecommendation

# Which cluster-assigned RiskLabels satisfy each user-stated preference word.
_PREFERENCE_MATCH: dict[str, set[RiskLabel]] = {
    "low":    {RiskLabel.LOW},
    "medium": {RiskLabel.MEDIUM},
    "high":   {RiskLabel.HIGH_POSITIVE, RiskLabel.HIGH_NEGATIVE},
}


def _volatility_fit(vol_pct: float, cap_pct: float) -> float:
    if vol_pct <= cap_pct:
        return 1.0
    # Full penalty once volatility is 2x the cap.
    excess = (vol_pct - cap_pct) / cap_pct
    return max(0.0, 1.0 - excess)


def _return_fit(ret_pct: float, target_pct: float) -> float:
    if ret_pct >= target_pct:
        return 1.0
    # Full penalty once return is `target` percentage points short.
    shortfall = target_pct - ret_pct
    return max(0.0, 1.0 - shortfall / max(target_pct, 1e-9))


def score_match(stock: StockRecommendation, goal: InvestmentGoal) -> float:
    """Score in [0, 100]. Returns 0 if required features are missing.

    When the user didn't specify a return target or volatility cap, that
    dimension is skipped (treated as neutral) rather than failing — this
    supports fundamentals-driven requests where the user doesn't care about
    historical numbers.
    """
    vol = stock.features.get("volatility")
    ret = stock.features.get("returns")
    if vol is None or ret is None:
        return 0.0

    fits: list[float] = []
    if goal.max_volatility_pct is not None:
        fits.append(_volatility_fit(vol * 100.0, goal.max_volatility_pct))
    if goal.target_annual_return_pct is not None:
        fits.append(_return_fit(ret * 100.0, goal.target_annual_return_pct))
    # No preferences at all → neutral score; the recommender's own ordering
    # (by risk_score) still breaks ties.
    base = sum(fits) / len(fits) if fits else 1.0

    # Risk-preference alignment: if the user said "moderate", strongly reward
    # stocks that got labeled Medium Risk (vs. just anything under the cap).
    if goal.risk_preference is not None:
        accepted = _PREFERENCE_MATCH.get(goal.risk_preference, set())
        # 0.7× penalty when the label doesn't match — keeps things scored but
        # ensures matching-label stocks always rank above non-matching ones.
        if stock.risk_label not in accepted:
            base *= 0.7

    return base * 100.0
