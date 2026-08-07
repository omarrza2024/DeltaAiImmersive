"""Pure-Python scoring of stocks against an InvestmentGoal.

The LLM never computes match scores — this module does. Two components,
each in [0, 1], averaged and rescaled to 0–100:

  * volatility fit — 1.0 at or below the cap, decaying linearly above it
  * return fit     — 1.0 at or above the target, decaying linearly below it
"""

from __future__ import annotations

from finance_dashboard.assistant.models import InvestmentGoal
from finance_dashboard.recommender.models import StockRecommendation


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
    """Score in [0, 100]. Returns 0 if required features are missing."""
    vol = stock.features.get("volatility")
    ret = stock.features.get("returns")
    if vol is None or ret is None:
        return 0.0
    if goal.max_volatility_pct is None or goal.target_annual_return_pct is None:
        return 0.0

    vol_fit = _volatility_fit(vol * 100.0, goal.max_volatility_pct)
    ret_fit = _return_fit(ret * 100.0, goal.target_annual_return_pct)
    return (vol_fit + ret_fit) / 2.0 * 100.0
