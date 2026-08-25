"""Portfolio enrichment — current value, P&L, weights, sector allocation, LLM summary."""

from __future__ import annotations


def enrich(positions: list[dict], prices: dict[str, float], sectors: dict[str, str]) -> dict:
    """Compute market values, P&L, weights, and sector rollup for a portfolio."""
    enriched_positions: list[dict] = []
    for pos in positions:
        ticker = pos["ticker"]
        shares = pos["shares"]
        avg_cost = pos["avg_cost"]
        price = prices.get(ticker)
        cost_basis = shares * avg_cost

        if price is None:
            mv = 0.0
            pnl = -cost_basis      # position is unquotable — treat as unknown
            pnl_pct = 0.0
        else:
            mv = shares * price
            pnl = mv - cost_basis
            pnl_pct = (pnl / cost_basis * 100.0) if cost_basis else 0.0

        enriched_positions.append({
            "ticker": ticker,
            "shares": shares,
            "avg_cost": avg_cost,
            "current_price": price,
            "market_value": mv,
            "cost_basis": cost_basis,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "sector": sectors.get(ticker, "Unknown"),
        })

    total_value = sum(p["market_value"] for p in enriched_positions)
    total_cost = sum(p["cost_basis"] for p in enriched_positions)
    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100.0) if total_cost else 0.0

    # Weights and sector allocation are only meaningful when there's value.
    for p in enriched_positions:
        p["weight_pct"] = (p["market_value"] / total_value * 100.0) if total_value else 0.0

    sector_allocation: dict[str, float] = {}
    if total_value:
        for p in enriched_positions:
            sector_allocation[p["sector"]] = sector_allocation.get(p["sector"], 0.0) + p["market_value"]
        sector_allocation = {s: (v / total_value * 100.0) for s, v in sector_allocation.items()}

    return {
        "positions": enriched_positions,
        "total_value": total_value,
        "total_cost": total_cost,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "sector_allocation": sector_allocation,
    }


def summary_for_llm(enriched: dict) -> str:
    """Compact plain-text portfolio summary for the AI assistant's system context."""
    positions = enriched["positions"]
    if not positions:
        return "[Portfolio is empty — no positions yet.]"

    lines = [
        f"User's portfolio: {len(positions)} position(s), "
        f"total value ${enriched['total_value']:,.0f} "
        f"({enriched['total_pnl_pct']:+.1f}% unrealized)."
    ]
    for p in positions:
        lines.append(
            f"- {p['ticker']} ({p['sector']}): {p['shares']:g} shares @ ${p['avg_cost']:.2f} avg, "
            f"now ${p['current_price']:.2f} ({p['pnl_pct']:+.1f}%), {p['weight_pct']:.0f}% of portfolio"
            if p["current_price"] is not None
            else f"- {p['ticker']}: {p['shares']:g} shares @ ${p['avg_cost']:.2f} — current price unavailable"
        )
    if enriched["sector_allocation"]:
        alloc = ", ".join(f"{s} {pct:.0f}%" for s, pct in sorted(
            enriched["sector_allocation"].items(), key=lambda kv: -kv[1]
        ))
        lines.append(f"Sector allocation: {alloc}")
    return "\n".join(lines)
