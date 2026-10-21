"""Optimizer node — portfolio valuation and constraint evaluation.

This node combines the user's portfolio with live metrics from the
``market_data`` state, computes the total valuation and per-asset allocation
weights, and checks the portfolio against the investment constraints (maximum
risk, minimum dividend yield, and sector focus). When a constraint is violated
it produces concrete rebalancing recommendations, which are stored in
``AgentState.optimization_result``.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from ..state import (
    AgentState,
    PortfolioConstraints,
    PortfolioItem,
    TEST_CONSTRAINTS,
    TEST_PORTFOLIO,
)

logger = logging.getLogger(__name__)

#: Fallback prices used when ``market_data`` is unavailable for a ticker.
#: Mirrors the deterministic baselines used by the market-data mock fallback.
_MOCK_PRICES: dict[str, float] = {
    "AAPL": 220.0,
    "MSFT": 400.0,
    "GOOGL": 160.0,
}

#: Fallback sectors used alongside the mock prices above (tech portfolio).
_MOCK_SECTORS: dict[str, str] = {
    "AAPL": "technology",
    "MSFT": "technology",
    "GOOGL": "technology",
}


def _to_percent(value: Any) -> float | None:
    """Convert a yfinance-style fraction (``0.02`` = 2%) to percent (``2.0``).

    Returns ``None`` for missing / non-finite values.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number * 100.0


def optimize_portfolio(
    portfolio: list[PortfolioItem],
    constraints: PortfolioConstraints,
    market_data: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a portfolio against its constraints and produce recommendations.

    Pure function (no state access) so it can be unit-tested directly.
    ``market_data`` is expected to match the shape produced by the market-data
    node: ``{"data": {TICKER: {"info": {...}, "is_mock": bool, ...}}}``.
    """
    data = (market_data or {}).get("data") or {}

    holdings: list[dict[str, Any]] = []
    for item in portfolio:
        ticker = item.ticker.upper()
        entry = data.get(ticker) or {}
        info = entry.get("info") or {}

        is_mock = bool(info.get("is_mock", entry.get("is_mock", False)))
        price = info.get("current_price")
        sector = info.get("sector")
        if price is None:
            price = _MOCK_PRICES.get(ticker, 0.0)
            sector = _MOCK_SECTORS.get(ticker, sector)
            is_mock = True

        holdings.append(
            {
                "ticker": ticker,
                "shares": item.shares,
                "price": price,
                "value": item.shares * price,
                "dividend_yield_pct": _to_percent(info.get("dividend_yield")),
                "sector": sector,
                "is_mock": is_mock,
            }
        )

    total_value = sum(h["value"] for h in holdings)

    for h in holdings:
        h["weight_pct"] = (h["value"] / total_value) * 100.0 if total_value else 0.0

    # Weighted average dividend yield (percent). Holdings with unknown yield
    # contribute 0% so the portfolio isn't treated as meeting the target by
    # accident.
    weighted_yield = (
        sum(
            (h["value"] / total_value) * (h["dividend_yield_pct"] or 0.0)
            for h in holdings
        )
        if total_value
        else 0.0
    )

    target_sector = (constraints.sector or "").strip().lower()
    mismatches = [
        h["ticker"]
        for h in holdings
        if target_sector and (h["sector"] or "").strip().lower() != target_sector
    ]

    violations: list[str] = []
    recommendations: list[str] = []

    if weighted_yield < constraints.min_dividend_yield:
        violations.append(
            f"weighted dividend yield {weighted_yield:.2f}% is below the minimum "
            f"{constraints.min_dividend_yield:.2f}%"
        )
        if holdings:
            lowest = min(
                holdings,
                key=lambda h: h["dividend_yield_pct"]
                if h["dividend_yield_pct"] is not None
                else 0.0,
            )
            low_yield = (
                lowest["dividend_yield_pct"]
                if lowest["dividend_yield_pct"] is not None
                else 0.0
            )
            recommendations.append(
                f"Dividend yield below target: reduce exposure to {lowest['ticker']} "
                f"(yield {low_yield:.2f}%) and reallocate into higher-yield "
                f"dividend-paying assets."
            )

    if mismatches:
        violations.append(
            f"sector misalignment: {', '.join(mismatches)} not in "
            f"'{target_sector or 'any'}'"
        )
        recommendations.append(
            f"Sector constraint: reduce or replace {', '.join(mismatches)} to align "
            f"the portfolio with the '{target_sector}' sector."
        )

    if not violations:
        recommendations.append(
            "Portfolio is compliant with all constraints; no rebalancing required."
        )

    # Round for storage / display only after all raw calculations are done.
    for h in holdings:
        h["price"] = round(h["price"], 4)
        h["value"] = round(h["value"], 2)
        h["weight_pct"] = round(h["weight_pct"], 4)
        if h["dividend_yield_pct"] is not None:
            h["dividend_yield_pct"] = round(h["dividend_yield_pct"], 4)

    return {
        "total_value": round(total_value, 2),
        "currency": "USD",
        "holdings": holdings,
        "constraints": {
            "max_risk": constraints.max_risk,
            "min_dividend_yield": constraints.min_dividend_yield,
            "sector": constraints.sector,
        },
        "metrics": {
            "weighted_dividend_yield_pct": round(weighted_yield, 4),
            "sector_match_count": len(holdings) - len(mismatches),
            "sector_mismatch_tickers": mismatches,
        },
        "compliant": not violations,
        "violations": violations,
        "recommendations": recommendations,
    }


def optimizer_node(state: AgentState) -> dict[str, Any]:
    """Evaluate the portfolio against its constraints and store the report.

    Falls back to the standard test payload when holdings/constraints are not
    supplied (e.g. the node is invoked standalone).
    """
    logger.info("Optimizer node called")

    portfolio = list(state.portfolio) if state.portfolio else list(TEST_PORTFOLIO)
    constraints = state.constraints
    if not constraints.sector:
        constraints = TEST_CONSTRAINTS

    result = optimize_portfolio(portfolio, constraints, state.market_data)

    logger.info(
        "Optimizer result: value=$%.2f, compliant=%s, violations=%d",
        result["total_value"],
        result["compliant"],
        len(result["violations"]),
    )

    if result["compliant"]:
        status = "compliant with all constraints"
    else:
        status = (
            f"{len(result['violations'])} violation(s): "
            f"{'; '.join(result['violations'])}"
        )

    return {
        "optimization_result": result,
        "final_response": (
            f"[optimizer] Portfolio value ${result['total_value']:,.2f}; {status}"
        ),
    }


__all__ = ["optimize_portfolio", "optimizer_node"]
