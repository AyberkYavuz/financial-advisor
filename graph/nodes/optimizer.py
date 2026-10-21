"""Optimizer node (stub).

The production implementation will evaluate the portfolio against its
constraints (max risk, minimum dividend yield, sector focus) and propose an
optimized allocation.
"""

from __future__ import annotations

from typing import Any

from ..state import AgentState


def optimizer_node(state: AgentState) -> dict[str, Any]:
    """Evaluate the portfolio against its constraints (stub)."""
    constraints = state.constraints
    holdings = [{"ticker": item.ticker, "shares": item.shares} for item in state.portfolio]

    # TODO: Replace with a real constraint-evaluation / allocation optimizer.
    optimization_result: dict[str, Any] = {
        "stub": True,
        "constraints": {
            "max_risk": constraints.max_risk,
            "min_dividend_yield": constraints.min_dividend_yield,
            "sector": constraints.sector,
        },
        "holdings": holdings,
        "message": "Optimizer stub — constraint evaluation pending.",
    }

    return {
        "optimization_result": optimization_result,
        "final_response": "[optimizer stub] Evaluated portfolio against constraints.",
    }


__all__ = ["optimizer_node"]
