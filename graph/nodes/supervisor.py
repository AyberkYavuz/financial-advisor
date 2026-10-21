"""Supervisor node: routes a user query to the most relevant specialist agent.

The production implementation should use an LLM (with structured output) to
classify the query. For this scaffold we use a deterministic, ordered keyword
router so the graph can be exercised end-to-end against the test queries from
``requirements.pdf`` without any external dependencies.
"""

from __future__ import annotations

from typing import Any

from ..state import AgentState

#: Ordered routing table. Order matters: the first table whose keyword matches
#: the (lower-cased) query wins, so more specific keywords come first.
ROUTING_TABLE: dict[str, tuple[str, ...]] = {
    "optimizer": ("optimize", "rebalance", "reallocate", "constraint", "allocation"),
    "news": ("news", "trend", "trends", "sector", "headline", "latest"),
    "rag": (
        "risk",
        "performance",
        "financial",
        "report",
        "10-k",
        "10k",
        "annual",
        "revenue",
        "earnings",
        "risk factor",
    ),
    "market_data": ("price", "quote", "market data", "current", "stock price", "ticker"),
}

#: Fallback node when no routing keyword matches.
DEFAULT_NODE = "rag"


def supervisor_node(state: AgentState) -> dict[str, Any]:
    """Classify ``state.query`` and record the next node to invoke.

    The result is stored in ``state.next_node`` and consumed by
    :func:`route_supervisor` as the conditional edge.
    """
    query = (state.query or "").lower()

    for node, keywords in ROUTING_TABLE.items():
        if any(keyword in query for keyword in keywords):
            return {"next_node": node}

    return {"next_node": DEFAULT_NODE}


def route_supervisor(state: AgentState) -> str:
    """Conditional-edge function returning the node selected by the supervisor."""
    return state.next_node


__all__ = ["ROUTING_TABLE", "DEFAULT_NODE", "supervisor_node", "route_supervisor"]
