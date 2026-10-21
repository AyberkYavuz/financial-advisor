"""News node (stub).

The production implementation will call the Web Search MCP to fetch the latest
news and market analysis for the portfolio and/or the user query.
"""

from __future__ import annotations

from typing import Any

from ..state import AgentState


def news_node(state: AgentState) -> dict[str, Any]:
    """Fetch the latest news and market analysis (stub)."""
    tickers = [item.ticker for item in state.portfolio]

    # TODO: Replace with a real Web Search MCP client call.
    news_results: list[dict[str, Any]] = [
        {
            "stub": True,
            "source": "web_search_mcp",
            "tickers": tickers,
            "query": state.query,
        }
    ]

    return {
        "news_results": news_results,
        "final_response": (
            f"[news stub] Fetched latest news for: "
            f"{', '.join(tickers) or state.query or 'the portfolio'}."
        ),
    }


__all__ = ["news_node"]
