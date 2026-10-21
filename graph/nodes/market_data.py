"""Market data node (stub).

The production implementation will call the Yahoo Finance MCP server to fetch
real-time quotes, historical prices, and fundamentals for each ticker in the
portfolio.
"""

from __future__ import annotations

from typing import Any

from ..state import AgentState


def market_data_node(state: AgentState) -> dict[str, Any]:
    """Fetch real-time market data for the portfolio tickers (stub)."""
    tickers = [item.ticker for item in state.portfolio]

    # TODO: Replace with a real Yahoo Finance MCP client call.
    market_data: dict[str, Any] = {
        "stub": True,
        "source": "yahoo_finance_mcp",
        "tickers": tickers,
        "message": "Market data stub — Yahoo Finance MCP integration pending.",
    }

    return {
        "market_data": market_data,
        "final_response": (
            f"[market_data stub] Retrieved market data for: "
            f"{', '.join(tickers) or 'no tickers'}."
        ),
    }


__all__ = ["market_data_node"]
