"""End-to-end integration tests for the supervisor + worker nodes.

Runs the full compiled graph offline: the LLM is replaced with the
deterministic keyword router (``get_llm -> None``) and the yfinance-backed
fetchers are monkeypatched, so no network or API keys are required.

These tests verify that the supervisor routes to the correct worker and that
the worker's result lands in the shared state.
"""

from __future__ import annotations

from typing import Any

import graph.nodes.market_data as md
import graph.nodes.news as nd
import graph.nodes.supervisor as sup
from graph import (
    FINISH,
    MARKET_DATA,
    NEWS,
    TEST_CONSTRAINTS,
    TEST_PORTFOLIO,
)


def _invoke(compiled_graph, monkeypatch, tmp_path, query: str) -> dict[str, Any]:
    """Run the graph with a deterministic keyword-router supervisor."""
    monkeypatch.setattr(sup, "PLAN_FILE", tmp_path / "plan.md")
    monkeypatch.setattr(sup, "get_llm", lambda: None)
    return compiled_graph.invoke(
        {
            "query": query,
            "portfolio": TEST_PORTFOLIO,
            "constraints": TEST_CONSTRAINTS,
        }
    )


def test_supervisor_routes_to_market_data(compiled_graph, monkeypatch, tmp_path):
    monkeypatch.setattr(
        md, "get_stock_info", lambda t: {"ticker": t, "current_price": 100.0}
    )
    monkeypatch.setattr(
        md,
        "get_historical_stock_prices",
        lambda t, period="1mo", interval="1d": [{"date": "2024-01-02", "close": 99.0}],
    )

    result = _invoke(
        compiled_graph, monkeypatch, tmp_path, "What is the current price of AAPL?"
    )

    assert result["next_node"] == FINISH
    assert result["plan"][0].route == MARKET_DATA
    assert result["market_data"]["tickers"] == ["AAPL", "MSFT", "GOOGL"]
    assert set(result["market_data"]["data"]) == {"AAPL", "MSFT", "GOOGL"}
    assert result["market_data"]["data"]["AAPL"]["info"]["current_price"] == 100.0


def test_supervisor_routes_to_news(compiled_graph, monkeypatch, tmp_path):
    monkeypatch.setattr(
        nd,
        "get_yahoo_finance_news",
        lambda t, limit=10: [
            {
                "ticker": t.upper(),
                "title": "Great news",
                "publisher": "YF",
                "link": "x",
                "published_at": None,
                "sentiment": "positive",
                "keywords": [],
            }
        ],
    )

    result = _invoke(
        compiled_graph,
        monkeypatch,
        tmp_path,
        "What do the latest news say about the portfolio?",
    )

    assert result["next_node"] == FINISH
    assert result["plan"][0].route == NEWS
    assert {r["ticker"] for r in result["news_results"]} == {"AAPL", "MSFT", "GOOGL"}
    assert result["news_results"][0]["articles"][0]["sentiment"] == "positive"
