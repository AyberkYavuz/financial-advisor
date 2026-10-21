"""Pydantic v2 state schemas for the multi-agent financial advisor.

This module is the single source of truth for the domain model:

* portfolio holdings (:class:`PortfolioItem`)
* investment constraints (:class:`PortfolioConstraints`)
* the full input payload (:class:`PortfolioInput`)
* the shared state threaded through every graph node (:class:`AgentState`)

It also captures the sample portfolio, constraints, and test queries extracted
from ``requirements.pdf`` so the graph can be exercised end-to-end.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PortfolioItem(BaseModel):
    """A single position within a portfolio."""

    ticker: str = Field(..., min_length=1, description="Stock ticker symbol (e.g. AAPL).")
    shares: int = Field(..., ge=0, description="Number of shares held.")


class PortfolioConstraints(BaseModel):
    """Investment constraints applied to a portfolio.

    Field names use Pythonic ``snake_case`` but accept the ``camelCase`` JSON keys
    from ``requirements.pdf`` (``maxRisk`` / ``minDividendYield``) via aliases.
    """

    model_config = ConfigDict(populate_by_name=True)

    max_risk: Literal["low", "medium", "high"] = Field(
        default="medium",
        alias="maxRisk",
        description="Maximum acceptable risk level.",
    )
    min_dividend_yield: float = Field(
        default=0.0,
        alias="minDividendYield",
        ge=0,
        description="Minimum dividend yield, in percent.",
    )
    sector: str = Field(
        default="",
        description="Preferred sector focus (e.g. 'technology').",
    )


class PortfolioInput(BaseModel):
    """Complete portfolio input payload: holdings plus constraints."""

    portfolio: list[PortfolioItem] = Field(default_factory=list)
    constraints: PortfolioConstraints = Field(default_factory=PortfolioConstraints)


class AgentState(BaseModel):
    """Shared state threaded through every node in the graph."""

    # --- User input -----------------------------------------------------
    query: str = Field(default="", description="The user's natural-language query.")
    portfolio: list[PortfolioItem] = Field(default_factory=list)
    constraints: PortfolioConstraints = Field(default_factory=PortfolioConstraints)

    # --- Supervisor routing ---------------------------------------------
    next_node: str = Field(
        default="",
        description="Target node selected by the supervisor for the current turn.",
    )

    # --- Agent outputs --------------------------------------------------
    market_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Real-time market data from the Yahoo Finance MCP.",
    )
    rag_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Documents retrieved by the RAG pipeline.",
    )
    news_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Latest news from the Web Search MCP.",
    )
    optimization_result: dict[str, Any] = Field(
        default_factory=dict,
        description="Output of the portfolio optimizer.",
    )

    # --- Final answer ---------------------------------------------------
    final_response: str = Field(default="", description="Final response returned to the user.")


def parse_portfolio_input(data: dict[str, Any]) -> PortfolioInput:
    """Parse a JSON-style portfolio payload into a :class:`PortfolioInput`.

    Accepts the exact shape from ``requirements.pdf``::

        {
          "portfolio": [
            {"ticker": "AAPL", "shares": 100},
            {"ticker": "MSFT", "shares": 50},
            {"ticker": "GOOGL", "shares": 25}
          ],
          "constraints": {
            "maxRisk": "medium",
            "minDividendYield": 2.0,
            "sector": "technology"
          }
        }
    """
    return PortfolioInput.model_validate(data)


# ---------------------------------------------------------------------------
# Sample data extracted from requirements.pdf
# ---------------------------------------------------------------------------

#: Test portfolio: AAPL (100 shares), MSFT (50 shares), GOOGL (25 shares).
TEST_PORTFOLIO: list[PortfolioItem] = [
    PortfolioItem(ticker="AAPL", shares=100),
    PortfolioItem(ticker="MSFT", shares=50),
    PortfolioItem(ticker="GOOGL", shares=25),
]

#: Test constraints: maxRisk "medium", minDividendYield 2.0, sector "technology".
TEST_CONSTRAINTS: PortfolioConstraints = PortfolioConstraints(
    max_risk="medium",
    min_dividend_yield=2.0,
    sector="technology",
)

#: Full test input (portfolio + constraints).
TEST_PORTFOLIO_INPUT: PortfolioInput = PortfolioInput(
    portfolio=TEST_PORTFOLIO,
    constraints=TEST_CONSTRAINTS,
)

#: Test queries extracted from requirements.pdf.
TEST_QUERIES: list[str] = [
    "How has AAPL's financial performance been over the last 5 years?",
    "What are MSFT's risk factors?",
    "What trends are being seen in the Tech sector?",
    "Optimize this portfolio",
    "What do the latest news say about the portfolio?",
]


__all__ = [
    "AgentState",
    "PortfolioItem",
    "PortfolioConstraints",
    "PortfolioInput",
    "parse_portfolio_input",
    "TEST_PORTFOLIO",
    "TEST_CONSTRAINTS",
    "TEST_PORTFOLIO_INPUT",
    "TEST_QUERIES",
]
