"""Multi-agent financial advisor graph package.

This package encapsulates the domain state (Pydantic v2 schemas) and the modular
agent nodes for the multi-agent financial-advisor system.

The LangGraph :class:`~langgraph.graph.StateGraph` that wires these nodes
together is defined and compiled in the top-level ``graph.py`` module (kept
outside this package to avoid the name collision between the ``graph`` package
and the ``graph.py`` module, and to keep the package free of import-time side
effects). ``langgraph.json`` references that compiled graph as
``./graph.py:graph``.

Typical usage::

    from graph import AgentState, PortfolioItem, parse_portfolio_input
    from graph.nodes import supervisor_node, rag_node
"""

from . import nodes, state
from .nodes import (
    market_data_node,
    news_node,
    optimizer_node,
    rag_node,
    route_supervisor,
    supervisor_node,
)
from .state import (
    AgentState,
    FINISH,
    MARKET_DATA,
    NEWS,
    OPTIMIZER,
    RAG,
    SUPERVISOR,
    WORKER_ROUTES,
    WorkerRoute,
    PlanStep,
    PortfolioConstraints,
    PortfolioInput,
    PortfolioItem,
    TEST_CONSTRAINTS,
    TEST_PORTFOLIO,
    TEST_PORTFOLIO_INPUT,
    TEST_QUERIES,
    parse_portfolio_input,
)

__all__ = [
    # State schemas
    "AgentState",
    "PortfolioItem",
    "PortfolioConstraints",
    "PortfolioInput",
    "PlanStep",
    # Route vocabulary
    "SUPERVISOR",
    "MARKET_DATA",
    "RAG",
    "NEWS",
    "OPTIMIZER",
    "FINISH",
    "WORKER_ROUTES",
    "WorkerRoute",
    # Helpers
    "parse_portfolio_input",
    # Sample data extracted from requirements.pdf
    "TEST_PORTFOLIO",
    "TEST_CONSTRAINTS",
    "TEST_PORTFOLIO_INPUT",
    "TEST_QUERIES",
    # Nodes
    "supervisor_node",
    "route_supervisor",
    "market_data_node",
    "rag_node",
    "news_node",
    "optimizer_node",
]
