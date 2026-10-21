"""Main LangGraph ``StateGraph`` definition and compilation.

Wires the modular nodes from the :mod:`graph` package into a single multi-agent
financial-advisor workflow::

    START -> supervisor -> (market_data | rag | news | optimizer) -> END

The compiled graph is exposed as ``graph`` and is the entry point referenced by
``langgraph.json`` for ``langgraph dev``.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from graph.nodes import (
    market_data_node,
    news_node,
    optimizer_node,
    rag_node,
    route_supervisor,
    supervisor_node,
)
from graph.state import AgentState

#: Node names (also used as the supervisor's routing targets).
SUPERVISOR = "supervisor"
MARKET_DATA = "market_data"
RAG = "rag"
NEWS = "news"
OPTIMIZER = "optimizer"

#: Specialist nodes the supervisor may route to.
_SPECIALISTS = (MARKET_DATA, RAG, NEWS, OPTIMIZER)


def build_graph() -> CompiledStateGraph:
    """Assemble and compile the multi-agent financial advisor graph."""
    builder = StateGraph(AgentState)

    # Register nodes.
    builder.add_node(SUPERVISOR, supervisor_node)
    builder.add_node(MARKET_DATA, market_data_node)
    builder.add_node(RAG, rag_node)
    builder.add_node(NEWS, news_node)
    builder.add_node(OPTIMIZER, optimizer_node)

    # Entry point.
    builder.add_edge(START, SUPERVISOR)

    # The supervisor conditionally routes to a specialist based on the query.
    builder.add_conditional_edges(
        SUPERVISOR,
        route_supervisor,
        {
            MARKET_DATA: MARKET_DATA,
            RAG: RAG,
            NEWS: NEWS,
            OPTIMIZER: OPTIMIZER,
        },
    )

    # Each specialist terminates the run for now. A future iteration can loop
    # back to the supervisor for multi-hop reasoning or add a synthesis node.
    for node in _SPECIALISTS:
        builder.add_edge(node, END)

    return builder.compile()


#: Compiled graph — the object referenced by ``langgraph.json``.
graph = build_graph()


__all__ = ["graph", "build_graph", "SUPERVISOR", "MARKET_DATA", "RAG", "NEWS", "OPTIMIZER"]
