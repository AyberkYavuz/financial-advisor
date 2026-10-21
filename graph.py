"""Main LangGraph ``StateGraph`` definition and compilation.

Wires the modular nodes from the :mod:`graph` package into a single multi-agent
financial-advisor workflow::

    START -> supervisor -> (market_data | rag | news | optimizer)
                 ^                    |
                 |                    v
                 +------ loop --------+
                        (or FINISH -> END)

The supervisor decomposes the user query into an ordered plan (written to
``plan.md``) and routes to one specialist at a time, looping back after each
sub-task to mark progress until the plan is complete.

The compiled graph is exposed as ``graph`` and is the entry point referenced by
``langgraph.json`` for ``langgraph dev``.
"""

from __future__ import annotations

from typing import Any

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
from graph.state import (
    FINISH,
    MARKET_DATA,
    NEWS,
    OPTIMIZER,
    RAG,
    SUPERVISOR,
    AgentState,
)

#: Specialist nodes the supervisor may route to.
_SPECIALISTS = (MARKET_DATA, RAG, NEWS, OPTIMIZER)


def build_graph(checkpointer: Any | None = None) -> CompiledStateGraph:
    """Assemble and compile the multi-agent financial advisor graph.

    Args:
        checkpointer: Optional checkpointer for standalone multi-turn use. Do
            NOT pass one when running under ``langgraph dev`` — LangGraph
            Studio/API provides its own persistence and rejects custom
            checkpointers.
    """
    builder = StateGraph(AgentState)

    # Register nodes.
    builder.add_node(SUPERVISOR, supervisor_node)
    builder.add_node(MARKET_DATA, market_data_node)
    builder.add_node(RAG, rag_node)
    builder.add_node(NEWS, news_node)
    builder.add_node(OPTIMIZER, optimizer_node)

    # Entry point.
    builder.add_edge(START, SUPERVISOR)

    # The supervisor routes to a specialist, or to FINISH once the plan is done.
    builder.add_conditional_edges(
        SUPERVISOR,
        route_supervisor,
        {
            MARKET_DATA: MARKET_DATA,
            RAG: RAG,
            NEWS: NEWS,
            OPTIMIZER: OPTIMIZER,
            FINISH: END,
        },
    )

    # Each specialist returns to the supervisor so the supervisor can mark the
    # completed sub-task in ``plan.md`` and route to the next one.
    for node in _SPECIALISTS:
        builder.add_edge(node, SUPERVISOR)

    return builder.compile(checkpointer=checkpointer)


#: Compiled graph — the object referenced by ``langgraph.json``.
#: No checkpointer is attached: LangGraph Studio/API handles persistence itself.
graph = build_graph()


__all__ = [
    "graph",
    "build_graph",
    "SUPERVISOR",
    "MARKET_DATA",
    "RAG",
    "NEWS",
    "OPTIMIZER",
    "FINISH",
]
