"""RAG node (stub).

The production implementation will embed financial reports (10-K / annual
reports / industry analyses) into a vector database (Chroma/Pinecone) and
retrieve the most relevant documents for the user query.
"""

from __future__ import annotations

from typing import Any

from ..state import AgentState


def rag_node(state: AgentState) -> dict[str, Any]:
    """Retrieve relevant documents from financial reports (stub)."""
    # TODO: Replace with a real embedding + vector-store retrieval pipeline.
    rag_results: list[dict[str, Any]] = [
        {
            "stub": True,
            "query": state.query,
            "document": "AAPL 10-K (2023) — placeholder retrieval result",
        }
    ]

    return {
        "rag_results": rag_results,
        "final_response": f"[rag stub] Retrieved documents for query: {state.query!r}",
    }


__all__ = ["rag_node"]
