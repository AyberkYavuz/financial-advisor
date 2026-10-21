"""RAG node — hybrid retrieval + cross-encoder re-ranking.

Queries the local ChromaDB instance (populated by ``init_chroma.py``) with an
OpenAI-embedded user query, retrieves a wide candidate pool via vector
similarity, re-scores it with a keyword/ticker hybrid signal, optionally
re-ranks the top candidates with a cross-encoder, and stores the final selected
chunks in ``AgentState.retrieved_docs``.

Every stage degrades gracefully: missing ChromaDB, missing API key, or an
unavailable cross-encoder all fall back to a safe (possibly empty) payload so
graph execution never crashes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from ..state import AgentState

logger = logging.getLogger(__name__)

#: Project root (``graph/nodes/rag.py`` -> root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load the root ``.env`` so ``OPENAI_API_KEY`` is available outside of
# ``langgraph dev`` (e.g. unit tests or scripts).
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover - python-dotenv is a project dependency
    pass

#: Local ChromaDB directory and collection (populated by ``init_chroma.py``).
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "financial_reports"
EMBEDDING_MODEL = "text-embedding-3-small"

#: Cross-encoder model used for re-ranking (loaded lazily, optional).
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

#: Retrieval pipeline sizing.
VECTOR_POOL_SIZE = 40   # wide candidate pool from vector search
HYBRID_POOL_SIZE = 20   # narrowed pool before cross-encoder re-ranking
TOP_K = 5               # final number of chunks returned

#: Weight of the keyword/ticker signal in the hybrid score.
KEYWORD_WEIGHT = 0.15


# ---------------------------------------------------------------------------
# Lazy, cached clients (OpenAI + cross-encoder)
# ---------------------------------------------------------------------------
_openai_client: Any | None = None
_openai_failed = False

_cross_encoder: Any | None = None
_cross_encoder_failed = False


def _get_openai_client() -> Any | None:
    """Return a cached OpenAI client, or ``None`` if no API key is available."""
    global _openai_client, _openai_failed
    if _openai_client is not None:
        return _openai_client
    if _openai_failed:
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY is not set; RAG retrieval unavailable.")
        _openai_failed = True
        return None

    try:
        from openai import OpenAI

        _openai_client = OpenAI(api_key=api_key)
        return _openai_client
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        logger.warning("Failed to create OpenAI client (%s).", exc)
        _openai_failed = True
        return None


def _get_cross_encoder() -> Any | None:
    """Return a cached cross-encoder, or ``None`` if it cannot be loaded."""
    global _cross_encoder, _cross_encoder_failed
    if _cross_encoder is not None:
        return _cross_encoder
    if _cross_encoder_failed:
        return None

    try:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder model '%s' ...", CROSS_ENCODER_MODEL)
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        return _cross_encoder
    except Exception as exc:  # noqa: BLE001 - graceful fallback to hybrid ranking
        logger.warning(
            "Cross-encoder unavailable (%s); using hybrid ranking only.", exc
        )
        _cross_encoder_failed = True
        return None


def _get_chroma_collection() -> Any | None:
    """Return the ChromaDB collection, or ``None`` if unavailable."""
    try:
        import chromadb
        from chromadb.config import Settings

        if not CHROMA_DIR.exists():
            logger.warning("ChromaDB directory not found at %s", CHROMA_DIR)
            return None

        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        return client.get_or_create_collection(name=COLLECTION_NAME)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        logger.warning("Failed to connect to ChromaDB (%s).", exc)
        return None


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def _embed_query(client: Any, query: str) -> list[float]:
    """Embed a query with OpenAI's embedding model."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Hybrid keyword scoring
# ---------------------------------------------------------------------------
def _keyword_score(
    query: str,
    tickers: list[str],
    text: str,
    metadata: dict[str, Any],
) -> float:
    """Score a chunk by ticker/keyword overlap with the query.

    * +1.0 for each portfolio ticker mentioned in the chunk text.
    * +0.5 for each portfolio ticker appearing in the source file name.

    This boosts chunks that are textually about the tickers a user is asking
    about, complementing the pure vector-similarity signal.
    """
    score = 0.0
    text_lower = (text or "").lower()
    source_lower = (metadata.get("source") or "").lower()

    for ticker in tickers:
        token = ticker.lower()
        if token in text_lower:
            score += 1.0
        if token in source_lower:
            score += 0.5
    return score


# ---------------------------------------------------------------------------
# Cross-encoder re-ranking
# ---------------------------------------------------------------------------
def _rerank(
    query: str,
    candidates: list[dict[str, Any]],
    cross_encoder: Any | None,
) -> list[dict[str, Any]]:
    """Re-rank candidates with a cross-encoder (or keep order if unavailable)."""
    if cross_encoder is None:
        return candidates

    pairs = [(query, candidate["text"]) for candidate in candidates]
    scores = cross_encoder.predict(pairs)
    for candidate, score in zip(candidates, scores):
        candidate["cross_encoder_score"] = float(score)
    candidates.sort(key=lambda c: c["cross_encoder_score"], reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------
def _fallback_result(reason: str) -> dict[str, Any]:
    logger.warning("Rag: falling back (%s); returning no documents.", reason)
    return {
        "retrieved_docs": [],
        "final_response": (
            f"[rag] Retrieval unavailable ({reason}); no documents returned."
        ),
    }


# ---------------------------------------------------------------------------
# Graph node
# ---------------------------------------------------------------------------
def rag_node(state: AgentState) -> dict[str, Any]:
    """Retrieve the most relevant financial-document chunks for the query."""
    logger.info("Rag node called")
    query = state.query
    logger.info("Rag query: %s", query)

    if not query:
        return _fallback_result("empty query")

    tickers = [item.ticker for item in state.portfolio]

    # 1. Embed the query ---------------------------------------------------
    client = _get_openai_client()
    if client is None:
        return _fallback_result("OpenAI client unavailable")

    logger.info("Rag: generating query embedding (model=%s)", EMBEDDING_MODEL)
    try:
        query_embedding = _embed_query(client, query)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        logger.warning("Rag: query embedding failed (%s).", exc)
        return _fallback_result("embedding failed")

    # 2. Vector retrieval --------------------------------------------------
    collection = _get_chroma_collection()
    if collection is None:
        return _fallback_result("ChromaDB unavailable")

    try:
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=VECTOR_POOL_SIZE,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        logger.warning("Rag: ChromaDB query failed (%s).", exc)
        return _fallback_result("ChromaDB query failed")

    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    logger.info("Rag: retrieved %d candidate chunk(s) from ChromaDB", len(ids))

    if not ids:
        return _fallback_result("no candidates returned")

    # 3. Hybrid scoring (vector similarity + keyword/ticker boost) ---------
    candidates: list[dict[str, Any]] = []
    for i in range(len(ids)):
        candidates.append(
            {
                "id": ids[i],
                "text": documents[i],
                "metadata": metadatas[i],
                "distance": distances[i],
            }
        )
    for candidate in candidates:
        candidate["keyword_score"] = _keyword_score(
            query, tickers, candidate["text"], candidate["metadata"]
        )
        # cosine distance: similarity = 1 - distance.
        candidate["hybrid_score"] = (1.0 - candidate["distance"]) + (
            KEYWORD_WEIGHT * candidate["keyword_score"]
        )
    candidates.sort(key=lambda c: c["hybrid_score"], reverse=True)
    hybrid_top = candidates[:HYBRID_POOL_SIZE]

    # 4. Cross-encoder re-ranking ------------------------------------------
    cross_encoder = _get_cross_encoder()
    if cross_encoder is None:
        logger.info("Rag: cross-encoder unavailable; keeping hybrid ranking")
    else:
        logger.info(
            "Rag: re-ranking %d candidate(s) with cross-encoder", len(hybrid_top)
        )
    hybrid_top = _rerank(query, hybrid_top, cross_encoder)
    top_k = hybrid_top[:TOP_K]

    # 5. Build the final result --------------------------------------------
    retrieved_docs = [
        {
            "id": candidate["id"],
            "source": candidate["metadata"].get("source"),
            "title": candidate["metadata"].get("title"),
            "page": candidate["metadata"].get("page"),
            "chunk_index": candidate["metadata"].get("chunk_index"),
            "text": candidate["text"],
            "distance": candidate["distance"],
            "keyword_score": candidate["keyword_score"],
            "cross_encoder_score": candidate.get("cross_encoder_score"),
        }
        for candidate in top_k
    ]

    logger.info("Rag: selected %d final context chunk(s)", len(retrieved_docs))
    for doc in retrieved_docs:
        logger.info(
            "  - %s (page %s, distance=%.4f, keyword=%.1f)",
            doc["source"],
            doc["page"],
            doc["distance"],
            doc["keyword_score"],
        )

    sources = {doc["source"] for doc in retrieved_docs}
    return {
        "retrieved_docs": retrieved_docs,
        "final_response": (
            f"[rag] Retrieved {len(retrieved_docs)} relevant document chunk(s) "
            f"from {len(sources)} source(s)."
        ),
    }


__all__ = ["rag_node"]
