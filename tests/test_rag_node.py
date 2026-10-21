"""Unit tests for the advanced RAG node (hybrid retrieval + re-ranking)."""

from __future__ import annotations

import pytest

from graph import AgentState
from graph.nodes import rag as rag_module
from graph.nodes.rag import _keyword_score, _rerank, rag_node


# ---------------------------------------------------------------------------
# Pure function: keyword scoring
# ---------------------------------------------------------------------------
def test_keyword_score_detects_ticker_in_text_and_source():
    score = _keyword_score(
        query="How is AAPL performing?",
        tickers=["AAPL", "MSFT"],
        text="Apple (AAPL) reported strong earnings this quarter.",
        metadata={"source": "AAPL 10-K (2023).pdf"},
    )
    # +1.0 (ticker in text) + 0.5 (ticker in source) = 1.5
    assert score == pytest.approx(1.5)


def test_keyword_score_zero_when_no_match():
    score = _keyword_score(
        query="What about GOOGL?",
        tickers=["AAPL", "MSFT"],
        text="The company operates in many markets.",
        metadata={"source": "2023_Annual_Report.pdf"},
    )
    assert score == 0.0


# ---------------------------------------------------------------------------
# Pure function: cross-encoder re-ranking
# ---------------------------------------------------------------------------
def test_rerank_sorts_by_cross_encoder_score():
    class FakeCrossEncoder:
        def predict(self, pairs):
            # Deterministic score: length of the document text.
            return [float(len(doc)) for _, doc in pairs]

    candidates = [
        {"text": "short", "cross_encoder_score": None},
        {"text": "medium text", "cross_encoder_score": None},
        {"text": "the longest text here", "cross_encoder_score": None},
    ]
    result = _rerank("query", candidates, FakeCrossEncoder())
    assert result[0]["text"] == "the longest text here"
    assert result[-1]["text"] == "short"
    assert result[0]["cross_encoder_score"] == pytest.approx(len("the longest text here"))


def test_rerank_fallback_preserves_order_when_no_encoder():
    candidates = [{"text": "a"}, {"text": "b"}]
    result = _rerank("query", candidates, None)
    assert result == candidates


def test_real_cross_encoder_reranking():
    """Verify the real cross-encoder (if installed) re-ranks correctly."""
    pytest.importorskip("sentence_transformers")
    from sentence_transformers import CrossEncoder

    encoder = CrossEncoder(rag_module.CROSS_ENCODER_MODEL)
    candidates = [
        {"text": "The company faces risks from competition and supply chain disruptions."},
        {"text": "The company reported record iPhone sales this quarter."},
    ]
    result = _rerank("What are Apple's risk factors?", candidates, encoder)
    assert result[0]["text"].startswith("The company faces risks")
    assert result[0]["cross_encoder_score"] > result[1]["cross_encoder_score"]


# ---------------------------------------------------------------------------
# Node fallback paths (offline, via monkeypatching)
# ---------------------------------------------------------------------------
def test_rag_node_fallback_on_missing_chroma(monkeypatch):
    monkeypatch.setattr(rag_module, "_get_openai_client", lambda: object())
    monkeypatch.setattr(rag_module, "_embed_query", lambda client, q: [0.0] * 1536)
    monkeypatch.setattr(rag_module, "_get_chroma_collection", lambda: None)

    state = AgentState(query="What are AAPL's risk factors?")
    out = rag_node(state)
    assert out["retrieved_docs"] == []
    assert "unavailable" in out["final_response"]


def test_rag_node_fallback_on_missing_api_key(monkeypatch):
    monkeypatch.setattr(rag_module, "_get_openai_client", lambda: None)

    state = AgentState(query="What are AAPL's risk factors?")
    out = rag_node(state)
    assert out["retrieved_docs"] == []


def test_rag_node_fallback_on_empty_query():
    state = AgentState(query="")
    out = rag_node(state)
    assert out["retrieved_docs"] == []


# ---------------------------------------------------------------------------
# Integration: real ChromaDB + OpenAI (cross-encoder disabled for speed)
# ---------------------------------------------------------------------------
def test_rag_node_retrieves_and_updates_state(monkeypatch):
    monkeypatch.setattr(rag_module, "_get_cross_encoder", lambda: None)

    state = AgentState(
        query="What are Apple's risk factors?",
        portfolio=[{"ticker": "AAPL", "shares": 100}],
    )
    out = rag_node(state)

    docs = out["retrieved_docs"]
    assert isinstance(docs, list)
    assert len(docs) > 0, (
        "Expected retrieved documents. Run `python init_chroma.py` first to "
        "populate the ChromaDB collection."
    )
    assert len(docs) <= rag_module.TOP_K

    first = docs[0]
    for field in ("id", "source", "title", "page", "text", "distance"):
        assert field in first, f"Missing field {field} in retrieved doc."
    assert "risk" in first["text"].lower() or "Risk" in first["text"]
