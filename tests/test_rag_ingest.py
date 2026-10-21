"""Verification for RAG Phase 1: ChromaDB ingestion.

Confirms that ``init_chroma.py`` wrote chunk embeddings into ChromaDB and that
the collection can be queried semantically.

Prerequisite: run ``python init_chroma.py`` from the project root first, so the
``./chroma_db`` collection is populated.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from init_chroma import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def collection():
    """Connect to the local persistent ChromaDB and return the collection."""
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def test_collection_has_chunks(collection):
    assert collection.count() > 0, (
        "Collection is empty. Run `python init_chroma.py` first to ingest the PDFs."
    )


def test_metadata_fields_present(collection):
    sample = collection.get(limit=1, include=["metadatas"])
    assert sample["metadatas"], "No metadata found in the collection."
    meta = sample["metadatas"][0]
    for field in ("source", "title", "page", "chunk_index"):
        assert field in meta, f"Missing metadata field: {field}"


def test_embeddings_are_stored(collection):
    sample = collection.get(limit=1, include=["embeddings"])
    embeddings = sample["embeddings"]
    assert embeddings is not None and len(embeddings) > 0, "No embeddings stored."
    assert len(embeddings[0]) > 0, "Embedding vector is empty."


def test_semantic_query_returns_results(collection):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY is not set; cannot embed the query."

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=["What are the risk factors discussed in the annual report?"],
    )
    query_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    assert results["ids"], "Semantic query returned no results."
    assert len(results["ids"][0]) == 3
    assert results["documents"], "Query results missing document text."
    assert results["metadatas"], "Query results missing metadata."
