"""ChromaDB initialization + PDF ingestion pipeline (RAG Phase 1).

Scans ``rag_files/`` for PDF documents, extracts text page-by-page, splits it
into overlapping chunks (500 characters with 50-character overlap) using a
recursive character text splitter, embeds every chunk with OpenAI's
``text-embedding-3-small`` model, and stores the chunks, their vector
embeddings, and rich metadata into a local persistent ChromaDB at
``./chroma_db``.

Run from the project root::

    python init_chroma.py

The script is idempotent: re-running it clears the collection and re-ingests
all PDFs from scratch.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RAG_FILES_DIR = PROJECT_ROOT / "rag_files"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "financial_reports"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_BATCH_SIZE = 100

#: Package name (for ``pip install``) keyed by the module import name.
REQUIRED_PACKAGES: dict[str, str] = {
    "chromadb": "chromadb",
    "openai": "openai",
    "pypdf": "pypdf",
    "dotenv": "python-dotenv",
    "langchain_text_splitters": "langchain-text-splitters",
}

LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"

logger = logging.getLogger("rag_ingest")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging() -> None:
    """Configure the root logger for clear console output at INFO level."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Remove any pre-existing handlers so re-runs / embedding don't double-log.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(handler)

    # Keep noisy third-party loggers out of the way.
    for name in ("chromadb", "httpx", "openai", "urllib3", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Dependency management
# ---------------------------------------------------------------------------
def _is_missing(import_name: str) -> bool:
    try:
        __import__(import_name)
        return False
    except ImportError:
        return True


def ensure_dependencies() -> None:
    """Install any required packages that are not importable yet."""
    missing = [pkg for mod, pkg in REQUIRED_PACKAGES.items() if _is_missing(mod)]
    if not missing:
        logger.info("All required packages are already installed.")
        return
    logger.info("Installing missing packages: %s", ", ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    logger.info("Package installation complete.")


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------
def extract_text_by_page(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract text from each page of ``pdf_path`` as ``(page_number, text)``."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        pages.append((page_number, page.extract_text() or ""))
    return pages


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_document(pdf_path: Path, pages: list[tuple[int, str]]) -> list[dict]:
    """Split a document into overlapping chunks with per-page metadata.

    Returns a list of dicts, each with keys ``text``, ``source``, ``title``,
    ``page``, and ``chunk_index``.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    title = pdf_path.stem
    chunks: list[dict] = []
    chunk_index = 0
    for page_number, text in pages:
        if not text.strip():
            continue
        for piece in splitter.split_text(text):
            chunks.append(
                {
                    "text": piece,
                    "source": pdf_path.name,
                    "title": title,
                    "page": page_number,
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1
    return chunks


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def embed_texts(client, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with OpenAI and return their vectors."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def main() -> int:
    setup_logging()
    logger.info("=== RAG ingestion pipeline (ChromaDB) ===")
    logger.info("Project root: %s", PROJECT_ROOT)

    # 1. Dependencies ------------------------------------------------------
    ensure_dependencies()

    # 2. Load environment + API key ---------------------------------------
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY is not set in %s", PROJECT_ROOT / ".env")
        logger.error("Add it to .env and re-run:  OPENAI_API_KEY=sk-...")
        return 1
    logger.info("OPENAI_API_KEY loaded from .env (masked: ...%s)", api_key[-4:])

    # 3. Scan for PDFs ----------------------------------------------------
    pdf_paths = sorted(RAG_FILES_DIR.glob("*.pdf"))
    if not pdf_paths:
        logger.error("No PDF files found in %s", RAG_FILES_DIR)
        return 1
    logger.info("Found %d PDF file(s) in %s:", len(pdf_paths), RAG_FILES_DIR)
    for path in pdf_paths:
        logger.info("  - %s", path.name)

    # 4. Extract + chunk --------------------------------------------------
    all_chunks: list[dict] = []
    for pdf_path in pdf_paths:
        logger.info("Extracting text from: %s", pdf_path.name)
        pages = extract_text_by_page(pdf_path)
        logger.info("  Extracted %d page(s)", len(pages))
        chunks = chunk_document(pdf_path, pages)
        all_chunks.extend(chunks)
        logger.info(
            "  Generated %d chunks for %s (size=%d, overlap=%d)",
            len(chunks),
            pdf_path.stem,
            CHUNK_SIZE,
            CHUNK_OVERLAP,
        )

    logger.info("Total chunks across all documents: %d", len(all_chunks))
    if not all_chunks:
        logger.error("No text chunks were produced; nothing to ingest.")
        return 1

    # 5. Generate embeddings ----------------------------------------------
    logger.info("Generating embeddings with OpenAI model '%s' ...", EMBEDDING_MODEL)
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        embeddings.extend(embed_texts(client, batch))
        logger.info(
            "  Embedded chunks %d-%d of %d",
            start + 1,
            min(start + EMBEDDING_BATCH_SIZE, total),
            total,
        )
    logger.info("Embedding generation complete (%d vectors).", len(embeddings))

    # 6. Store in ChromaDB -------------------------------------------------
    logger.info("Storing vectors in ChromaDB at %s ...", CHROMA_DIR)
    import chromadb
    from chromadb.config import Settings

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Clear any prior run so re-ingestion is idempotent.
    existing_ids = collection.get()["ids"]
    if existing_ids:
        logger.info("  Clearing %d existing record(s) from collection.", len(existing_ids))
        collection.delete(ids=existing_ids)

    ids = [f"{chunk['source']}#{chunk['chunk_index']}" for chunk in all_chunks]
    metadatas = [
        {
            "source": chunk["source"],
            "title": chunk["title"],
            "page": chunk["page"],
            "chunk_index": chunk["chunk_index"],
        }
        for chunk in all_chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    logger.info(
        "Inserted %d chunk(s) into collection '%s'.",
        collection.count(),
        COLLECTION_NAME,
    )

    logger.info("=== RAG ingestion complete ===")
    logger.info("ChromaDB is persisted at: %s", CHROMA_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
