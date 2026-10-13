"""Knowledge base ingestion entry point.

Usage:
    python scripts/ingest_knowledge_base.py
    python scripts/ingest_knowledge_base.py --no-reset   # append instead of rebuild

Run this once after adding/updating any file in data/knowledge_base/
(PDFs, disease_info.json, pesticide_database.json), and again any time
that content changes — the FastAPI/Streamlit layers assume the ChromaDB
store is already built and query it read-only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CHROMADB_DIR, KNOWLEDGE_BASE_DIR, configure_logging
from src.knowledge_ingestor import (
    IngestionError,
    build_vector_store,
    load_json_chunks,
    load_pdf_chunks,
)

logger = configure_logging(__name__, log_filename="ingest_knowledge_base.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest the AgriSense AI knowledge base into ChromaDB.")
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Append to the existing collection instead of rebuilding it from scratch.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logger.info("Loading PDF sources from '%s'...", KNOWLEDGE_BASE_DIR)
    pdf_chunks = load_pdf_chunks(KNOWLEDGE_BASE_DIR)

    logger.info("Loading disease_info.json / pesticide_database.json...")
    try:
        json_chunks = load_json_chunks(KNOWLEDGE_BASE_DIR)
    except IngestionError as exc:
        logger.error("Failed to load JSON knowledge base: %s", exc)
        return 1

    all_chunks = pdf_chunks + json_chunks
    logger.info(
        "Total chunks to ingest: %d (%d from PDFs, %d from JSON).",
        len(all_chunks), len(pdf_chunks), len(json_chunks),
    )

    try:
        count = build_vector_store(all_chunks, persist_dir=CHROMADB_DIR, reset=not args.no_reset)
    except IngestionError as exc:
        logger.error("Ingestion failed: %s", exc)
        return 1

    logger.info("Ingestion complete: %d chunks stored in '%s'.", count, CHROMADB_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
