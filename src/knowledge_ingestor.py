"""Knowledge base ingestion: PDFs + JSON databases -> ChromaDB vector store.

Consumed by ``scripts/ingest_knowledge_base.py`` (run once, or whenever the
knowledge base content changes) and imported by ``rag_engine.py`` for
lazily rebuilding a missing store.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

from src.config import CHROMADB_DIR, KNOWLEDGE_BASE_DIR, configure_logging

logger = configure_logging(__name__)

COLLECTION_NAME = "agrisense_knowledge_base"
CHUNK_SIZE_CHARS = 1000
CHUNK_OVERLAP_CHARS = 150
# Free, local sentence-transformers model — no API key required for ingestion.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class IngestionError(Exception):
    """Raised when the knowledge base cannot be ingested."""


@dataclass
class KnowledgeChunk:
    """A single chunk of text ready to embed, with traceable metadata."""

    text: str
    source: str
    doc_type: str  # "pdf" | "disease_info" | "pesticide_database"
    chunk_index: int


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Split long text into overlapping fixed-size chunks.

    Args:
        text: Raw text to split.
        chunk_size: Target chunk length in characters.
        overlap: Number of characters of overlap between consecutive chunks,
            which helps preserve context across chunk boundaries.

    Returns:
        List of text chunks (never empty unless ``text`` is empty/whitespace).
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if end >= len(text):
            break
    return chunks


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract raw text from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Concatenated text of all pages.

    Raises:
        IngestionError: If the PDF cannot be read or has no extractable text.
    """
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:  # pypdf raises varied exception types
        raise IngestionError(f"Could not open PDF '{pdf_path}': {exc}") from exc

    pages_text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages_text.append(page_text)

    full_text = "\n".join(pages_text).strip()
    if not full_text:
        raise IngestionError(f"No extractable text found in '{pdf_path}'.")
    return full_text


def load_pdf_chunks(knowledge_base_dir: Path = KNOWLEDGE_BASE_DIR) -> list[KnowledgeChunk]:
    """Extract and chunk every PDF found under ``knowledge_base_dir``.

    Args:
        knowledge_base_dir: Directory containing ICAR/advisory/FAQ PDFs.

    Returns:
        List of :class:`KnowledgeChunk` covering every PDF in the directory.
    """
    chunks: list[KnowledgeChunk] = []
    pdf_paths = sorted(knowledge_base_dir.glob("*.pdf"))

    if not pdf_paths:
        logger.warning("No PDF files found in '%s'.", knowledge_base_dir)

    for pdf_path in pdf_paths:
        try:
            text = extract_pdf_text(pdf_path)
        except IngestionError as exc:
            logger.warning("Skipping PDF: %s", exc)
            continue

        for i, chunk_text in enumerate(_chunk_text(text)):
            chunks.append(
                KnowledgeChunk(text=chunk_text, source=pdf_path.name, doc_type="pdf", chunk_index=i)
            )
        logger.info("Chunked '%s' into %d piece(s).", pdf_path.name, len(_chunk_text(text)))

    return chunks


def load_json_chunks(knowledge_base_dir: Path = KNOWLEDGE_BASE_DIR) -> list[KnowledgeChunk]:
    """Flatten ``disease_info.json`` and ``pesticide_database.json`` into
    retrievable text chunks, one per disease.

    Each chunk combines the disease's symptoms and treatment protocol into a
    single natural-language passage so retrieval returns everything needed
    to answer a farmer's question about that disease in one hit.

    Args:
        knowledge_base_dir: Directory containing the two JSON files.

    Returns:
        List of :class:`KnowledgeChunk`, one per disease class found.

    Raises:
        IngestionError: If either JSON file is missing or malformed.
    """
    disease_info_path = knowledge_base_dir / "disease_info.json"
    pesticide_path = knowledge_base_dir / "pesticide_database.json"

    for path in (disease_info_path, pesticide_path):
        if not path.exists():
            raise IngestionError(f"Required knowledge base file not found: '{path}'.")

    try:
        disease_info = json.loads(disease_info_path.read_text(encoding="utf-8"))
        pesticide_data = json.loads(pesticide_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IngestionError(f"Malformed JSON in knowledge base: {exc}") from exc

    chunks: list[KnowledgeChunk] = []
    for class_name, info in disease_info.items():
        treatment = pesticide_data.get(class_name, {})
        chem = treatment.get("chemical_treatment", {})

        passage = (
            f"Crop: {info.get('crop', 'Unknown')}. "
            f"Disease: {info.get('disease_name_en', class_name)} "
            f"({info.get('disease_name_hi', '')}). "
            f"Pathogen: {info.get('pathogen', 'Unknown')}. "
            f"Symptoms: {info.get('symptoms_en', '')} "
            f"Severity: {info.get('severity', 'unknown')}. "
            f"Immediate action: {treatment.get('immediate_action_en', 'N/A')} "
            f"Chemical treatment: {chem.get('pesticide_name', 'N/A')}, "
            f"dose {chem.get('dosage', 'N/A')}, frequency {chem.get('frequency', 'N/A')}. "
            f"Safety: {chem.get('safety_en', 'N/A')} "
            f"Organic alternative: {treatment.get('organic_alternative_en', 'N/A')} "
            f"Prevention: {treatment.get('prevention_en', 'N/A')} "
            f"Estimated recovery: {treatment.get('estimated_recovery', 'N/A')}"
        )
        chunks.append(
            KnowledgeChunk(text=passage, source=class_name, doc_type="disease_pesticide", chunk_index=0)
        )

    logger.info("Built %d disease/pesticide knowledge chunks.", len(chunks))
    return chunks


def get_chroma_client(persist_dir: Path = CHROMADB_DIR) -> chromadb.ClientAPI:
    """Get a persistent ChromaDB client.

    Args:
        persist_dir: Directory where ChromaDB stores its data.

    Returns:
        A ChromaDB persistent client.
    """
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def build_vector_store(
    chunks: list[KnowledgeChunk],
    persist_dir: Path = CHROMADB_DIR,
    collection_name: str = COLLECTION_NAME,
    reset: bool = True,
) -> int:
    """Embed and store all knowledge chunks in ChromaDB.

    Args:
        chunks: Chunks produced by :func:`load_pdf_chunks` /
            :func:`load_json_chunks`.
        persist_dir: Directory for the persistent ChromaDB store.
        collection_name: Name of the Chroma collection to (re)build.
        reset: If ``True``, drop any existing collection with this name
            before ingesting (use ``False`` to append incrementally).

    Returns:
        Number of chunks stored.

    Raises:
        IngestionError: If ``chunks`` is empty.
    """
    if not chunks:
        raise IngestionError("No chunks to ingest — knowledge base would be empty.")

    client = get_chroma_client(persist_dir)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    if reset:
        try:
            client.delete_collection(collection_name)
            logger.info("Dropped existing collection '%s'.", collection_name)
        except Exception:  # noqa: BLE001 - chromadb raises if collection absent
            pass

    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_fn
    )

    ids = [f"{c.doc_type}_{c.source}_{c.chunk_index}".replace(" ", "_") for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [{"source": c.source, "doc_type": c.doc_type, "chunk_index": c.chunk_index} for c in chunks]

    # Chroma recommends batching very large ingestions; our KB is small
    # enough (~150 pages equivalent) to insert in one call.
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    logger.info("Stored %d chunks in collection '%s'.", len(chunks), collection_name)
    return len(chunks)


def ingest_all(
    knowledge_base_dir: Path = KNOWLEDGE_BASE_DIR,
    persist_dir: Path = CHROMADB_DIR,
) -> int:
    """Run the full ingestion pipeline: PDFs + JSON -> ChromaDB.

    Args:
        knowledge_base_dir: Directory containing PDFs and JSON databases.
        persist_dir: Directory for the persistent ChromaDB store.

    Returns:
        Total number of chunks stored.
    """
    pdf_chunks = load_pdf_chunks(knowledge_base_dir)
    json_chunks = load_json_chunks(knowledge_base_dir)
    all_chunks = pdf_chunks + json_chunks
    return build_vector_store(all_chunks, persist_dir=persist_dir)


if __name__ == "__main__":
    total = ingest_all()
    logger.info("Ingestion complete: %d total chunks stored.", total)
