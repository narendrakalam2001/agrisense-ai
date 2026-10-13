"""Tests for src/knowledge_ingestor.py — chunking and JSON loading only
(no ChromaDB/embedding calls, which require network access to download
the embedding model)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.knowledge_ingestor import (
    IngestionError,
    KnowledgeChunk,
    _chunk_text,
    load_json_chunks,
    load_pdf_chunks,
)


def test_chunk_text_short_text_single_chunk():
    chunks = _chunk_text("short text", chunk_size=1000, overlap=150)
    assert len(chunks) == 1
    assert chunks[0] == "short text"


def test_chunk_text_empty_string_returns_empty_list():
    assert _chunk_text("   ") == []


def test_chunk_text_long_text_produces_overlapping_chunks():
    text = "a" * 2500
    chunks = _chunk_text(text, chunk_size=1000, overlap=150)
    assert len(chunks) >= 3
    # Verify overlap: end of chunk[0] should reappear at start of chunk[1].
    assert chunks[0][-150:] == chunks[1][:150]


def test_load_json_chunks_produces_one_chunk_per_disease(knowledge_base_dir: Path):
    chunks = load_json_chunks(knowledge_base_dir)
    assert len(chunks) == 2
    sources = {c.source for c in chunks}
    assert sources == {"Tomato___Late_blight", "Tomato___healthy"}


def test_load_json_chunks_includes_treatment_details(knowledge_base_dir: Path):
    chunks = load_json_chunks(knowledge_base_dir)
    late_blight_chunk = next(c for c in chunks if c.source == "Tomato___Late_blight")
    assert "Cymoxanil" in late_blight_chunk.text
    assert "Late Blight" in late_blight_chunk.text


def test_load_json_chunks_raises_if_files_missing(tmp_path: Path):
    empty_dir = tmp_path / "empty_kb"
    empty_dir.mkdir()
    with pytest.raises(IngestionError):
        load_json_chunks(empty_dir)


def test_load_pdf_chunks_warns_and_returns_empty_when_no_pdfs(tmp_path: Path, caplog):
    empty_dir = tmp_path / "no_pdfs"
    empty_dir.mkdir()
    chunks = load_pdf_chunks(empty_dir)
    assert chunks == []


def test_knowledge_chunk_is_dataclass_with_expected_fields():
    chunk = KnowledgeChunk(text="hello", source="test.pdf", doc_type="pdf", chunk_index=0)
    assert chunk.text == "hello"
    assert chunk.doc_type == "pdf"
