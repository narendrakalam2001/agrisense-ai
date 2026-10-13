"""Tests for src/rag_engine.py. ChromaDB and the embedding model are mocked
so tests run without network access or downloading model weights."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.rag_engine import RAGEngine, RAGEngineError, TreatmentPlan


def _make_mock_collection(count=1, query_docs=None):
    collection = MagicMock()
    collection.count.return_value = count
    collection.query.return_value = {"documents": [query_docs or ["some context passage"]]}
    return collection


@patch("src.rag_engine.embedding_functions.SentenceTransformerEmbeddingFunction")
@patch("src.rag_engine.chromadb.PersistentClient")
def test_rag_engine_get_treatment_plan_exact_json_match(
    mock_client_cls, mock_embed_fn, tmp_path: Path, knowledge_base_dir: Path
):
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = _make_mock_collection(count=5)
    mock_client_cls.return_value = mock_client

    persist_dir = tmp_path / "chromadb_store"
    engine = RAGEngine(
        persist_dir=persist_dir, knowledge_base_dir=knowledge_base_dir, auto_ingest_if_empty=False
    )

    plan = engine.get_treatment_plan("Tomato___Late_blight")
    assert isinstance(plan, TreatmentPlan)
    assert plan.disease_name_en == "Late Blight"
    assert "Cymoxanil" in plan.chemical_treatment.get("pesticide_name", "")
    assert plan.retrieved_context == ["some context passage"]


@patch("src.rag_engine.embedding_functions.SentenceTransformerEmbeddingFunction")
@patch("src.rag_engine.chromadb.PersistentClient")
def test_rag_engine_raises_for_unknown_disease_with_no_context(
    mock_client_cls, mock_embed_fn, tmp_path: Path, knowledge_base_dir: Path
):
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = _make_mock_collection(count=0, query_docs=[])
    mock_client_cls.return_value = mock_client

    persist_dir = tmp_path / "chromadb_store"
    engine = RAGEngine(
        persist_dir=persist_dir, knowledge_base_dir=knowledge_base_dir, auto_ingest_if_empty=False
    )

    with pytest.raises(RAGEngineError):
        engine.get_treatment_plan("Corn___Nonexistent_disease")


@patch("src.rag_engine.embedding_functions.SentenceTransformerEmbeddingFunction")
@patch("src.rag_engine.chromadb.PersistentClient")
def test_rag_engine_retrieve_returns_empty_when_collection_empty(
    mock_client_cls, mock_embed_fn, tmp_path: Path, knowledge_base_dir: Path
):
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = _make_mock_collection(count=0)
    mock_client_cls.return_value = mock_client

    persist_dir = tmp_path / "chromadb_store"
    engine = RAGEngine(
        persist_dir=persist_dir, knowledge_base_dir=knowledge_base_dir, auto_ingest_if_empty=False
    )

    assert engine.retrieve("late blight tomato") == []


@patch("src.rag_engine.embedding_functions.SentenceTransformerEmbeddingFunction")
@patch("src.rag_engine.chromadb.PersistentClient")
def test_rag_engine_answer_free_text_query(
    mock_client_cls, mock_embed_fn, tmp_path: Path, knowledge_base_dir: Path
):
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = _make_mock_collection(
        count=3, query_docs=["passage A", "passage B"]
    )
    mock_client_cls.return_value = mock_client

    persist_dir = tmp_path / "chromadb_store"
    engine = RAGEngine(
        persist_dir=persist_dir, knowledge_base_dir=knowledge_base_dir, auto_ingest_if_empty=False
    )

    results = engine.answer_free_text_query("kaunsi dawai use karein?")
    assert results == ["passage A", "passage B"]
