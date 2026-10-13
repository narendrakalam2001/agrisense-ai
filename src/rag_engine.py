"""RAG (Retrieval-Augmented Generation) treatment advisory engine.

Given a detected disease name, retrieves the most relevant knowledge-base
passages from ChromaDB and returns a structured treatment plan: immediate
action, chemical treatment, organic alternative, prevention, and estimated
recovery time — the exact shape the Streamlit treatment card and the
``/detect`` API response expect.

Falls back gracefully to the structured ``pesticide_database.json`` entry
directly if the vector store is empty or unavailable, so the app never
shows a blank treatment card.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from src.config import CHROMADB_DIR, KNOWLEDGE_BASE_DIR, configure_logging
from src.knowledge_ingestor import COLLECTION_NAME, EMBEDDING_MODEL_NAME, IngestionError, ingest_all

logger = configure_logging(__name__)

DEFAULT_TOP_K = 4


class RAGEngineError(Exception):
    """Raised when the RAG engine cannot answer a query."""


@dataclass
class TreatmentPlan:
    """Structured treatment plan returned to the API / Streamlit layer."""

    disease_name_en: str
    disease_name_hi: str
    immediate_action_en: str
    immediate_action_hi: str
    chemical_treatment: dict
    organic_alternative_en: str
    organic_alternative_hi: str
    prevention_en: str
    prevention_hi: str
    estimated_recovery: str
    retrieved_context: list[str]

    def to_dict(self) -> dict:
        return {
            "disease_name_en": self.disease_name_en,
            "disease_name_hi": self.disease_name_hi,
            "immediate_action_en": self.immediate_action_en,
            "immediate_action_hi": self.immediate_action_hi,
            "chemical_treatment": self.chemical_treatment,
            "organic_alternative_en": self.organic_alternative_en,
            "organic_alternative_hi": self.organic_alternative_hi,
            "prevention_en": self.prevention_en,
            "prevention_hi": self.prevention_hi,
            "estimated_recovery": self.estimated_recovery,
            "retrieved_context": self.retrieved_context,
        }


class RAGEngine:
    """Thin wrapper around a persistent ChromaDB collection for retrieval.

    Instantiate once (e.g. as a FastAPI startup dependency) and reuse —
    embedding model loading is the expensive part.
    """

    def __init__(
        self,
        persist_dir: Path = CHROMADB_DIR,
        knowledge_base_dir: Path = KNOWLEDGE_BASE_DIR,
        collection_name: str = COLLECTION_NAME,
        auto_ingest_if_empty: bool = True,
    ) -> None:
        """Connect to (or lazily build) the ChromaDB collection.

        Args:
            persist_dir: Directory for the persistent ChromaDB store.
            knowledge_base_dir: Source directory used for lazy re-ingestion.
            collection_name: Name of the Chroma collection to query.
            auto_ingest_if_empty: If ``True``, automatically run the full
                ingestion pipeline the first time the collection is empty
                or missing, so a fresh clone "just works".
        """
        self.persist_dir = persist_dir
        self.knowledge_base_dir = knowledge_base_dir
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )

        self._collection = self._client.get_or_create_collection(
            name=collection_name, embedding_function=self._embedding_fn
        )

        if auto_ingest_if_empty and self._collection.count() == 0:
            logger.info("Vector store is empty — running one-time ingestion.")
            try:
                ingest_all(knowledge_base_dir, persist_dir)
                self._collection = self._client.get_or_create_collection(
                    name=collection_name, embedding_function=self._embedding_fn
                )
            except IngestionError as exc:
                logger.error("Auto-ingestion failed: %s. RAG will use JSON fallback only.", exc)

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[str]:
        """Retrieve the top-k most relevant passages for a query.

        Args:
            query: Natural-language query (e.g. a disease name or a
                farmer's question).
            top_k: Number of passages to retrieve.

        Returns:
            List of passage strings, most relevant first. Empty list if the
            collection has no data.
        """
        if self._collection.count() == 0:
            logger.warning("Vector store is empty — returning no context.")
            return []

        results = self._collection.query(query_texts=[query], n_results=top_k)
        documents = results.get("documents", [[]])[0]
        return documents

    def _load_json_fallback(self, disease_class_name: str) -> dict | None:
        """Load a disease's treatment directly from the JSON databases.

        Used when the vector store lookup fails or the class name is known
        exactly (e.g. straight from the CV model's prediction), which is
        both faster and more precise than a semantic search for this case.

        Args:
            disease_class_name: Exact class name as used in ``CLASS_NAMES``.

        Returns:
            Merged dict of disease_info + pesticide_database fields, or
            ``None`` if the class name isn't found in either file.
        """
        info_path = self.knowledge_base_dir / "disease_info.json"
        pesticide_path = self.knowledge_base_dir / "pesticide_database.json"
        if not info_path.exists() or not pesticide_path.exists():
            return None

        disease_info = json.loads(info_path.read_text(encoding="utf-8"))
        pesticide_data = json.loads(pesticide_path.read_text(encoding="utf-8"))

        if disease_class_name not in disease_info:
            return None

        return {**disease_info[disease_class_name], **pesticide_data.get(disease_class_name, {})}

    def get_treatment_plan(self, disease_class_name: str, top_k: int = DEFAULT_TOP_K) -> TreatmentPlan:
        """Produce a full structured treatment plan for a detected disease.

        Prefers an exact JSON lookup (precise, zero hallucination risk) and
        enriches it with semantically retrieved context passages (e.g. from
        the FAQ or ICAR guidelines) for the chatbot to draw on for
        follow-up questions.

        Args:
            disease_class_name: Exact class name as predicted by the CV
                model (must match ``CLASS_NAMES`` in ``src.config``).
            top_k: Number of supporting passages to retrieve for context.

        Returns:
            A populated :class:`TreatmentPlan`.

        Raises:
            RAGEngineError: If the disease isn't found in the knowledge base
                at all (neither JSON nor vector store).
        """
        record = self._load_json_fallback(disease_class_name)
        query = disease_class_name.replace("___", " ").replace("_", " ")
        retrieved_context = self.retrieve(query, top_k=top_k)

        if record is None:
            if not retrieved_context:
                raise RAGEngineError(
                    f"No knowledge base entry found for '{disease_class_name}'."
                )
            # Degrade gracefully: return whatever the vector store found,
            # without structured fields.
            return TreatmentPlan(
                disease_name_en=disease_class_name,
                disease_name_hi="",
                immediate_action_en="See retrieved context below.",
                immediate_action_hi="नीचे दिए गए संदर्भ को देखें।",
                chemical_treatment={},
                organic_alternative_en="",
                organic_alternative_hi="",
                prevention_en="",
                prevention_hi="",
                estimated_recovery="",
                retrieved_context=retrieved_context,
            )

        return TreatmentPlan(
            disease_name_en=record.get("disease_name_en", disease_class_name),
            disease_name_hi=record.get("disease_name_hi", ""),
            immediate_action_en=record.get("immediate_action_en", ""),
            immediate_action_hi=record.get("immediate_action_hi", ""),
            chemical_treatment=record.get("chemical_treatment", {}),
            organic_alternative_en=record.get("organic_alternative_en", ""),
            organic_alternative_hi=record.get("organic_alternative_hi", ""),
            prevention_en=record.get("prevention_en", ""),
            prevention_hi=record.get("prevention_hi", ""),
            estimated_recovery=record.get("estimated_recovery", ""),
            retrieved_context=retrieved_context,
        )

    def answer_free_text_query(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[str]:
        """Retrieve supporting passages for an arbitrary farmer question.

        Used by the chatbot's ``treatment_retriever`` tool when the
        question doesn't map cleanly onto a single known disease class
        (e.g. "kaunsi dawai use karein?" following an earlier diagnosis).

        Args:
            query: The farmer's free-text question (Hindi, English, or
                Hinglish — embedding model handles multilingual input
                reasonably well; exact translation happens upstream in
                ``translator.py`` if needed).
            top_k: Number of passages to retrieve.

        Returns:
            List of relevant passages, most relevant first.
        """
        return self.retrieve(query, top_k=top_k)
