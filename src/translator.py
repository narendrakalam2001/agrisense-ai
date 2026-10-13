"""Language detection and translation helpers for the Kisan Saathi chatbot.

Farmers may type in Hindi (Devanagari script), English, or Hinglish
(Hindi written in Latin script — very common on Indian keyboards). This
module detects which one is in play and translates as needed so the RAG
engine (English-indexed knowledge base) and the LLM chatbot can both work
reliably regardless of the input script.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from src.config import configure_logging

logger = configure_logging(__name__)

# Common Hinglish tokens — words a farmer is likely to type in Latin script
# that signal Hindi intent even with no Devanagari characters present.
# Not exhaustive; extend based on real conversation logs.
HINGLISH_MARKERS = frozenset(
    {
        "hai", "hain", "kya", "kaunsi", "kaunsa", "kaise", "karein", "karna",
        "mera", "meri", "mere", "fasal", "patte", "patta", "dawai", "dawa",
        "spray", "chahiye", "kab", "kyun", "nahi", "haan", "paani", "khet",
        "kheti", "bimari", "rog", "upchar", "aaj", "kal", "subah", "shaam",
    }
)

DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")


class DetectedLanguage(str, Enum):
    """Language/script detected in a piece of user input."""

    HINDI = "hindi"  # Devanagari script
    HINGLISH = "hinglish"  # Hindi words, Latin script
    ENGLISH = "english"


@dataclass
class LanguageDetectionResult:
    """Result of running :func:`detect_language` on a message."""

    language: DetectedLanguage
    devanagari_char_count: int
    hinglish_marker_count: int


class TranslationError(Exception):
    """Raised when a translation request cannot be completed."""


def detect_language(text: str) -> LanguageDetectionResult:
    """Detect whether text is Hindi (Devanagari), Hinglish, or English.

    Heuristic, not a full language-ID model: counts Devanagari characters
    first (unambiguous signal for Hindi script), then falls back to
    counting known Hinglish marker words for Latin-script Hindi.

    Args:
        text: Raw user message.

    Returns:
        A :class:`LanguageDetectionResult` with the detected language and
        the signal counts that produced the decision.
    """
    devanagari_count = len(DEVANAGARI_PATTERN.findall(text))
    if devanagari_count > 0:
        return LanguageDetectionResult(DetectedLanguage.HINDI, devanagari_count, 0)

    words = re.findall(r"[a-zA-Z]+", text.lower())
    hinglish_hits = sum(1 for w in words if w in HINGLISH_MARKERS)

    if hinglish_hits >= 1:
        return LanguageDetectionResult(DetectedLanguage.HINGLISH, 0, hinglish_hits)

    return LanguageDetectionResult(DetectedLanguage.ENGLISH, 0, 0)


def build_translation_prompt(text: str, target_language: str) -> str:
    """Build the LLM prompt used to translate a short farmer-facing message.

    Args:
        text: Source text to translate.
        target_language: One of ``"Hindi"``, ``"English"``.

    Returns:
        A prompt string ready to send to the Gemini API.
    """
    return (
        f"Translate the following agricultural advisory text into "
        f"{target_language}, using simple words a farmer with basic "
        f"schooling would understand. Keep pesticide/chemical names "
        f"unchanged. Return ONLY the translation, no explanation.\n\n"
        f"Text: {text}"
    )


def translate_text(text: str, target_language: str, gemini_client=None) -> str:
    """Translate text via the Gemini API.

    Args:
        text: Source text.
        target_language: ``"Hindi"`` or ``"English"``.
        gemini_client: Optional pre-configured ``google.generativeai``
            client/model instance, primarily for dependency injection in
            tests. If omitted, a default model is constructed using
            ``GEMINI_API_KEY`` from the environment.

    Returns:
        Translated text.

    Raises:
        TranslationError: If the API call fails or returns empty output.
    """
    if gemini_client is None:
        from src.llm_client import GeminiClientError, GeminiTextClient

        try:
            gemini_client = GeminiTextClient()
        except GeminiClientError as exc:
            raise TranslationError(str(exc)) from exc

    prompt = build_translation_prompt(text, target_language)

    try:
        response = gemini_client.generate_content(prompt)
    except Exception as exc:  # noqa: BLE001 - surfaced as a domain error
        raise TranslationError(f"Gemini translation call failed: {exc}") from exc

    translated = (getattr(response, "text", "") or "").strip()
    if not translated:
        raise TranslationError("Gemini returned an empty translation.")
    return translated


def normalize_for_retrieval(text: str, detection: LanguageDetectionResult | None = None) -> str:
    """Best-effort normalization of user text before RAG retrieval.

    The knowledge base is indexed primarily in English. For Hindi/Hinglish
    input we currently pass text through unchanged and rely on the
    multilingual embedding model's cross-lingual retrieval ability; if
    retrieval quality proves insufficient in testing, wire this up to
    :func:`translate_text` to translate to English first.

    Args:
        text: Raw user message.
        detection: Pre-computed detection result, to avoid detecting twice.

    Returns:
        Text ready to pass to ``RAGEngine.retrieve``.
    """
    detection = detection or detect_language(text)
    logger.debug("normalize_for_retrieval: detected=%s text=%r", detection.language, text)
    return text
