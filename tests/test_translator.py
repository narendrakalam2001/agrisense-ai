"""Tests for src/translator.py."""

from __future__ import annotations

import pytest

from src.translator import (
    DetectedLanguage,
    TranslationError,
    build_translation_prompt,
    detect_language,
    normalize_for_retrieval,
)


def test_detect_language_devanagari_text():
    result = detect_language("मेरे टमाटर के पत्ते पर धब्बे हैं")
    assert result.language == DetectedLanguage.HINDI
    assert result.devanagari_char_count > 0


def test_detect_language_pure_english():
    result = detect_language("What pesticide should I use for late blight?")
    assert result.language == DetectedLanguage.ENGLISH


def test_detect_language_hinglish():
    result = detect_language("Mere tomato ke patte pe kale daag hain")
    assert result.language == DetectedLanguage.HINGLISH
    assert result.hinglish_marker_count > 0


def test_detect_language_hinglish_kaunsi_dawai():
    result = detect_language("Kaunsi dawai use karein?")
    assert result.language == DetectedLanguage.HINGLISH


def test_build_translation_prompt_contains_source_text():
    prompt = build_translation_prompt("Spray Mancozeb 2.5g per litre", "Hindi")
    assert "Mancozeb" in prompt
    assert "Hindi" in prompt


def test_normalize_for_retrieval_passthrough():
    text = "Kaunsi dawai use karein?"
    assert normalize_for_retrieval(text) == text


def test_translation_error_is_exception():
    assert issubclass(TranslationError, Exception)
