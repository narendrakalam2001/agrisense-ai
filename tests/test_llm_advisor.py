"""Tests for src/llm_advisor.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.llm_advisor import (
    LLMAdvisorError,
    build_explanation_prompt,
    build_fallback_explanation,
    explain_treatment_plan,
)
from src.rag_engine import TreatmentPlan


@pytest.fixture
def sample_plan() -> TreatmentPlan:
    return TreatmentPlan(
        disease_name_en="Late Blight",
        disease_name_hi="पछेती झुलसा रोग",
        immediate_action_en="Spray within 24-48 hours.",
        immediate_action_hi="24-48 घंटे के भीतर छिड़काव करें।",
        chemical_treatment={
            "pesticide_name": "Cymoxanil + Mancozeb", "brand_examples": "Curzate M-8",
            "dosage": "3 g per litre", "frequency": "Every 5-7 days", "safety_en": "Treat the whole field.",
        },
        organic_alternative_en="Copper oxychloride spray.",
        organic_alternative_hi="कॉपर ऑक्सीक्लोराइड छिड़काव।",
        prevention_en="Improve drainage.",
        prevention_hi="जल निकासी सुधारें।",
        estimated_recovery="Critical — act fast.",
        retrieved_context=[],
    )


def test_build_explanation_prompt_includes_chemical_name(sample_plan):
    prompt = build_explanation_prompt(sample_plan, language="english")
    assert "Cymoxanil" in prompt
    assert "English" in prompt


def test_build_explanation_prompt_hinglish_instruction(sample_plan):
    prompt = build_explanation_prompt(sample_plan, language="hinglish")
    assert "Hinglish" in prompt


def test_explain_treatment_plan_uses_injected_client(sample_plan):
    mock_response = MagicMock()
    mock_response.text = "Spray Cymoxanil + Mancozeb now."
    mock_client = MagicMock()
    mock_client.generate_content.return_value = mock_response

    result = explain_treatment_plan(sample_plan, language="english", gemini_client=mock_client)
    assert "Cymoxanil" in result.explanation_text
    mock_client.generate_content.assert_called_once()


def test_explain_treatment_plan_raises_on_empty_response(sample_plan):
    mock_response = MagicMock()
    mock_response.text = ""
    mock_client = MagicMock()
    mock_client.generate_content.return_value = mock_response

    with pytest.raises(LLMAdvisorError):
        explain_treatment_plan(sample_plan, language="english", gemini_client=mock_client)


def test_explain_treatment_plan_raises_on_client_exception(sample_plan):
    mock_client = MagicMock()
    mock_client.generate_content.side_effect = RuntimeError("API down")

    with pytest.raises(LLMAdvisorError):
        explain_treatment_plan(sample_plan, language="english", gemini_client=mock_client)


def test_build_fallback_explanation_english_contains_pesticide(sample_plan):
    result = build_fallback_explanation(sample_plan, language="english")
    assert "Cymoxanil" in result.explanation_text
    assert result.language == "english"


def test_build_fallback_explanation_hindi_contains_hindi_disease_name(sample_plan):
    result = build_fallback_explanation(sample_plan, language="hindi")
    assert "पछेती झुलसा रोग" in result.explanation_text
