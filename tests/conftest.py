"""Shared pytest fixtures for the AgriSense AI test suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def sample_disease_info() -> dict:
    """A minimal 2-class disease_info.json payload for isolated tests."""
    return {
        "Tomato___Late_blight": {
            "crop": "Tomato", "crop_hindi": "टमाटर",
            "disease_name_en": "Late Blight", "disease_name_hi": "पछेती झुलसा रोग",
            "pathogen": "Phytophthora infestans (oomycete)",
            "symptoms_en": "Water-soaked patches turning brown-black rapidly.",
            "symptoms_hi": "पानी-भीगे धब्बे तेज़ी से भूरे-काले होते हैं।",
            "severity": "critical",
        },
        "Tomato___healthy": {
            "crop": "Tomato", "crop_hindi": "टमाटर",
            "disease_name_en": "Healthy", "disease_name_hi": "स्वस्थ",
            "pathogen": "None",
            "symptoms_en": "No disease symptoms observed.",
            "symptoms_hi": "कोई रोग लक्षण नहीं दिखा।",
            "severity": "none",
        },
    }


@pytest.fixture
def sample_pesticide_data() -> dict:
    """A minimal 2-class pesticide_database.json payload matching sample_disease_info."""
    return {
        "Tomato___Late_blight": {
            "immediate_action_en": "Spray within 24-48 hours.",
            "immediate_action_hi": "24-48 घंटे के भीतर छिड़काव करें।",
            "chemical_treatment": {
                "pesticide_name": "Cymoxanil + Mancozeb", "brand_examples": "Curzate M-8",
                "dosage": "3 g per litre", "frequency": "Every 5-7 days",
                "safety_en": "Treat the whole field.", "safety_hi": "पूरे खेत का उपचार करें।",
            },
            "organic_alternative_en": "Copper oxychloride spray.",
            "organic_alternative_hi": "कॉपर ऑक्सीक्लोराइड छिड़काव।",
            "prevention_en": "Improve drainage.", "prevention_hi": "जल निकासी सुधारें।",
            "estimated_recovery": "Critical — act fast.",
        },
        "Tomato___healthy": {
            "immediate_action_en": "No treatment needed.",
            "immediate_action_hi": "उपचार आवश्यक नहीं।",
            "chemical_treatment": {
                "pesticide_name": "Not applicable", "brand_examples": "N/A",
                "dosage": "N/A", "frequency": "N/A", "safety_en": "N/A", "safety_hi": "N/A",
            },
            "organic_alternative_en": "Maintain nutrition.",
            "organic_alternative_hi": "पोषण बनाए रखें।",
            "prevention_en": "Rotate crops.", "prevention_hi": "फसल चक्र अपनाएँ।",
            "estimated_recovery": "N/A",
        },
    }


@pytest.fixture
def knowledge_base_dir(tmp_path: Path, sample_disease_info: dict, sample_pesticide_data: dict) -> Path:
    """A temporary knowledge_base/ directory populated with sample JSON files."""
    kb_dir = tmp_path / "knowledge_base"
    kb_dir.mkdir()
    (kb_dir / "disease_info.json").write_text(json.dumps(sample_disease_info), encoding="utf-8")
    (kb_dir / "pesticide_database.json").write_text(json.dumps(sample_pesticide_data), encoding="utf-8")
    return kb_dir
