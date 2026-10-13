"""Tests for serving/api.py. Requires TensorFlow (imported transitively via
services.detection_service) and fastapi; skipped automatically if either is
unavailable. All model/RAG/chatbot singletons are monkeypatched — no real
model file or API keys needed."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("tensorflow")
fastapi_testclient = pytest.importorskip("fastapi.testclient")

import serving.api as api_module
from fastapi.testclient import TestClient
from services.detection_service import ClassPrediction, DetectionResult


@pytest.fixture
def client(monkeypatch):
    """A TestClient with the detection/RAG singletons replaced by mocks."""
    mock_detection_result = DetectionResult(
        predicted_class="Tomato___Late_blight",
        confidence=0.91,
        top_3=[
            ClassPrediction("Tomato___Late_blight", 0.91),
            ClassPrediction("Tomato___Early_blight", 0.06),
            ClassPrediction("Tomato___healthy", 0.03),
        ],
        gradcam_image_base64=None,
    )
    mock_detection_service = MagicMock()
    mock_detection_service.detect.return_value = mock_detection_result

    mock_plan = MagicMock()
    mock_plan.to_dict.return_value = {"disease_name_en": "Late Blight"}
    mock_rag_engine = MagicMock()
    mock_rag_engine.get_treatment_plan.return_value = mock_plan

    monkeypatch.setattr(api_module, "_detection_service", mock_detection_service)
    monkeypatch.setattr(api_module, "_rag_engine", mock_rag_engine)
    monkeypatch.setattr(api_module, "_chatbot_sessions", {})

    return TestClient(api_module.app)


def test_health_endpoint_ok_when_both_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["rag_loaded"] is True


def test_health_endpoint_degraded_when_model_missing(client, monkeypatch):
    monkeypatch.setattr(api_module, "_detection_service", None)
    response = client.get("/health")
    assert response.json()["status"] == "degraded"


def test_detect_endpoint_returns_prediction(client):
    files = {"file": ("leaf.jpg", b"fake-image-bytes", "image/jpeg")}
    response = client.post("/detect", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == "Tomato___Late_blight"
    assert body["confidence"] == pytest.approx(0.91)
    assert len(body["top_3"]) == 3


def test_detect_endpoint_503_when_model_not_loaded(client, monkeypatch):
    monkeypatch.setattr(api_module, "_detection_service", None)
    files = {"file": ("leaf.jpg", b"fake-image-bytes", "image/jpeg")}
    response = client.post("/detect", files=files)
    assert response.status_code == 503


def test_chat_endpoint_creates_and_reuses_session(client, monkeypatch):
    mock_session = MagicMock()
    mock_session.send_message.return_value = "Spray Mancozeb for early blight."

    monkeypatch.setattr(
        api_module, "KisanSaathiChatbot", MagicMock(return_value=mock_session)
    )

    response = client.post("/chat", json={"conversation_id": "farmer1", "message": "hello"})
    assert response.status_code == 200
    assert response.json()["reply"] == "Spray Mancozeb for early blight."
    assert "farmer1" in api_module._chatbot_sessions


def test_chat_endpoint_503_when_rag_engine_missing(client, monkeypatch):
    monkeypatch.setattr(api_module, "_rag_engine", None)
    response = client.post("/chat", json={"conversation_id": "farmer1", "message": "hello"})
    assert response.status_code == 503


def test_get_disease_returns_404_for_unknown_class(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api_module, "KNOWLEDGE_BASE_DIR", tmp_path)
    (tmp_path / "disease_info.json").write_text("{}", encoding="utf-8")
    (tmp_path / "pesticide_database.json").write_text("{}", encoding="utf-8")

    response = client.get("/disease/Unknown___Class")
    assert response.status_code == 404


def test_list_diseases_returns_sorted_names(client, monkeypatch, tmp_path):
    import json

    monkeypatch.setattr(api_module, "KNOWLEDGE_BASE_DIR", tmp_path)
    (tmp_path / "disease_info.json").write_text(
        json.dumps({"Tomato___healthy": {}, "Apple___Apple_scab": {}}), encoding="utf-8"
    )

    response = client.get("/diseases")
    assert response.status_code == 200
    assert response.json() == ["Apple___Apple_scab", "Tomato___healthy"]
