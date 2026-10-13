"""Tests for services/detection_service.py. TensorFlow model loading is
mocked so tests don't need a real trained .h5 file; PIL/numpy are used for
lightweight image fixtures."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("tensorflow")
from PIL import Image

from services.detection_service import DetectionService, DetectionServiceError


def _make_test_image_bytes(size=(50, 50)) -> bytes:
    img = Image.new("RGB", size, color=(100, 150, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_detection_service_raises_if_model_file_missing(tmp_path: Path):
    missing_model_path = tmp_path / "does_not_exist.h5"
    with pytest.raises(DetectionServiceError):
        DetectionService(model_path=missing_model_path, class_names=["a", "b"])


@patch("services.detection_service.tf.keras.models.load_model")
def test_detection_service_detect_returns_top3(mock_load_model, tmp_path: Path):
    fake_model_path = tmp_path / "model.h5"
    fake_model_path.write_bytes(b"fake")

    mock_model = MagicMock()
    # 4-class fake softmax output, class index 2 is the winner.
    mock_model.predict.return_value = np.array([[0.05, 0.10, 0.70, 0.15]])
    mock_load_model.return_value = mock_model

    class_names = ["Apple___healthy", "Apple___scab", "Tomato___Late_blight", "Tomato___healthy"]
    service = DetectionService(model_path=fake_model_path, class_names=class_names)

    result = service.detect(_make_test_image_bytes(), include_gradcam=False)

    assert result.predicted_class == "Tomato___Late_blight"
    assert result.confidence == pytest.approx(0.70)
    assert len(result.top_3) == 3
    assert result.gradcam_image_base64 is None


@patch("services.detection_service.tf.keras.models.load_model")
def test_detection_service_raises_on_invalid_image_bytes(mock_load_model, tmp_path: Path):
    fake_model_path = tmp_path / "model.h5"
    fake_model_path.write_bytes(b"fake")
    mock_load_model.return_value = MagicMock()

    service = DetectionService(model_path=fake_model_path, class_names=["a", "b"])
    with pytest.raises(DetectionServiceError):
        service.detect(b"not a real image", include_gradcam=False)


@patch("services.detection_service.tf.keras.models.load_model")
def test_detection_service_result_to_dict(mock_load_model, tmp_path: Path):
    fake_model_path = tmp_path / "model.h5"
    fake_model_path.write_bytes(b"fake")
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.9, 0.1]])
    mock_load_model.return_value = mock_model

    service = DetectionService(model_path=fake_model_path, class_names=["healthy", "diseased"])
    result = service.detect(_make_test_image_bytes(), include_gradcam=False)
    result_dict = result.to_dict()

    assert result_dict["predicted_class"] == "healthy"
    assert "top_3" in result_dict


@patch("services.detection_service.tf.keras.models.load_model")
def test_detection_service_falls_back_to_config_class_names_when_no_model_card(
    mock_load_model, tmp_path: Path
):
    fake_model_path = tmp_path / "model.h5"
    fake_model_path.write_bytes(b"fake")
    mock_load_model.return_value = MagicMock()

    missing_card_path = tmp_path / "no_such_model_card.json"
    service = DetectionService(model_path=fake_model_path)
    # No class_names passed and no model card on disk -> falls back to config CLASS_NAMES.
    from src.config import CLASS_NAMES

    names = service._load_class_names_from_model_card(model_card_path=missing_card_path)
    assert names == CLASS_NAMES


@patch("services.detection_service.tf.keras.models.load_model")
def test_detection_service_reads_class_names_from_model_card(mock_load_model, tmp_path: Path):
    import json

    fake_model_path = tmp_path / "model.h5"
    fake_model_path.write_bytes(b"fake")
    mock_load_model.return_value = MagicMock()

    model_card_path = tmp_path / "model_card.json"
    discovered_names = ["Tomato_Late_blight", "Tomato_healthy"]
    model_card_path.write_text(json.dumps({"class_names": discovered_names}), encoding="utf-8")

    service = DetectionService(model_path=fake_model_path)
    names = service._load_class_names_from_model_card(model_card_path=model_card_path)
    assert names == discovered_names
