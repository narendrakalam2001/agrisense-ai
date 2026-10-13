"""Detection service layer.

Wraps the trained EfficientNetB3 model as a singleton service: load once at
startup, expose a simple ``detect(image_bytes)`` call that both the FastAPI
``/detect`` endpoint and the Streamlit app can use without duplicating
model-loading or Grad-CAM logic.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import CLASS_NAMES, IMG_SIZE, MODEL_CARD_PATH, MODEL_H5_PATH, configure_logging
from src.grad_cam import (
    GradCAMError,
    generate_gradcam_heatmap,
    overlay_heatmap_on_image,
)

logger = configure_logging(__name__)


class DetectionServiceError(Exception):
    """Raised when detection cannot be performed (bad image, missing model)."""


@dataclass
class ClassPrediction:
    """A single class's prediction, used for the top-3 breakdown."""

    class_name: str
    confidence: float

    def to_dict(self) -> dict:
        return {"class_name": self.class_name, "confidence": self.confidence}


@dataclass
class DetectionResult:
    """Full result of a single disease-detection inference call."""

    predicted_class: str
    confidence: float
    top_3: list[ClassPrediction]
    gradcam_image_base64: str | None

    def to_dict(self) -> dict:
        return {
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "top_3": [p.to_dict() for p in self.top_3],
            "gradcam_image_base64": self.gradcam_image_base64,
        }


class DetectionService:
    """Loads the trained model once and serves predictions + Grad-CAM overlays."""

    def __init__(self, model_path=MODEL_H5_PATH, class_names: list[str] | None = None) -> None:
        """Load the trained model from disk.

        Args:
            model_path: Path to the trained ``.h5`` model.
            class_names: Ordered class names matching the model's output
                layer. If ``None`` (the default), this is read from
                ``models/model_card.json`` — the class list actually
                discovered on disk at training time (see
                ``src.data_loader.build_datasets``), which is what the
                model was really trained against. Falls back to the
                ``CLASS_NAMES`` constant in ``src.config`` only if no model
                card is found, e.g. before Day 1 training has ever run.

        Raises:
            DetectionServiceError: If the model file doesn't exist or fails to load.
        """
        if not model_path.exists():
            raise DetectionServiceError(
                f"Model file not found at '{model_path}'. Run scripts/train_model.py first."
            )

        try:
            self.model = tf.keras.models.load_model(model_path, safe_mode=False)
        except Exception as exc:  # noqa: BLE001
            raise DetectionServiceError(f"Failed to load model from '{model_path}': {exc}") from exc

        self.class_names = class_names or self._load_class_names_from_model_card()
        logger.info("Loaded model from '%s' with %d classes.", model_path, len(self.class_names))

    @staticmethod
    def _load_class_names_from_model_card(model_card_path=MODEL_CARD_PATH) -> list[str]:
        """Read the class list actually used at training time, if available.

        Args:
            model_card_path: Path to ``model_card.json`` written by
                ``src.model_evaluation.write_model_card``.

        Returns:
            The ``class_names`` list from the model card, or the
            ``CLASS_NAMES`` constant from ``src.config`` if the model card
            doesn't exist or can't be parsed (e.g. training hasn't run yet).
        """
        if not model_card_path.exists():
            logger.warning(
                "No model card found at '%s' — falling back to the CLASS_NAMES "
                "constant in src/config.py. If your dataset's folder-naming "
                "convention differs from that constant, predictions will be "
                "mislabeled. Run scripts/train_model.py to generate a model "
                "card with the actual discovered class names.",
                model_card_path,
            )
            return CLASS_NAMES

        try:
            card = json.loads(model_card_path.read_text(encoding="utf-8"))
            class_names = card["class_names"]
            if not isinstance(class_names, list) or not class_names:
                raise ValueError("model_card.json 'class_names' is missing or empty.")
            return class_names
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning(
                "Could not read class_names from '%s' (%s) — falling back to "
                "the CLASS_NAMES constant in src/config.py.",
                model_card_path, exc,
            )
            return CLASS_NAMES

    def _decode_image(self, image_bytes: bytes) -> tuple[tf.Tensor, np.ndarray]:
        """Decode raw image bytes into a model-ready tensor and a display array.

        Args:
            image_bytes: Raw uploaded file bytes (JPG/PNG).

        Returns:
            Tuple of ``(batch_tensor, original_uint8_array)`` where
            ``batch_tensor`` has shape ``(1, H, W, 3)`` and
            ``original_uint8_array`` has shape ``(H, W, 3)`` for Grad-CAM overlay.

        Raises:
            DetectionServiceError: If the bytes cannot be decoded as an image.
        """
        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            pil_image = pil_image.resize(IMG_SIZE)
        except Exception as exc:  # noqa: BLE001
            raise DetectionServiceError(f"Could not decode image: {exc}") from exc

        array = np.array(pil_image).astype("float32")
        batch = np.expand_dims(array, axis=0)
        return tf.convert_to_tensor(batch), np.array(pil_image).astype("uint8")

    def detect(self, image_bytes: bytes, include_gradcam: bool = True) -> DetectionResult:
        """Run disease detection on an uploaded image.

        Args:
            image_bytes: Raw uploaded file bytes (JPG/PNG).
            include_gradcam: Whether to compute and return a Grad-CAM overlay
                (adds latency — disable for lightweight/batch calls).

        Returns:
            A populated :class:`DetectionResult`.

        Raises:
            DetectionServiceError: If the image is invalid or inference fails.
        """
        batch_tensor, original_array = self._decode_image(image_bytes)

        probabilities = self.model.predict(batch_tensor, verbose=0)[0]
        top_indices = np.argsort(probabilities)[::-1][:3]

        top_3 = [
            ClassPrediction(class_name=self.class_names[i], confidence=float(probabilities[i]))
            for i in top_indices
        ]
        predicted_index = int(top_indices[0])
        predicted_class = self.class_names[predicted_index]
        confidence = float(probabilities[predicted_index])

        gradcam_b64 = None
        if include_gradcam:
            try:
                heatmap = generate_gradcam_heatmap(self.model, batch_tensor, predicted_index)
                overlaid = overlay_heatmap_on_image(original_array, heatmap)
                gradcam_b64 = self._encode_image_base64(overlaid)
            except GradCAMError as exc:
                logger.warning("Grad-CAM generation failed, continuing without it: %s", exc)

        logger.info("Detected '%s' with confidence %.4f.", predicted_class, confidence)
        return DetectionResult(
            predicted_class=predicted_class,
            confidence=confidence,
            top_3=top_3,
            gradcam_image_base64=gradcam_b64,
        )

    @staticmethod
    def _encode_image_base64(image_array: np.ndarray) -> str:
        """Encode a uint8 RGB array as a base64 PNG data string.

        Args:
            image_array: ``(H, W, 3)`` uint8 array.

        Returns:
            Base64-encoded PNG bytes as a UTF-8 string (no data-URI prefix).
        """
        pil_image = Image.fromarray(image_array)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
