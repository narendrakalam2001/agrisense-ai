"""Grad-CAM heatmap generation for the EfficientNetB3 disease classifier.

Grad-CAM highlights which region of a leaf image most influenced the
model's prediction — used both in the notebook for qualitative model
checks and live in the Streamlit app / FastAPI ``/detect`` response so
farmers can see *why* a disease was flagged.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from src.config import IMG_SIZE, PLOTS_DIR, configure_logging
from src.model_training import find_base_model_layer, ModelBuildError

logger = configure_logging(__name__)


class GradCAMError(Exception):
    """Raised when a Grad-CAM heatmap cannot be produced."""


def find_last_conv_layer(model: tf.keras.Model, base_layer_name: str | None = None) -> str:
    """Locate the name of the last convolutional layer inside the base model.

    Args:
        model: The full AgriSense model (see ``model_training.build_model``).
        base_layer_name: Unused — kept for backward compatibility. The base
            model is now found by type (via ``find_base_model_layer``)
            rather than by name, since Keras 3 does not reliably honor a
            layer rename after construction.

    Returns:
        The name of the last ``Conv2D``-family layer found in the base.

    Raises:
        GradCAMError: If the base model or a convolutional layer can't be found.
    """
    try:
        base_model = find_base_model_layer(model)
    except ModelBuildError as exc:
        raise GradCAMError(str(exc)) from exc

    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.name.lower():
            logger.info("Using '%s' as the last conv layer for Grad-CAM.", layer.name)
            return layer.name

    raise GradCAMError("No convolutional layer found in the base model.")


def generate_gradcam_heatmap(
    model: tf.keras.Model,
    image_tensor: tf.Tensor,
    class_index: int,
    base_layer_name: str | None = None,
    last_conv_layer_name: str | None = None,
) -> np.ndarray:
    """Compute a Grad-CAM heatmap for a single image and target class.

    Args:
        model: Trained AgriSense model.
        image_tensor: A ``(1, H, W, 3)`` float32 tensor (pixel values in
            ``[0, 255]``, matching what ``build_model`` expects).
        class_index: Index of the predicted (or target) class to explain.
        base_layer_name: Unused — kept for backward compatibility.
        last_conv_layer_name: Override for the conv layer to hook into. If
            ``None``, it is auto-detected via :func:`find_last_conv_layer`.

    Returns:
        A 2D numpy array of shape ``(h, w)`` with values in ``[0, 1]``.

    Raises:
        GradCAMError: If gradients cannot be computed or the target layer
            is invalid.
    """
    base_model = find_base_model_layer(model)
    conv_layer_name = last_conv_layer_name or find_last_conv_layer(model)

    try:
        conv_layer = base_model.get_layer(conv_layer_name)
    except ValueError as exc:
        raise GradCAMError(f"Conv layer '{conv_layer_name}' not found in base model.") from exc

    grad_model = tf.keras.models.Model(
        inputs=base_model.inputs,
        outputs=[conv_layer.output, base_model.output],
    )

    with tf.GradientTape() as tape:
        conv_output, base_output = grad_model(image_tensor)
        tape.watch(conv_output)

        x = base_output
        for layer in model.layers[model.layers.index(base_model) + 1 :]:
            x = layer(x)
        predictions = x
        target_score = predictions[:, class_index]

    grads = tape.gradient(target_score, conv_output)
    if grads is None:
        raise GradCAMError(
            "Gradient computation returned None — check that the target "
            "layer is part of the differentiable path to the output."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_heatmap_on_image(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Resize and overlay a Grad-CAM heatmap onto the original leaf image.

    Args:
        original_image: The original image as a ``(H, W, 3)`` uint8 RGB array.
        heatmap: Output of :func:`generate_gradcam_heatmap`, values in ``[0, 1]``.
        alpha: Blend strength of the heatmap overlay (0 = invisible, 1 = opaque).
        colormap: OpenCV colormap constant.

    Returns:
        A ``(H, W, 3)`` uint8 RGB image with the heatmap overlaid.
    """
    h, w = original_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlaid = cv2.addWeighted(
        original_image.astype(np.uint8), 1 - alpha, heatmap_colored_rgb, alpha, 0
    )
    return overlaid


def save_gradcam_overlay(
    overlaid_image: np.ndarray, output_path: Path, class_name: str
) -> Path:
    """Save a Grad-CAM overlay image to disk.

    Args:
        overlaid_image: Output of :func:`overlay_heatmap_on_image`.
        output_path: Destination path (parent dirs auto-created). If a
            directory is given instead of a file, a filename is derived
            from ``class_name``.
        class_name: Disease class name, used to derive a filename if needed.

    Returns:
        The final path the image was written to.
    """
    if output_path.is_dir() or output_path.suffix == "":
        output_path = output_path / f"gradcam_{class_name}.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bgr_image = cv2.cvtColor(overlaid_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr_image)
    logger.info("Saved Grad-CAM overlay to '%s'.", output_path)
    return output_path