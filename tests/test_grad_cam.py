"""Tests for src/grad_cam.py. Requires TensorFlow; skipped automatically
if unavailable in the current environment."""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
cv2 = pytest.importorskip("cv2")

from src.grad_cam import GradCAMError, find_last_conv_layer, overlay_heatmap_on_image
from src.model_training import build_model


def test_find_last_conv_layer_raises_without_nested_base_model():
    inputs = tf.keras.layers.Input(shape=(10,))
    outputs = tf.keras.layers.Dense(2)(inputs)
    plain_model = tf.keras.Model(inputs, outputs)

    with pytest.raises(GradCAMError):
        find_last_conv_layer(plain_model)


def test_find_last_conv_layer_finds_a_layer():
    model = build_model(num_classes=3, img_size=(64, 64))
    layer_name = find_last_conv_layer(model)
    assert isinstance(layer_name, str)
    assert len(layer_name) > 0


def test_overlay_heatmap_on_image_preserves_dimensions():
    original = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
    heatmap = np.random.rand(8, 8).astype(np.float32)

    overlaid = overlay_heatmap_on_image(original, heatmap)
    assert overlaid.shape == original.shape
    assert overlaid.dtype == np.uint8
