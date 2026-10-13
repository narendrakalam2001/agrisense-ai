"""Tests for src/model_training.py. Requires TensorFlow; skipped automatically
in environments where it isn't installed (e.g. lightweight CI runners without
GPU/TF pinned yet — see .github/workflows/ci.yml for the full-dependency job)."""

from __future__ import annotations

import pytest

tf = pytest.importorskip("tensorflow")

from src.model_training import ModelBuildError, build_model, compile_model, find_base_model_layer, unfreeze_top_layers


def test_build_model_raises_on_non_positive_num_classes():
    with pytest.raises(ModelBuildError):
        build_model(num_classes=0)


def test_build_model_output_shape_matches_num_classes():
    model = build_model(num_classes=5, img_size=(64, 64))
    assert model.output_shape[-1] == 5


def test_build_model_base_frozen_by_default():
    model = build_model(num_classes=5, img_size=(64, 64), freeze_base=True)
    base = find_base_model_layer(model)
    assert base.trainable is False


def test_compile_model_sets_optimizer_and_loss():
    model = build_model(num_classes=3, img_size=(64, 64))
    compiled = compile_model(model, learning_rate=1e-3)
    assert compiled.optimizer is not None
    assert compiled.loss == "categorical_crossentropy"


def test_unfreeze_top_layers_raises_without_base_layer():
    inputs = tf.keras.layers.Input(shape=(10,))
    outputs = tf.keras.layers.Dense(2)(inputs)
    plain_model = tf.keras.Model(inputs, outputs)

    with pytest.raises(ModelBuildError):
        unfreeze_top_layers(plain_model)


def test_unfreeze_top_layers_makes_base_trainable():
    model = build_model(num_classes=3, img_size=(64, 64), freeze_base=True)
    unfreeze_top_layers(model, n_layers=5)
    base = find_base_model_layer(model)
    assert base.trainable is True
