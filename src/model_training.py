"""EfficientNetB3 transfer-learning pipeline for crop disease classification.

Two-phase strategy:
    Phase 1 — freeze the ImageNet-pretrained base, train only the new
        classification head (fast, stabilizes the head's random weights).
    Phase 2 — unfreeze the top ``UNFREEZE_LAST_N_LAYERS`` layers of the base
        and fine-tune the whole thing end-to-end at a much lower learning
        rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from src.config import (
    IMG_CHANNELS,
    IMG_SIZE,
    LOGS_DIR,
    MODEL_H5_PATH,
    NUM_CLASSES,
    PHASE1_EPOCHS,
    PHASE1_LR,
    PHASE2_EPOCHS,
    PHASE2_LR,
    UNFREEZE_LAST_N_LAYERS,
    configure_logging,
)
from src.preprocessing import build_augmentation_pipeline

logger = configure_logging(__name__)


class ModelBuildError(Exception):
    """Raised when the model architecture cannot be constructed."""


@dataclass
class TrainingHistory:
    """Combined Phase 1 + Phase 2 training history, for plotting/reporting."""

    phase1: dict[str, list[float]]
    phase2: dict[str, list[float]]

    def combined(self) -> dict[str, list[float]]:
        """Concatenate phase 1 and phase 2 metrics into single series."""
        merged: dict[str, list[float]] = {}
        for key in self.phase1:
            merged[key] = list(self.phase1[key]) + list(self.phase2.get(key, []))
        return merged


def build_model(
    num_classes: int = NUM_CLASSES,
    img_size: tuple[int, int] = IMG_SIZE,
    img_channels: int = IMG_CHANNELS,
    freeze_base: bool = True,
) -> tf.keras.Model:
    """Build an EfficientNetB3 classifier with a custom head.

    Args:
        num_classes: Number of output disease classes.
        img_size: Input (height, width).
        img_channels: Number of input channels (3 for RGB).
        freeze_base: Whether to freeze the EfficientNetB3 base initially
            (should be ``True`` for Phase 1, ``False`` when re-building for
            Phase 2 — though :func:`unfreeze_top_layers` is preferred over
            rebuilding).

    Returns:
        A compiled-ready (but not yet compiled) ``tf.keras.Model``.

    Raises:
        ModelBuildError: If ``num_classes`` is not a positive integer.
    """
    if num_classes <= 0:
        raise ModelBuildError(f"num_classes must be positive, got {num_classes}")

    input_shape = (img_size[0], img_size[1], img_channels)
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Note: EfficientNet's preprocess_input is a documented no-op (the
    # model's own first layer does rescaling internally), so there is no
    # separate preprocessing step here. An earlier version wrapped it in a
    # Lambda layer anyway — Keras 3's load_model() blocks Lambda layers by
    # default (safe_mode=True), which would have broken serving later.
    base_model = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling=None,
    )
    base_model.trainable = not freeze_base
    # Not renaming base_model here — Keras 3 does not reliably honor a
    # layer rename after construction. find_base_model_layer() below finds
    # it by type instead, so naming no longer matters.

    x = base_model(inputs, training=False if freeze_base else None)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_batchnorm")(x)
    x = layers.Dropout(0.3, name="head_dropout_1")(x)
    x = layers.Dense(512, activation="relu", name="head_dense_512")(x)
    x = layers.Dropout(0.2, name="head_dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="disease_predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="agrisense_efficientnetb3")
    logger.info(
        "Built model '%s' — base_trainable=%s, total_params=%d",
        model.name,
        base_model.trainable,
        model.count_params(),
    )
    return model


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    """Compile the model with Adam optimizer and categorical crossentropy.

    Args:
        model: The Keras model to compile (modified in place and returned).
        learning_rate: Learning rate for the Adam optimizer.

    Returns:
        The same model instance, now compiled.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def find_base_model_layer(model: tf.keras.Model) -> tf.keras.Model:
    """Find the nested EfficientNetB3 base model inside the full model.

    Looks it up by type (the only nested ``tf.keras.Model`` layer in this
    architecture) rather than by name, since Keras 3 does not reliably
    honor a layer rename after construction.

    Args:
        model: A model built by :func:`build_model`.

    Returns:
        The nested base model layer.

    Raises:
        ModelBuildError: If no (or more than one) nested model layer is found.
    """
    candidates = [layer for layer in model.layers if isinstance(layer, tf.keras.Model)]
    if len(candidates) != 1:
        raise ModelBuildError(
            f"Expected exactly one nested base model layer, found {len(candidates)}. "
            f"Was the model built with build_model()?"
        )
    return candidates[0]


def unfreeze_top_layers(model: tf.keras.Model, n_layers: int = UNFREEZE_LAST_N_LAYERS) -> tf.keras.Model:
    """Unfreeze the last ``n_layers`` of the EfficientNetB3 base for fine-tuning.

    Batch-normalization layers are deliberately kept frozen even within the
    unfrozen range, since fine-tuning BN statistics on a small batch size
    tends to destabilize training.

    Args:
        model: A model built by :func:`build_model`.
        n_layers: How many trailing layers of the base model to unfreeze.

    Returns:
        The same model instance with the base partially unfrozen.
    """
    base_model = find_base_model_layer(model)

    base_model.trainable = True
    freeze_until = max(0, len(base_model.layers) - n_layers)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    for layer in base_model.layers[freeze_until:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    logger.info(
        "Unfroze top %d layers of base model (%d/%d layers now trainable).",
        n_layers,
        trainable_count,
        len(base_model.layers),
    )
    return model

def build_callbacks(
    checkpoint_path: Path = MODEL_H5_PATH,
    tensorboard_subdir: str = "tensorboard",
    patience: int = 5,
) -> list[Callback]:
    """Build the standard callback stack: EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, TensorBoard.

    Args:
        checkpoint_path: Where to save the best model (by validation loss).
        tensorboard_subdir: Sub-folder of ``LOGS_DIR`` for TensorBoard logs.
        patience: Epochs with no val_loss improvement before early stopping.

    Returns:
        List of configured Keras callbacks.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tb_dir = LOGS_DIR / tensorboard_subdir
    tb_dir.mkdir(parents=True, exist_ok=True)

    return [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        TensorBoard(log_dir=str(tb_dir)),
    ]


def train_two_phase(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    num_classes: int = NUM_CLASSES,
    phase1_epochs: int = PHASE1_EPOCHS,
    phase2_epochs: int = PHASE2_EPOCHS,
) -> tuple[tf.keras.Model, TrainingHistory]:
    """Run the full Phase 1 (frozen base) + Phase 2 (fine-tune) training loop.

    Args:
        train_ds: Batched, augmented training dataset.
        val_ds: Batched validation dataset (no augmentation).
        num_classes: Number of output classes.
        phase1_epochs: Max epochs for the frozen-base phase.
        phase2_epochs: Max epochs for the fine-tuning phase.

    Returns:
        Tuple of ``(trained_model, TrainingHistory)``.
    """
    model = build_model(num_classes=num_classes, freeze_base=True)
    compile_model(model, learning_rate=PHASE1_LR)

    logger.info("=== Phase 1: training classification head (base frozen) ===")
    callbacks_phase1 = build_callbacks(tensorboard_subdir="tensorboard/phase1")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=callbacks_phase1,
        verbose=1,
    )

    logger.info("=== Phase 2: fine-tuning top layers of the base ===")
    unfreeze_top_layers(model, n_layers=UNFREEZE_LAST_N_LAYERS)
    compile_model(model, learning_rate=PHASE2_LR)
    callbacks_phase2 = build_callbacks(tensorboard_subdir="tensorboard/phase2")
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase2_epochs,
        callbacks=callbacks_phase2,
        verbose=1,
    )

    training_history = TrainingHistory(phase1=history1.history, phase2=history2.history)
    logger.info("Training complete. Best model saved to '%s'.", MODEL_H5_PATH)
    return model, training_history


def export_tflite(model: tf.keras.Model, output_path: Path) -> None:
    """Convert the trained Keras model to a mobile-ready TFLite model.

    Args:
        model: Trained Keras model.
        output_path: Destination ``.tflite`` file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    logger.info(
        "Exported TFLite model to '%s' (%.2f MB).",
        output_path,
        len(tflite_model) / (1024 * 1024),
    )
