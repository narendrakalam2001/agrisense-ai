"""Image preprocessing and augmentation pipeline.

Builds a ``tf.keras`` augmentation stack applied only to the training split
(validation/test data is only rescaled, never augmented) as specified in the
project's model training standard: horizontal/vertical flips, rotation up
to ±15°, zoom 0.9-1.1, brightness/contrast jitter, and random crop.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

from src.config import IMG_SIZE, configure_logging

logger = configure_logging(__name__)

ROTATION_FACTOR = 15 / 360  # keras RandomRotation takes a fraction of 2*pi... see note below
ZOOM_RANGE = (-0.1, 0.1)  # maps to 0.9x - 1.1x zoom
BRIGHTNESS_FACTOR = 0.15
CONTRAST_FACTOR = 0.15
CROP_FRACTION = 0.9  # random crop keeps 90% of the image before resizing back


def build_augmentation_pipeline(img_size: tuple[int, int] = IMG_SIZE) -> tf.keras.Sequential:
    """Construct the training-time data augmentation stack.

    Note on rotation: ``layers.RandomRotation`` expects its factor as a
    fraction of a full rotation (2*pi radians), so ±15 degrees corresponds
    to a factor of ``15/360``.

    Args:
        img_size: Target (height, width) images are resized back to after
            the random crop step.

    Returns:
        A ``tf.keras.Sequential`` model that can be called on a batch of
        images (float32, any range) and returns augmented images of the
        same shape.
    """
    pipeline = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(ROTATION_FACTOR),
            layers.RandomZoom(ZOOM_RANGE),
            layers.RandomBrightness(BRIGHTNESS_FACTOR),
            layers.RandomContrast(CONTRAST_FACTOR),
            layers.RandomCrop(
                int(img_size[0] * CROP_FRACTION), int(img_size[1] * CROP_FRACTION)
            ),
            layers.Resizing(img_size[0], img_size[1]),
        ],
        name="augmentation_pipeline",
    )
    logger.info(
        "Built augmentation pipeline (rotation=±15°, zoom=%s, "
        "brightness=±%.2f, contrast=±%.2f, crop_fraction=%.2f).",
        ZOOM_RANGE,
        BRIGHTNESS_FACTOR,
        CONTRAST_FACTOR,
        CROP_FRACTION,
    )
    return pipeline


def build_rescaling_layer() -> layers.Rescaling:
    """EfficientNet-style rescaling.

    Keras' ``EfficientNetB3`` application expects raw pixel values in
    ``[0, 255]`` — its own ``preprocess_input`` handles normalization
    internally — so this layer is a no-op passthrough kept for pipelines
    that need an explicit rescale step for other models.

    Returns:
        A ``layers.Rescaling`` layer with scale factor 1.0 (identity).
    """
    return layers.Rescaling(scale=1.0)


def apply_augmentation(
    dataset: tf.data.Dataset, augmentation: tf.keras.Sequential, training: bool = True
) -> tf.data.Dataset:
    """Map the augmentation pipeline over a batched image/label dataset.

    Args:
        dataset: A batched ``(images, labels)`` dataset.
        augmentation: Pipeline built by :func:`build_augmentation_pipeline`.
        training: If ``False``, the dataset is returned unmodified — use
            this for validation/test splits.

    Returns:
        The augmented dataset (or the original dataset if ``training`` is
        ``False``).
    """
    if not training:
        return dataset

    def _augment(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return augmentation(images, training=True), labels

    return dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)


def preprocess_single_image(image_bytes: bytes, img_size: tuple[int, int] = IMG_SIZE) -> tf.Tensor:
    """Decode and resize a single uploaded image for inference.

    Used by the FastAPI ``/detect`` endpoint and the Streamlit app to turn a
    raw uploaded file into a model-ready tensor.

    Args:
        image_bytes: Raw bytes of an uploaded JPG/PNG file.
        img_size: Target (height, width).

    Returns:
        A ``(1, H, W, 3)`` float32 tensor ready to feed to the model.

    Raises:
        tf.errors.InvalidArgumentError: If ``image_bytes`` cannot be decoded
            as an image.
    """
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, img_size)
    image = tf.expand_dims(image, axis=0)
    return image
