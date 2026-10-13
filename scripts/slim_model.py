"""Create a smaller, serving-only copy of the trained model (no optimizer state).

Why rebuild instead of ``model.save(include_optimizer=False)``?
    A model that was *loaded* from a legacy ``.h5`` cannot be re-saved by Keras 3.12
    (``TypeError: cannot pickle 'module' object``). So this script loads the trained
    model, builds a fresh copy of the same architecture with ``build_model()``, copies
    the weights layer by layer, and saves that fresh model. The result has no Adam
    optimizer state (~40 % smaller), and GitHub blocks files > 100 MB.

Usage:
    python scripts/slim_model.py

Writes models/efficientnet_plantvillage_slim.h5 and verifies that its predictions are
identical to the original. Nothing is overwritten.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

from src.model_training import build_model

SRC = ROOT / "models" / "efficientnet_plantvillage.h5"
DST = ROOT / "models" / "efficientnet_plantvillage_slim.h5"


def size_mb(path: Path) -> float:
    """File size in MB."""
    return path.stat().st_size / (1024 * 1024)


def copy_weights(source: tf.keras.Model, target: tf.keras.Model) -> int:
    """Copy weights layer by layer (recursing into the nested EfficientNetB3 base).

    Layer-by-layer copying avoids relying on the global ``get_weights()`` order, which
    depends on which layers are trainable.

    Args:
        source: Model that holds the trained weights.
        target: Freshly built model with the same architecture.

    Returns:
        Number of weight tensors copied.

    Raises:
        ValueError: If the two architectures do not match.
    """
    if len(source.layers) != len(target.layers):
        raise ValueError(f"Layer count differs: {len(source.layers)} vs {len(target.layers)}")

    copied = 0
    for src_layer, dst_layer in zip(source.layers, target.layers):
        if getattr(src_layer, "layers", None):
            copied += copy_weights(src_layer, dst_layer)
            continue
        weights = src_layer.get_weights()
        if not weights:
            continue
        if [w.shape for w in weights] != [w.shape for w in dst_layer.get_weights()]:
            raise ValueError(f"Weight shapes differ for layer '{src_layer.name}' / '{dst_layer.name}'")
        dst_layer.set_weights(weights)
        copied += len(weights)
    return copied


def main() -> int:
    """Slim the model and verify it. Returns a process exit code."""
    if not SRC.exists():
        print(f"Model not found: {SRC}")
        return 1

    original = tf.keras.models.load_model(str(SRC), compile=False, safe_mode=False)
    num_classes = int(original.output_shape[-1])

    fresh = build_model(num_classes=num_classes, freeze_base=True)
    copied = copy_weights(original, fresh)
    print(f"Copied {copied} weight tensors ({num_classes} classes).")

    fresh.save(str(DST))
    slim = tf.keras.models.load_model(str(DST), compile=False, safe_mode=False)

    x = np.random.RandomState(0).uniform(0, 255, size=(2, 300, 300, 3)).astype("float32")
    max_diff = float(np.max(np.abs(original.predict(x, verbose=0) - slim.predict(x, verbose=0))))
    print(f"original: {size_mb(SRC):.1f} MB | slim: {size_mb(DST):.1f} MB | max prediction diff: {max_diff:.2e}")

    if max_diff > 1e-5:
        print("WARNING: predictions differ - do NOT replace the original model.")
        return 2

    print("OK - identical predictions. Next: move the ORIGINAL .h5 outside the repo, then rename")
    print("      efficientnet_plantvillage_slim.h5 -> efficientnet_plantvillage.h5")
    return 0


if __name__ == "__main__":
    sys.exit(main())