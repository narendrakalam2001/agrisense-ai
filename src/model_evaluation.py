"""Model evaluation utilities: confusion matrix, per-class metrics, model card.

All plots are saved under ``PLOTS_DIR`` (never gitignored — see project
README standard) so they can be embedded directly in the README and in the
model card.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)

from src.config import (
    MODEL_CARD_PATH,
    NUM_CLASSES,
    PLOTS_DIR,
    TARGET_ACCURACY,
    TARGET_F1_PER_CLASS,
    TARGET_INFERENCE_SECONDS,
    configure_logging,
)

logger = configure_logging(__name__)


class EvaluationError(Exception):
    """Raised when evaluation cannot proceed (e.g. empty test set)."""


@dataclass
class EvaluationReport:
    """Structured evaluation results, ready to serialize into the model card."""

    overall_accuracy: float
    top5_accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_f1: dict[str, float]
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    mean_inference_seconds: float
    meets_accuracy_target: bool
    meets_f1_target: bool
    meets_latency_target: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _collect_predictions(
    model: tf.keras.Model, test_ds: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over the full test set and collect labels/probabilities.

    Args:
        model: Trained model to evaluate.
        test_ds: Batched ``(images, one_hot_labels)`` test dataset.

    Returns:
        Tuple of ``(y_true, y_pred, y_proba)`` as numpy arrays, where
        ``y_true``/``y_pred`` are integer class indices and ``y_proba`` is
        the full ``(N, num_classes)`` probability matrix.

    Raises:
        EvaluationError: If the test dataset yields no batches.
    """
    y_true_batches: list[np.ndarray] = []
    y_proba_batches: list[np.ndarray] = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_proba_batches.append(probs)
        y_true_batches.append(labels.numpy())

    if not y_true_batches:
        raise EvaluationError("Test dataset produced zero batches — cannot evaluate.")

    y_true_onehot = np.concatenate(y_true_batches, axis=0)
    y_proba = np.concatenate(y_proba_batches, axis=0)
    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(y_proba, axis=1)
    return y_true, y_pred, y_proba


def measure_inference_latency(
    model: tf.keras.Model, sample_batch: tf.Tensor, n_runs: int = 20
) -> float:
    """Measure mean single-image CPU inference latency.

    Args:
        model: Trained model.
        sample_batch: A single-image batch, shape ``(1, H, W, 3)``.
        n_runs: Number of repeated inferences to average over.

    Returns:
        Mean inference time in seconds.
    """
    # Warm-up run (excludes graph-tracing overhead from the measurement).
    _ = model.predict(sample_batch, verbose=0)

    durations = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict(sample_batch, verbose=0)
        durations.append(time.perf_counter() - start)

    return float(np.mean(durations))


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: Path = PLOTS_DIR / "confusion_matrix.png",
) -> Path:
    """Plot and save a 38x38 confusion matrix heatmap.

    Args:
        y_true: Ground-truth integer class indices.
        y_pred: Predicted integer class indices.
        class_names: Ordered list of class names matching the indices.
        output_path: Where to save the PNG.

    Returns:
        The path the plot was saved to.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("AgriSense AI — Confusion Matrix (38 classes)")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix plot to '%s'.", output_path)
    return output_path


def plot_training_curves(
    history: dict[str, list[float]],
    output_path: Path = PLOTS_DIR / "training_curves.png",
) -> Path:
    """Plot combined accuracy/loss curves across both training phases.

    Args:
        history: Combined history dict (see ``TrainingHistory.combined()``).
        output_path: Where to save the PNG.

    Returns:
        The path the plot was saved to.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.get("accuracy", []), label="train_accuracy")
    axes[0].plot(history.get("val_accuracy", []), label="val_accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.get("loss", []), label="train_loss")
    axes[1].plot(history.get("val_loss", []), label="val_loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved training curves plot to '%s'.", output_path)
    return output_path


def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: list[str],
    sample_batch_for_latency: tf.Tensor | None = None,
) -> EvaluationReport:
    """Run full evaluation: accuracy, top-5, per-class F1/precision/recall,
    confusion matrix plot, and latency check.

    Args:
        model: Trained model.
        test_ds: Batched test dataset.
        class_names: Ordered class names.
        sample_batch_for_latency: Optional single-image batch for latency
            measurement. If omitted, latency is not measured (set to 0.0).

    Returns:
        A populated :class:`EvaluationReport`.
    """
    y_true, y_pred, y_proba = _collect_predictions(model, test_ds)

    overall_accuracy = float(np.mean(y_true == y_pred))
    top5_accuracy = float(
        top_k_accuracy_score(y_true, y_proba, k=5, labels=list(range(len(class_names))))
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class_f1 = {c: report[c]["f1-score"] for c in class_names}
    per_class_precision = {c: report[c]["precision"] for c in class_names}
    per_class_recall = {c: report[c]["recall"] for c in class_names}
    macro_f1 = float(report["macro avg"]["f1-score"])
    weighted_f1 = float(report["weighted avg"]["f1-score"])

    plot_confusion_matrix(y_true, y_pred, class_names)

    mean_latency = 0.0
    if sample_batch_for_latency is not None:
        mean_latency = measure_inference_latency(model, sample_batch_for_latency)

    min_class_f1 = min(per_class_f1.values()) if per_class_f1 else 0.0

    evaluation = EvaluationReport(
        overall_accuracy=overall_accuracy,
        top5_accuracy=top5_accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class_f1=per_class_f1,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        mean_inference_seconds=mean_latency,
        meets_accuracy_target=overall_accuracy >= TARGET_ACCURACY,
        meets_f1_target=min_class_f1 >= TARGET_F1_PER_CLASS,
        meets_latency_target=(
            mean_latency <= TARGET_INFERENCE_SECONDS if sample_batch_for_latency is not None else True
        ),
    )

    logger.info(
        "Evaluation complete — accuracy=%.4f, top5=%.4f, macro_f1=%.4f, "
        "min_per_class_f1=%.4f, mean_inference=%.3fs",
        overall_accuracy,
        top5_accuracy,
        macro_f1,
        min_class_f1,
        mean_latency,
    )
    return evaluation


def write_model_card(
    evaluation: EvaluationReport,
    class_names: list[str],
    output_path: Path = MODEL_CARD_PATH,
    extra_metadata: dict | None = None,
) -> Path:
    """Write a model card JSON documenting architecture, metrics, and targets.

    Args:
        evaluation: Result of :func:`evaluate_model`.
        class_names: Ordered class names the model predicts.
        output_path: Destination JSON file.
        extra_metadata: Optional additional fields (e.g. training duration,
            dataset version) to merge into the card.

    Returns:
        The path the model card was saved to.
    """
    card = {
        "model_name": "AgriSense AI — EfficientNetB3 Crop Disease Classifier",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": "EfficientNetB3 (ImageNet pretrained, two-phase fine-tuned)",
        "num_classes": NUM_CLASSES,
        "class_names": class_names,
        "evaluation": evaluation.to_dict(),
        "targets": {
            "accuracy": TARGET_ACCURACY,
            "min_per_class_f1": TARGET_F1_PER_CLASS,
            "max_inference_seconds": TARGET_INFERENCE_SECONDS,
        },
    }
    if extra_metadata:
        card.update(extra_metadata)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2, ensure_ascii=False)
    logger.info("Wrote model card to '%s'.", output_path)
    return output_path
