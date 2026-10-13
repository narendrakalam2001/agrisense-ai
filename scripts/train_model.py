"""End-to-end training entry point.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --phase1-epochs 5 --phase2-epochs 10
    python scripts/train_model.py --raw-data-dir "D:\\Data_Science_Datasets\\agrisense-ai\\PlantVillage"

Runs: dataset validation -> build datasets -> augment -> two-phase training
-> evaluation -> Grad-CAM samples -> model card -> TFLite export.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/train_model.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf

from src.config import (
    MODEL_CARD_PATH,
    MODEL_H5_PATH,
    MODEL_TFLITE_PATH,
    PHASE1_EPOCHS,
    PHASE2_EPOCHS,
    PLOTS_DIR,
    RAW_DATA_DIR,
    configure_logging,
)
from src.data_loader import DatasetError, build_datasets
from src.grad_cam import (
    GradCAMError,
    generate_gradcam_heatmap,
    overlay_heatmap_on_image,
    save_gradcam_overlay,
)
from src.model_evaluation import evaluate_model, plot_training_curves, write_model_card
from src.model_training import export_tflite, train_two_phase
from src.preprocessing import apply_augmentation, build_augmentation_pipeline

logger = configure_logging(__name__, log_filename="train_model.log")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training run."""
    parser = argparse.ArgumentParser(description="Train the AgriSense AI disease classifier.")
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help=(
            "Path to the folder that directly contains the 38 class "
            "sub-folders (e.g. .../PlantVillage). Overrides AGRISENSE_DATA_DIR "
            "for this run only. Default: %(default)s"
        ),
    )
    parser.add_argument("--phase1-epochs", type=int, default=PHASE1_EPOCHS)
    parser.add_argument("--phase2-epochs", type=int, default=PHASE2_EPOCHS)
    parser.add_argument(
        "--skip-gradcam-samples",
        action="store_true",
        help="Skip generating sample Grad-CAM overlays after training.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the full Day 1 pipeline. Returns process exit code (0 = success)."""
    args = parse_args()

    try:
        logger.info("Loading datasets from '%s'...", args.raw_data_dir)
        train_ds, val_ds, test_ds, class_names = build_datasets(raw_dir=args.raw_data_dir)
    except DatasetError as exc:
        logger.error("Dataset error: %s", exc)
        return 1

    logger.info("Applying augmentation to training split...")
    augmentation = build_augmentation_pipeline()
    train_ds = apply_augmentation(train_ds, augmentation, training=True)

    logger.info("Starting two-phase training...")
    model, history = train_two_phase(
        train_ds,
        val_ds,
        num_classes=len(class_names),
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
    )

    logger.info("Plotting training curves...")
    plot_training_curves(history.combined())

    logger.info("Evaluating on held-out test set...")
    sample_batch = None
    for images, _ in test_ds.take(1):
        sample_batch = images[:1]
        break

    evaluation = evaluate_model(model, test_ds, class_names, sample_batch_for_latency=sample_batch)

    logger.info("Writing model card...")
    write_model_card(evaluation, class_names)

    logger.info("Exporting TFLite model...")
    export_tflite(model, MODEL_TFLITE_PATH)

    if not args.skip_gradcam_samples and sample_batch is not None:
        logger.info("Generating a sample Grad-CAM overlay...")
        try:
            predicted_class = int(tf.argmax(model.predict(sample_batch, verbose=0)[0]))
            heatmap = generate_gradcam_heatmap(model, sample_batch, predicted_class)
            original = tf.cast(sample_batch[0], tf.uint8).numpy()
            overlaid = overlay_heatmap_on_image(original, heatmap)
            save_gradcam_overlay(
                overlaid, PLOTS_DIR / "sample_gradcam.png", class_names[predicted_class]
            )
        except GradCAMError as exc:
            logger.warning("Skipping Grad-CAM sample generation: %s", exc)

    logger.info(
        "Pipeline complete. Model: %s | Accuracy: %.4f | Meets targets: acc=%s f1=%s latency=%s",
        MODEL_H5_PATH,
        evaluation.overall_accuracy,
        evaluation.meets_accuracy_target,
        evaluation.meets_f1_target,
        evaluation.meets_latency_target,
    )
    logger.info("Model card written to '%s'.", MODEL_CARD_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
