"""Data loading utilities for the PlantVillage crop-disease dataset.

The PlantVillage / "New Plant Diseases" Kaggle datasets are distributed as
class-named folders of JPG images:

    data/raw/
        Apple___Apple_scab/
            image1.jpg
            image2.jpg
        Apple___Black_rot/
            ...

This module builds reproducible train/val/test ``tf.data.Dataset`` pipelines
directly from that folder structure, validates image integrity, and reports
class distribution so imbalance can be spotted before training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from src.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    IMG_SIZE,
    NUM_CLASSES,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SEED,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
    configure_logging,
)

logger = configure_logging(__name__)


class DatasetError(Exception):
    """Raised for any unrecoverable problem with the raw dataset."""


@dataclass
class DatasetStats:
    """Summary statistics for a loaded split, used for logging and reports."""

    split_name: str
    num_images: int
    class_counts: dict[str, int]

    def to_dict(self) -> dict:
        """Serialize to a plain dict (for JSON dumping)."""
        return {
            "split_name": self.split_name,
            "num_images": self.num_images,
            "class_counts": self.class_counts,
        }


def find_presplit_dirs(raw_dir: Path) -> tuple[Path, Path] | None:
    """Detect a Kaggle-style pre-split dataset layout under ``raw_dir``.

    Kaggle's "New Plant Diseases Dataset (Augmented)" (and similar mirrors)
    ship as ``<raw_dir>/train/<38 class folders>`` and
    ``<raw_dir>/valid/<38 class folders>`` rather than one flat folder of
    class sub-folders. Pointing ``--raw-data-dir`` straight at that dataset's
    root should use its real train/valid split instead of re-splitting
    randomly (avoids wasting the provided validation set and avoids any
    train/val leakage risk).

    Args:
        raw_dir: Candidate dataset root.

    Returns:
        ``(train_dir, valid_dir)`` if both exist as directories under
        ``raw_dir`` (checking ``valid`` then ``val`` as the second name),
        else ``None`` — meaning ``raw_dir`` should be treated as a flat
        folder of class sub-folders instead.
    """
    train_dir = raw_dir / "train"
    if not train_dir.is_dir():
        return None

    for valid_name in ("valid", "val"):
        valid_dir = raw_dir / valid_name
        if valid_dir.is_dir():
            return train_dir, valid_dir

    return None


def validate_raw_dataset(raw_dir: Path = RAW_DATA_DIR) -> None:
    """Verify the raw dataset directory has the expected class structure.

    Transparently handles both a flat folder of class sub-folders and a
    Kaggle-style pre-split ``train/`` + ``valid/`` layout (see
    :func:`find_presplit_dirs`) — in the pre-split case, the class folders
    inside ``train/`` are what's validated.

    Args:
        raw_dir: Root directory containing one sub-folder per class, OR a
            directory containing ``train/`` and ``valid/`` (or ``val/``)
            sub-directories that themselves each contain one sub-folder per
            class.

    Raises:
        DatasetError: If the directory is missing, empty, or none of the
            expected class folders are present.
    """
    if not raw_dir.exists():
        raise DatasetError(
            f"Raw data directory not found at '{raw_dir}'. Download the "
            "PlantVillage dataset from Kaggle "
            "(kaggle.com/datasets/emmarex/plantdisease) and extract it here, "
            "or set AGRISENSE_DATA_DIR to point at your dataset location."
        )

    presplit = find_presplit_dirs(raw_dir)
    class_folder_root = presplit[0] if presplit else raw_dir

    found_classes = {p.name for p in class_folder_root.iterdir() if p.is_dir()}
    expected_classes = set(CLASS_NAMES)
    missing = expected_classes - found_classes
    unexpected = found_classes - expected_classes

    if not found_classes:
        raise DatasetError(f"No class folders found under '{class_folder_root}'.")

    if missing:
        logger.warning(
            "Raw dataset is missing %d expected class folder(s): %s",
            len(missing),
            sorted(missing),
        )
    if unexpected:
        logger.warning(
            "Raw dataset has %d unexpected folder(s) not in CLASS_NAMES: %s",
            len(unexpected),
            sorted(unexpected),
        )

    logger.info(
        "Validated raw dataset at '%s'%s: %d class folders found.",
        raw_dir,
        " (pre-split train/valid layout)" if presplit else "",
        len(found_classes),
    )


def scan_corrupt_images(raw_dir: Path = RAW_DATA_DIR) -> list[Path]:
    """Walk every image and flag files that fail to open (corrupt/truncated).

    This is intentionally I/O-heavy and should be run once (e.g. during EDA)
    rather than on every training run.

    Args:
        raw_dir: Root directory containing one sub-folder per class.

    Returns:
        List of paths that could not be opened as valid images.
    """
    corrupt: list[Path] = []
    image_paths = list(raw_dir.rglob("*.[jJ][pP][gG]")) + list(
        raw_dir.rglob("*.[jJ][pP][eE][gG]")
    ) + list(raw_dir.rglob("*.[pP][nN][gG]"))

    logger.info("Scanning %d images for corruption...", len(image_paths))
    for path in image_paths:
        try:
            with Image.open(path) as img:
                img.verify()
        except (UnidentifiedImageError, OSError) as exc:
            logger.warning("Corrupt image detected: %s (%s)", path, exc)
            corrupt.append(path)

    logger.info("Corruption scan complete: %d corrupt file(s) found.", len(corrupt))
    return corrupt


def compute_class_distribution(raw_dir: Path = RAW_DATA_DIR) -> dict[str, int]:
    """Count images per class folder.

    Args:
        raw_dir: Root directory containing one sub-folder per class.

    Returns:
        Mapping of class name to image count, sorted by class name.
    """
    counts: dict[str, int] = {}
    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        n = sum(1 for _ in class_dir.glob("*") if _.is_file())
        counts[class_dir.name] = n
    return dict(sorted(counts.items()))


def build_datasets(
    raw_dir: Path = RAW_DATA_DIR,
    img_size: tuple[int, int] = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Build train/val/test ``tf.data.Dataset`` pipelines from raw images.

    Auto-detects the dataset layout:

    - **Pre-split** (Kaggle's "New Plant Diseases Dataset (Augmented)" and
      similar): ``raw_dir/train/<class folders>`` +
      ``raw_dir/valid/<class folders>`` (or ``val/``) both exist -> the
      provided train set is used as-is, and the provided valid set is split
      ~50/50 into validation and test (see :func:`find_presplit_dirs`).
    - **Flat** (plain PlantVillage layout): ``raw_dir`` directly contains one
      folder per class -> split internally at ``TRAIN_SPLIT`` /
      ``VAL_SPLIT`` / ``TEST_SPLIT`` proportions from ``src.config``.

    Labels are one-hot encoded categorical in both cases.

    Args:
        raw_dir: Dataset root — either a flat folder of class sub-folders,
            or a folder containing pre-split ``train/`` + ``valid/`` (or
            ``val/``) sub-directories.
        img_size: Target (height, width) for resizing.
        batch_size: Batch size for all three splits.
        seed: Random seed for the shuffle/split, for reproducibility.

    Returns:
        A tuple ``(train_ds, val_ds, test_ds, class_names)``. ``class_names``
        is auto-discovered from the actual sub-folder names on disk — it is
        NOT forced to match ``CLASS_NAMES`` in ``src.config``. Different
        Kaggle re-uploads of PlantVillage use inconsistent folder-naming
        conventions, and forcing an exact match here would make training
        crash on any mirror that doesn't match byte-for-byte. Downstream
        code (evaluation, model card, Grad-CAM, serving) all use this
        returned list, not the config constant, so training works
        regardless of which mirror's naming convention is on disk.

    Raises:
        DatasetError: If ``raw_dir`` is invalid (see :func:`validate_raw_dataset`),
            or if the number of discovered classes doesn't match
            ``NUM_CLASSES`` (a strong signal the wrong folder was pointed at).
    """
    validate_raw_dataset(raw_dir)

    presplit = find_presplit_dirs(raw_dir)
    if presplit:
        train_dir, valid_dir = presplit
        logger.info(
            "Detected pre-split dataset layout — using '%s' as train and "
            "'%s' as the source for val/test.", train_dir, valid_dir,
        )
        train_ds, val_ds, test_ds, class_names = _build_datasets_presplit(
            train_dir, valid_dir, img_size=img_size, batch_size=batch_size, seed=seed
        )
    else:
        train_ds, val_ds, test_ds, class_names = _build_datasets_flat(
            raw_dir, img_size=img_size, batch_size=batch_size, seed=seed
        )

    if len(class_names) != NUM_CLASSES:
        raise DatasetError(
            f"Discovered {len(class_names)} class folder(s) under '{raw_dir}', "
            f"expected {NUM_CLASSES}. Check that raw_dir points at the folder "
            f"that directly CONTAINS the 38 class sub-folders (not a parent "
            f"or child of it)."
        )

    if set(class_names) != set(CLASS_NAMES):
        logger.warning(
            "Discovered class folder names differ from the CLASS_NAMES "
            "convention in src/config.py (different Kaggle mirror naming, "
            "most likely). This is handled automatically — training, "
            "evaluation, the model card, and Grad-CAM all use the names "
            "actually discovered on disk. Only the disease_info.json / "
            "pesticide_database.json RAG lookup keys off the exact class "
            "name string, so if treatment lookups come back empty after "
            "training, that JSON's keys will need to be renamed to match "
            "what's on disk. Discovered names (first 3): %s",
            class_names[:3],
        )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    logger.info(
        "Built datasets — train batches: %s, val batches: %s, test batches: %s, "
        "classes: %d",
        tf.data.experimental.cardinality(train_ds).numpy(),
        tf.data.experimental.cardinality(val_ds).numpy(),
        tf.data.experimental.cardinality(test_ds).numpy(),
        len(class_names),
    )

    return train_ds, val_ds, test_ds, class_names


def _build_datasets_flat(
    raw_dir: Path, img_size: tuple[int, int], batch_size: int, seed: int
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Split a single flat folder of class sub-folders into train/val/test.

    Internal helper for :func:`build_datasets` — see its docstring.
    """
    # First carve off the test split, then split the remainder into
    # train/val. keras' validation_split only supports a single 2-way split,
    # so we call image_dataset_from_directory twice with complementary
    # subsets to obtain a 3-way split.
    val_and_test_fraction = VAL_SPLIT + TEST_SPLIT

    # Deliberately NOT passing class_names= here — let Keras auto-discover
    # class folders from disk (alphabetical order) rather than forcing an
    # exact match against the config.py constant.
    full_train = tf.keras.utils.image_dataset_from_directory(
        raw_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=val_and_test_fraction,
        subset="training",
    )
    val_test_combined = tf.keras.utils.image_dataset_from_directory(
        raw_dir,
        labels="inferred",
        label_mode="categorical",
        # Reuse the exact class list/order discovered above, so train and
        # val/test splits can't disagree on which index maps to which class.
        class_names=full_train.class_names,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=val_and_test_fraction,
        subset="validation",
    )

    # Split the combined val/test batches roughly proportionally.
    val_fraction_of_combined = VAL_SPLIT / val_and_test_fraction
    n_val_test_batches = tf.data.experimental.cardinality(val_test_combined).numpy()
    n_val_batches = max(1, int(round(n_val_test_batches * val_fraction_of_combined)))

    val_ds = val_test_combined.take(n_val_batches)
    test_ds = val_test_combined.skip(n_val_batches)

    return full_train, val_ds, test_ds, full_train.class_names


def _build_datasets_presplit(
    train_dir: Path, valid_dir: Path, img_size: tuple[int, int], batch_size: int, seed: int
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Use a Kaggle-style pre-split train/valid layout directly.

    The provided training set is used in full (no further splitting). The
    provided validation set is split roughly 50/50 into validation and test
    — the dataset doesn't ship a separate labeled test set, so this trades
    a little validation-set size for a genuine held-out test set that
    ``scripts/train_model.py`` never touches during training.

    Internal helper for :func:`build_datasets` — see its docstring.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    valid_full = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        labels="inferred",
        label_mode="categorical",
        # Force the same class order as the training set discovered above,
        # so a class index means the same thing in both splits.
        class_names=train_ds.class_names,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    n_valid_batches = tf.data.experimental.cardinality(valid_full).numpy()
    n_val_batches = max(1, n_valid_batches // 2)

    val_ds = valid_full.take(n_val_batches)
    test_ds = valid_full.skip(n_val_batches)

    return train_ds, val_ds, test_ds, train_ds.class_names


def save_split_manifest(
    stats: list[DatasetStats], output_path: Path = PROCESSED_DATA_DIR / "split_manifest.json"
) -> None:
    """Persist split statistics to disk for reproducibility and the README.

    Args:
        stats: One :class:`DatasetStats` per split (train/val/test).
        output_path: Destination JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [s.to_dict() for s in stats]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Saved split manifest to '%s'.", output_path)


if __name__ == "__main__":
    # Quick manual smoke test: `python -m src.data_loader`
    validate_raw_dataset()
    distribution = compute_class_distribution()
    logger.info("Class distribution: %s", distribution)
