"""Tests for src/data_loader.py (dataset-independent helper functions)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tensorflow")  # src.data_loader imports tf at module level

from src.data_loader import (
    DatasetError,
    DatasetStats,
    compute_class_distribution,
    find_presplit_dirs,
    save_split_manifest,
    validate_raw_dataset,
)


def test_find_presplit_dirs_returns_none_for_flat_layout(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "Tomato___healthy").mkdir()
    assert find_presplit_dirs(raw_dir) is None


def test_find_presplit_dirs_detects_train_valid(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    (raw_dir / "train").mkdir(parents=True)
    (raw_dir / "valid").mkdir()
    result = find_presplit_dirs(raw_dir)
    assert result == (raw_dir / "train", raw_dir / "valid")


def test_find_presplit_dirs_detects_train_val(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    (raw_dir / "train").mkdir(parents=True)
    (raw_dir / "val").mkdir()
    result = find_presplit_dirs(raw_dir)
    assert result == (raw_dir / "train", raw_dir / "val")


def test_find_presplit_dirs_none_when_only_train_exists(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    (raw_dir / "train").mkdir(parents=True)
    assert find_presplit_dirs(raw_dir) is None


def test_validate_raw_dataset_checks_inside_train_for_presplit_layout(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    (raw_dir / "train" / "Tomato___healthy").mkdir(parents=True)
    (raw_dir / "valid" / "Tomato___healthy").mkdir(parents=True)
    validate_raw_dataset(raw_dir)  # should not raise


def test_validate_raw_dataset_raises_on_missing_dir(tmp_path: Path):
    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(DatasetError):
        validate_raw_dataset(missing_dir)


def test_validate_raw_dataset_raises_on_empty_dir(tmp_path: Path):
    empty_dir = tmp_path / "raw"
    empty_dir.mkdir()
    with pytest.raises(DatasetError):
        validate_raw_dataset(empty_dir)


def test_validate_raw_dataset_passes_with_class_folders(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "Tomato___healthy").mkdir()
    (raw_dir / "Tomato___Late_blight").mkdir()
    validate_raw_dataset(raw_dir)  # should not raise


def test_compute_class_distribution_counts_files(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    class_dir = raw_dir / "Tomato___healthy"
    class_dir.mkdir(parents=True)
    (class_dir / "img1.jpg").write_bytes(b"fake")
    (class_dir / "img2.jpg").write_bytes(b"fake")

    distribution = compute_class_distribution(raw_dir)
    assert distribution["Tomato___healthy"] == 2


def test_dataset_stats_to_dict_roundtrip():
    stats = DatasetStats(split_name="train", num_images=100, class_counts={"a": 50, "b": 50})
    d = stats.to_dict()
    assert d["split_name"] == "train"
    assert d["num_images"] == 100
    assert d["class_counts"] == {"a": 50, "b": 50}


def test_save_split_manifest_writes_json(tmp_path: Path):
    stats = [DatasetStats("train", 10, {}), DatasetStats("val", 2, {})]
    output_path = tmp_path / "manifest.json"
    save_split_manifest(stats, output_path)
    assert output_path.exists()

    import json
    content = json.loads(output_path.read_text())
    assert len(content) == 2
    assert content[0]["split_name"] == "train"
