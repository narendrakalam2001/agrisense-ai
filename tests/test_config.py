"""Tests for src/config.py."""

from __future__ import annotations

import pytest

from src.config import (
    CLASS_NAMES,
    IMG_SIZE,
    NUM_CLASSES,
    MissingAPIKeyError,
    require_api_key,
)


def test_class_names_count_matches_num_classes():
    assert len(CLASS_NAMES) == NUM_CLASSES


def test_class_names_are_unique():
    assert len(CLASS_NAMES) == len(set(CLASS_NAMES))


def test_img_size_is_positive_tuple():
    assert len(IMG_SIZE) == 2
    assert all(dim > 0 for dim in IMG_SIZE)


def test_require_api_key_returns_value_when_present():
    assert require_api_key("SOME_KEY", "abc123") == "abc123"


def test_require_api_key_raises_when_missing():
    with pytest.raises(MissingAPIKeyError):
        require_api_key("SOME_KEY", None)


def test_require_api_key_raises_when_empty_string():
    with pytest.raises(MissingAPIKeyError):
        require_api_key("SOME_KEY", "")


def test_missing_api_key_error_message_contains_key_name():
    err = MissingAPIKeyError("GEMINI_API_KEY")
    assert "GEMINI_API_KEY" in str(err)
