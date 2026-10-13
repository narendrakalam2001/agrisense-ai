"""Central configuration for AgriSense AI.

Loads environment variables via python-dotenv and exposes strongly-typed
constants used across the data, training, RAG, chatbot, and serving layers.
Never hardcode secrets here — everything sensitive comes from the
environment (see .env.example).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

# Base data directory can be overridden via env var, e.g. on Windows:
# AGRISENSE_DATA_DIR=D:\Data Science Datasets\agrisense-ai
DATA_DIR: Final[Path] = Path(
    os.getenv("AGRISENSE_DATA_DIR", PROJECT_ROOT / "data")
)
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
KNOWLEDGE_BASE_DIR: Final[Path] = DATA_DIR / "knowledge_base"

MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
CHROMADB_DIR: Final[Path] = PROJECT_ROOT / "chromadb_store"
LOGS_DIR: Final[Path] = PROJECT_ROOT / "logs"
PLOTS_DIR: Final[Path] = PROJECT_ROOT / "plots"

for _dir in (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    KNOWLEDGE_BASE_DIR,
    MODELS_DIR,
    CHROMADB_DIR,
    LOGS_DIR,
    PLOTS_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)

MODEL_H5_PATH: Final[Path] = MODELS_DIR / "efficientnet_plantvillage.h5"
MODEL_TFLITE_PATH: Final[Path] = MODELS_DIR / "efficientnet_plantvillage.tflite"
MODEL_CARD_PATH: Final[Path] = MODELS_DIR / "model_card.json"

# ---------------------------------------------------------------------------
# Dataset / class configuration
# ---------------------------------------------------------------------------
# The 38 PlantVillage classes (folder names as distributed on Kaggle).
# Order matters — it defines the index<->label mapping used everywhere
# (training, evaluation, Grad-CAM, serving). Do not reorder after training.
CLASS_NAMES: Final[list[str]] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
    "Blueberry___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
]
NUM_CLASSES: Final[int] = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Image / training hyperparameters
# ---------------------------------------------------------------------------
IMG_SIZE: Final[tuple[int, int]] = (300, 300)  # EfficientNetB3 native input
IMG_CHANNELS: Final[int] = 3
BATCH_SIZE: Final[int] = int(os.getenv("AGRISENSE_BATCH_SIZE", "32"))
SEED: Final[int] = 42

TRAIN_SPLIT: Final[float] = 0.70
VAL_SPLIT: Final[float] = 0.15
TEST_SPLIT: Final[float] = 0.15

PHASE1_EPOCHS: Final[int] = int(os.getenv("AGRISENSE_PHASE1_EPOCHS", "10"))
PHASE2_EPOCHS: Final[int] = int(os.getenv("AGRISENSE_PHASE2_EPOCHS", "20"))
PHASE1_LR: Final[float] = 1e-3
PHASE2_LR: Final[float] = 1e-5
UNFREEZE_LAST_N_LAYERS: Final[int] = 30

TARGET_ACCURACY: Final[float] = 0.92
TARGET_F1_PER_CLASS: Final[float] = 0.85
TARGET_INFERENCE_SECONDS: Final[float] = 2.0

# ---------------------------------------------------------------------------
# External API keys (never hardcode — read from environment only)
# ---------------------------------------------------------------------------
GEMINI_API_KEY: Final[str | None] = os.getenv("GEMINI_API_KEY")
OPENWEATHERMAP_API_KEY: Final[str | None] = os.getenv("OPENWEATHERMAP_API_KEY")


class MissingAPIKeyError(Exception):
    """Raised when a required API key is absent from the environment.

    Note: deliberately NOT a frozen dataclass. Python sets ``__traceback__``
    on an exception instance when it's raised/caught, which fails against
    a frozen dataclass's ``__setattr__`` and masks the real error with a
    confusing ``FrozenInstanceError``.
    """

    def __init__(self, key_name: str) -> None:
        self.key_name = key_name
        self.message = (
            f"Required environment variable '{key_name}' is not set. "
            f"Copy .env.example to .env and fill it in."
        )
        super().__init__(self.message)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


def require_api_key(key_name: str, value: str | None) -> str:
    """Return ``value`` or raise :class:`MissingAPIKeyError` if it is empty.

    Args:
        key_name: Name of the environment variable, used in the error message.
        value: The value read from the environment.

    Returns:
        The non-empty API key string.

    Raises:
        MissingAPIKeyError: If ``value`` is ``None`` or an empty string.
    """
    if not value:
        raise MissingAPIKeyError(key_name)
    return value


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def configure_logging(name: str, log_filename: str | None = None) -> logging.Logger:
    """Create a module-level logger that writes to console and to ``logs/``.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        log_filename: Optional file name under ``LOGS_DIR``. Defaults to
            ``f"{name}.log"``.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured (e.g. re-imported in a notebook) — avoid dupes.
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    file_name = log_filename or f"{name.replace('.', '_')}.log"
    file_handler = logging.FileHandler(LOGS_DIR / file_name, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
