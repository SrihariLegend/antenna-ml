"""Centralized configuration and constants for the antenna-ml pipeline."""

import logging
import sys

# --- Dataset ---
DATASET_PATH = "clean_dataset_rect.csv"
FREQUENCY_COL = "freq"
FREQ_MIN = 1.5
FREQ_MAX = 10.5

# Feature columns (model inputs) and target columns (model outputs).
# The model takes only the operating frequency as input and predicts all
# antenna geometry parameters + S11 (multi-output regression).
FEATURE_COLUMNS = ["freq"]
TARGET_COLUMNS = [
    "S11",
    "patch_length",
    "width of patch in mm",
    "substrate_height",
    "substrate_length",
    "substrate_width",
    "effective_er",
]

EXPECTED_COLUMNS = FEATURE_COLUMNS + TARGET_COLUMNS

# Geometry search space for optimization (from CST simulation grid)
PATCH_LENGTH_RANGE = (8.0, 34.11)    # mm
SUBSTRATE_HEIGHT_VALUES = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2]  # mm (CST + interpolated)

# --- Train / test split ---
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Model artifact file paths ---
MODEL_FILES = {
    "base": {
        "model": "rf_antenna_model.pkl",
        "scaler": "scaler_X.pkl",
        "target_columns": "target_columns.pkl",
    },
    "tuned": {
        "model": "rf_antenna_model_tuned.pkl",
        "scaler": "scaler_X_tuned.pkl",
        "target_columns": "target_columns_tuned.pkl",
    },
}

# --- Default Random Forest hyperparameters ---
DEFAULT_RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": 1,
}

# --- GridSearchCV parameter grid ---
PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}

# --- Gradio app ---
GRADIO_HOST = "0.0.0.0"
GRADIO_PORT = 7860

# --- Output ---
VISUALIZATION_PATH = "prediction_results.png"


# --- Dataset management ---
DATASETS_DIR = "datasets"                          # directory for uploaded CSVs
DATASET_REGISTRY_PATH = "dataset_registry.json"    # registry JSON file
MIN_DATASET_ROWS = 10                              # minimum rows for validation
DEFAULT_DATASET_NAME = "Patch Antenna (CST Simulated)"             # name for the bundled dataset

# --- Dimension constraint parameter names and units ---
CONSTRAINT_PARAMS = {
    "patch_length": "mm",
    "patch_width": "mm",
    "substrate_height": "mm",
    "substrate_width": "mm",
    "substrate_length": "mm",
}

# --- PyInstaller ---
PYINSTALLER_ENTRY = "gradio_app.py"                # entry point for the executable
PYINSTALLER_NAME = "antenna-ml"                     # output executable name
PYINSTALLER_BUNDLE_DATA = [                         # extra data files to bundle
    ("augmented_dataset_clean.csv", "."),
    ("clean_dataset_rect.csv", "."),
    ("dataset_registry.json", "."),
    ("datasets", "datasets"),
]


def setup_logging(level: int = logging.INFO, log_file: str | None = None) -> None:
    """Configure root logger with a consistent format.

    Args:
        level: Logging level (e.g. logging.INFO, logging.DEBUG).
        log_file: Optional path to write log output to a file in addition to stdout.
    """
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
