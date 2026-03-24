"""Centralized configuration and constants for the antenna-ml pipeline."""

import logging
import sys

# --- Dataset ---
DATASET_PATH = "dataset_WIFI7.csv"
FREQUENCY_COL = "Frequency(GHz)"
# Expanded frequency range (was 2.0–7.0)
FREQ_MIN = 1.0
FREQ_MAX = 10.0

# Expected columns in the dataset (in order).
# Note: the second "length of patch in mm" was renamed to "width of patch in mm".
EXPECTED_COLUMNS = [
    "Frequency(GHz)",
    "length of patch in mm",
    "width of patch in mm",
    "length of substrate in mm",
    "width of Substrate in mm",
    "Area of Slots(mm^2)",
    "Radiaus of Circular Slot(mm)",
    "S11(dB)",
]

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
DEFAULT_DATASET_NAME = "WiFi7 Default"             # name for the bundled dataset

# --- Dimension constraint parameter names and units ---
CONSTRAINT_PARAMS = {
    "length of patch in mm": "mm",
    "width of patch in mm": "mm",
    "length of substrate in mm": "mm",
    "width of Substrate in mm": "mm",
    "Area of Slots(mm^2)": "mm²",
    "Radiaus of Circular Slot(mm)": "mm",
    "S11(dB)": "dB",
}

# --- PyInstaller ---
PYINSTALLER_ENTRY = "gradio_app.py"                # entry point for the executable
PYINSTALLER_NAME = "antenna-ml"                     # output executable name
PYINSTALLER_BUNDLE_DATA = [                         # extra data files to bundle
    ("dataset_WIFI7.csv", "."),
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
