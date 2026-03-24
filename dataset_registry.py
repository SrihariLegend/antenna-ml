"""Dataset registry CRUD operations for the antenna-ml pipeline.

Manages a JSON-based registry that maps dataset names to file paths and metadata.
"""

import json
import logging
import os
from datetime import datetime, timezone

import pandas as pd

import config

logger = logging.getLogger(__name__)


def load_registry() -> dict:
    """Read and return the registry dict. Creates empty registry if file missing.

    Returns:
        Registry dict with a "datasets" key mapping names to metadata.
    """
    if not os.path.exists(config.DATASET_REGISTRY_PATH):
        logger.info("Registry file not found at '%s'; returning empty registry.", config.DATASET_REGISTRY_PATH)
        return {"datasets": {}}

    try:
        with open(config.DATASET_REGISTRY_PATH, "r") as f:
            registry = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read registry at '%s': %s. Returning empty registry.", config.DATASET_REGISTRY_PATH, exc)
        return {"datasets": {}}

    # Ensure the top-level key exists
    if "datasets" not in registry:
        registry["datasets"] = {}

    return registry


def save_registry(registry: dict) -> None:
    """Write registry dict to JSON file.

    Args:
        registry: Registry dict with a "datasets" key.
    """
    with open(config.DATASET_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    logger.info("Registry saved to '%s'.", config.DATASET_REGISTRY_PATH)


def list_datasets() -> list[str]:
    """Return sorted list of registered dataset names.

    Returns:
        Alphabetically sorted list of dataset name strings.
    """
    registry = load_registry()
    return sorted(registry["datasets"].keys())


def get_dataset_path(name: str) -> str:
    """Return file path for a named dataset.

    Args:
        name: Registered dataset name.

    Returns:
        File path string for the dataset CSV.

    Raises:
        KeyError: If the dataset name is not found in the registry.
    """
    registry = load_registry()
    if name not in registry["datasets"]:
        raise KeyError(f"Dataset '{name}' not found in registry.")
    return registry["datasets"][name]["path"]


def register_dataset(name: str, path: str, df: pd.DataFrame) -> None:
    """Add a dataset entry to the registry with metadata extracted from the DataFrame.

    Args:
        name: Human-readable dataset name.
        path: Relative path to the CSV file.
        df: DataFrame loaded from the CSV (used to extract metadata).

    Raises:
        ValueError: If a dataset with the given name already exists.
    """
    registry = load_registry()

    if name in registry["datasets"]:
        raise ValueError(f"Dataset '{name}' already exists in registry.")

    freq_values = df[config.FREQUENCY_COL] if config.FREQUENCY_COL in df.columns else pd.Series(dtype=float)

    registry["datasets"][name] = {
        "path": path,
        "columns": list(df.columns),
        "row_count": len(df),
        "freq_min": float(freq_values.min()) if len(freq_values) > 0 else 0.0,
        "freq_max": float(freq_values.max()) if len(freq_values) > 0 else 0.0,
        "added_at": datetime.now(timezone.utc).isoformat(),
    }

    save_registry(registry)
    logger.info("Registered dataset '%s' (%d rows) at '%s'.", name, len(df), path)


def init_default_dataset() -> None:
    """Ensure the default dataset is registered on first run.

    If the default dataset name is not already in the registry, registers it
    using the bundled dataset CSV path and loads metadata from the file.
    Creates the datasets directory if it does not exist.
    """
    os.makedirs(config.DATASETS_DIR, exist_ok=True)

    registry = load_registry()
    if config.DEFAULT_DATASET_NAME in registry["datasets"]:
        logger.info("Default dataset '%s' already registered.", config.DEFAULT_DATASET_NAME)
        return

    if not os.path.exists(config.DATASET_PATH):
        logger.warning(
            "Default dataset file '%s' not found; skipping default registration.",
            config.DATASET_PATH,
        )
        return

    df = pd.read_csv(config.DATASET_PATH)
    register_dataset(config.DEFAULT_DATASET_NAME, config.DATASET_PATH, df)
    logger.info("Default dataset '%s' initialized.", config.DEFAULT_DATASET_NAME)
