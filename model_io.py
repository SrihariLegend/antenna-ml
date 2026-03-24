"""Model persistence with integrity checks for the antenna-ml pipeline."""

import logging
import os

import joblib
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

import config

logger = logging.getLogger(__name__)


def _artifact_paths(prefix: str) -> dict[str, str]:
    """Return artifact file paths for a given prefix.

    If prefix is in MODEL_FILES, use those paths.
    Otherwise, generate paths: rf_antenna_model_{prefix}.pkl, etc.

    Args:
        prefix: Model variant key (e.g. "base", "tuned", or a dataset-specific name).

    Returns:
        Dict with keys "model", "scaler", "target_columns" mapping to file paths.
    """
    if prefix in config.MODEL_FILES:
        return config.MODEL_FILES[prefix]

    return {
        "model": f"rf_antenna_model_{prefix}.pkl",
        "scaler": f"scaler_X_{prefix}.pkl",
        "target_columns": f"target_columns_{prefix}.pkl",
    }


def save_model(
    model: BaseEstimator,
    scaler: StandardScaler,
    target_columns: list[str],
    prefix: str = "base",
) -> None:
    """Persist model artifacts to disk.

    Args:
        model: Trained scikit-learn estimator.
        scaler: Fitted StandardScaler for the frequency feature.
        target_columns: List of target parameter names.
        prefix: Model variant key — "base", "tuned", or a dataset-specific name.

    Raises:
        OSError: If writing any artifact to disk fails.
    """
    files = _artifact_paths(prefix)
    artifacts = {
        files["model"]: model,
        files["scaler"]: scaler,
        files["target_columns"]: target_columns,
    }
    for path, obj in artifacts.items():
        try:
            joblib.dump(obj, path)
            logger.info("Saved artifact: %s", path)
        except OSError as exc:
            raise OSError(f"Failed to write artifact '{path}': {exc}") from exc


def load_model(
    prefix: str = "base",
) -> tuple[BaseEstimator, StandardScaler, list[str]]:
    """Load model artifacts from disk with basic integrity checks.

    Args:
        prefix: Model variant key — "base", "tuned", or a dataset-specific name.

    Returns:
        Tuple of (model, scaler, target_columns).

    Raises:
        FileNotFoundError: If any artifact file is missing.
        ValueError: If loaded objects fail integrity checks.
    """
    files = _artifact_paths(prefix)

    for path in files.values():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model artifact not found: '{path}'. "
                "Run train_model.py (or tune_hyperparameters.py for tuned model) first."
            )

    try:
        model = joblib.load(files["model"])
        scaler = joblib.load(files["scaler"])
        target_columns = joblib.load(files["target_columns"])
    except Exception as exc:
        raise ValueError(f"Failed to load model artifacts for prefix '{prefix}': {exc}") from exc

    # Integrity checks
    if not hasattr(model, "predict"):
        raise ValueError(f"Loaded model '{files['model']}' does not have a predict() method.")
    if not hasattr(scaler, "transform"):
        raise ValueError(f"Loaded scaler '{files['scaler']}' does not have a transform() method.")
    if not isinstance(target_columns, list) or not all(
        isinstance(c, str) for c in target_columns
    ):
        raise ValueError(
            f"Loaded target_columns '{files['target_columns']}' is not a list of strings."
        )

    logger.info(
        "Loaded %s model: %d target columns.", prefix, len(target_columns)
    )
    return model, scaler, target_columns


def find_best_model() -> str:
    """Return the prefix of the best available model variant.

    Prefers the tuned model over the base model, then checks for
    dataset-specific prefixes. Checks that all three artifact files
    for the chosen variant exist.

    Returns:
        The prefix of the best available model variant.

    Raises:
        FileNotFoundError: If no complete set of model artifacts is found.
    """
    # Check well-known prefixes first (tuned preferred over base)
    for prefix in ("tuned", "base"):
        files = _artifact_paths(prefix)
        if all(os.path.exists(p) for p in files.values()):
            logger.info("Found model variant: %s", prefix)
            return prefix

    # Check dataset-specific prefixes from the registry
    try:
        import dataset_registry

        for name in dataset_registry.list_datasets():
            prefix = name
            files = _artifact_paths(prefix)
            if all(os.path.exists(p) for p in files.values()):
                logger.info("Found dataset-specific model variant: %s", prefix)
                return prefix
    except Exception as exc:
        logger.debug("Could not search dataset-specific models: %s", exc)

    raise FileNotFoundError(
        "No trained model found. Run train_model.py first."
    )
