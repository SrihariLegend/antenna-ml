"""Dataset loading, validation, and preprocessing for the antenna-ml pipeline."""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of dataset validation."""

    valid: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def load_dataset(path: str = config.DATASET_PATH) -> pd.DataFrame:
    """Load the antenna dataset from a CSV file.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the file is empty or cannot be parsed.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Ensure the dataset CSV is in the working directory."
        )
    except pd.errors.EmptyDataError:
        raise ValueError(f"Dataset file '{path}' is empty.")
    except pd.errors.ParserError as exc:
        raise ValueError(f"Failed to parse '{path}': {exc}") from exc

    logger.info("Loaded dataset: %d rows, %d columns from '%s'", len(df), len(df.columns), path)
    return df


def validate_dataset(df: pd.DataFrame) -> ValidationResult:
    """Validate the dataset structure and content.

    Checks:
    - Correct number of columns
    - No duplicate column names
    - All columns are numeric
    - No unexpected NaN values
    - Frequency values within expected range (with tolerance)

    Args:
        df: DataFrame to validate.

    Returns:
        ValidationResult with valid flag, warnings, and errors lists.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Check required feature and target columns exist
    required = config.FEATURE_COLUMNS + config.TARGET_COLUMNS
    actual = list(df.columns)
    missing = [c for c in required if c not in actual]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    # Check for unexpected columns
    required_set = set(required)
    unexpected = [c for c in df.columns if c not in required_set]
    if unexpected:
        errors.append(f"Unexpected columns found: {unexpected}")

    # Check for duplicate column names
    seen: set[str] = set()
    duplicates: list[str] = []
    for col in df.columns:
        if col in seen:
            duplicates.append(col)
        seen.add(col)
    if duplicates:
        warnings.append(f"Duplicate column names detected: {duplicates}")

    # Check for non-numeric columns
    non_numeric = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        errors.append(f"Non-numeric columns found: {non_numeric}")

    # Check for NaN values (iterate positionally to avoid ambiguity with duplicate column names)
    nan_cols = [col for i, col in enumerate(df.columns) if df.iloc[:, i].isna().any()]
    if nan_cols:
        errors.append(f"NaN values found in columns: {nan_cols}")

    # Check frequency column if it exists
    if config.FREQUENCY_COL in df.columns:
        tolerance = 0.5
        freq_min = df[config.FREQUENCY_COL].min()
        freq_max = df[config.FREQUENCY_COL].max()
        if freq_min < config.FREQ_MIN - tolerance or freq_max > config.FREQ_MAX + tolerance:
            warnings.append(
                f"Frequency range [{freq_min:.2f}, {freq_max:.2f}] GHz is outside "
                f"expected [{config.FREQ_MIN}, {config.FREQ_MAX}] GHz (±{tolerance} tolerance)."
            )
    else:
        errors.append(f"Frequency column '{config.FREQUENCY_COL}' not found in dataset.")

    valid = len(errors) == 0
    for warning in warnings:
        logger.warning("Dataset validation: %s", warning)
    for error in errors:
        logger.error("Dataset validation: %s", error)

    return ValidationResult(valid=valid, warnings=warnings, errors=errors)


def validate_uploaded_dataset(df: pd.DataFrame) -> ValidationResult:
    """Stricter validation for user-uploaded CSVs.

    Checks: frequency column exists, all numeric, no NaN, min rows.
    Returns ValidationResult with all detected errors (not fail-fast).
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Check frequency column exists
    if config.FREQUENCY_COL not in df.columns:
        errors.append(
            f"Missing required column '{config.FREQUENCY_COL}'."
        )
        logger.error("Upload validation: missing '%s' column.", config.FREQUENCY_COL)

    # Check all columns are numeric
    non_numeric = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        errors.append(f"Non-numeric columns found: {non_numeric}")
        logger.error("Upload validation: non-numeric columns %s.", non_numeric)

    # Check for NaN values
    nan_cols = [col for col in df.columns if df[col].isna().any()]
    if nan_cols:
        errors.append(f"NaN values found in columns: {nan_cols}")
        logger.error("Upload validation: NaN values in columns %s.", nan_cols)

    # Check minimum row count
    if len(df) < config.MIN_DATASET_ROWS:
        errors.append(
            f"Dataset has {len(df)} rows, minimum required is {config.MIN_DATASET_ROWS}."
        )
        logger.error(
            "Upload validation: only %d rows, need at least %d.",
            len(df),
            config.MIN_DATASET_ROWS,
        )

    # Require at least one target column beyond frequency
    target_cols = [c for c in df.columns if c != config.FREQUENCY_COL]
    if not target_cols:
        errors.append(
            f"Dataset must contain at least one target column besides '{config.FREQUENCY_COL}'."
        )
        logger.error("Upload validation: no target columns found.")

    valid = len(errors) == 0
    if valid:
        logger.info("Upload validation passed: %d rows, %d columns.", len(df), len(df.columns))
    else:
        logger.warning("Upload validation failed with %d error(s).", len(errors))

    return ValidationResult(valid=valid, warnings=warnings, errors=errors)


def prepare_features_targets(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract feature array, target array, and target column names from the dataset.

    Args:
        df: Validated DataFrame.

    Returns:
        Tuple of (X, y, target_columns) where:
            X: Feature array of shape (n_samples, n_features).
            y: Target array of shape (n_samples, n_targets).
            target_columns: List of target column names.

    Raises:
        KeyError: If required columns are not present in the DataFrame.
    """
    # Use explicit feature/target columns from config if available,
    # otherwise fall back to frequency-only for uploaded datasets.
    feature_cols = getattr(config, "FEATURE_COLUMNS", [config.FREQUENCY_COL])
    target_cols = getattr(config, "TARGET_COLUMNS", None)

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise KeyError(
            f"Feature columns {missing_features} not found. "
            f"Available columns: {list(df.columns)}"
        )

    if target_cols:
        # Use the intersection of configured target columns and the actual
        # columns present in the dataframe.  This lets the function work with
        # both the full augmented dataset (all 7 targets) and the bare CST CSV
        # (3 targets) without raising an error.
        target_columns = [c for c in target_cols if c in df.columns]
        if not target_columns:
            # None of the configured targets are present — fall back to all
            # non-feature columns.
            target_columns = [col for col in df.columns if col not in feature_cols]
        elif len(target_columns) < len(target_cols):
            missing_targets = [c for c in target_cols if c not in df.columns]
            logger.warning(
                "Some configured target columns are absent from this dataset and will "
                "be skipped: %s.  Using available targets: %s",
                missing_targets,
                target_columns,
            )
    else:
        target_columns = [col for col in df.columns if col not in feature_cols]

    X = df[feature_cols].values
    y = df[target_columns].values

    logger.info(
        "Features: %s, shape %s. Targets (%d): %s, shape %s.",
        feature_cols,
        X.shape,
        len(target_columns),
        target_columns,
        y.shape,
    )
    return X, y, target_columns


def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Split data into train/test sets and scale the feature array.

    The scaler is fit only on the training data to prevent data leakage.

    Args:
        X: Feature array of shape (n_samples, 1).
        y: Target array of shape (n_samples, n_targets).

    Returns:
        Dictionary with keys:
            X_train_scaled, X_test_scaled, y_train, y_test, scaler.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info(
        "Train samples: %d, test samples: %d.", len(X_train), len(X_test)
    )
    return {
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }
