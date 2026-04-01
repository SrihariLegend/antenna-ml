"""Hyperparameter tuning for the antenna parameter prediction model."""

import logging
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

import config
import data_loader
import model_io

logger = logging.getLogger(__name__)


def tune_on_dataset(dataset_path: str, model_prefix: str) -> dict:
    """Run GridSearchCV hyperparameter tuning on a given dataset.

    Args:
        dataset_path: Path to the CSV file.
        model_prefix: Prefix for saving tuned model artifacts (suffixed with '_tuned').

    Returns:
        Dict with keys: test_r2, test_mse, test_mae, best_params, elapsed_seconds,
        per_param (list of dicts with r2, mae per column).

    Raises:
        ValueError: If dataset validation fails.
    """
    logger.info("=" * 60)
    logger.info("HYPERPARAMETER TUNING - ANTENNA PARAMETERS")
    logger.info("=" * 60)

    # Load and validate dataset
    df = data_loader.load_dataset(dataset_path)
    if dataset_path == config.DATASET_PATH:
        result = data_loader.validate_dataset(df)
    else:
        result = data_loader.validate_uploaded_dataset(df)
    if not result.valid:
        raise ValueError(f"Dataset validation failed: {result.errors}")

    logger.info(
        "Dataset: %d samples, frequency range %.2f – %.2f GHz",
        len(df),
        df[config.FREQUENCY_COL].min(),
        df[config.FREQUENCY_COL].max(),
    )

    X, y, target_columns = data_loader.prepare_features_targets(df)
    splits = data_loader.split_and_scale(X, y)
    X_train_scaled = splits["X_train_scaled"]
    X_test_scaled = splits["X_test_scaled"]
    y_train = splits["y_train"]
    y_test = splits["y_test"]
    scaler = splits["scaler"]

    # Log parameter grid
    total_combinations = int(np.prod([len(v) for v in config.PARAM_GRID.values()]))
    logger.info("=" * 60)
    logger.info("PARAMETER GRID (%d total combinations)", total_combinations)
    logger.info("=" * 60)
    for param, values in config.PARAM_GRID.items():
        logger.info("  %s: %s", param, values)
    logger.info("5-fold cross-validation — estimated 5-10 minutes")

    # Grid search
    logger.info("Starting grid search...")
    rf = RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=config.PARAM_GRID,
        cv=5,
        scoring="r2",
        verbose=2,
        n_jobs=-1,
        return_train_score=True,
    )

    start_time = time.time()
    y_train_fit = y_train.ravel() if y_train.ndim == 2 and y_train.shape[1] == 1 else y_train
    grid_search.fit(X_train_scaled, y_train_fit)
    elapsed = time.time() - start_time

    logger.info("=" * 60)
    logger.info("TUNING COMPLETE in %.1f minutes (%.1f seconds)", elapsed / 60, elapsed)
    logger.info("=" * 60)

    # Best parameters
    logger.info("BEST PARAMETERS")
    for param, value in grid_search.best_params_.items():
        logger.info("  %s: %s", param, value)
    logger.info("Best cross-validation R²: %.4f", grid_search.best_score_)

    # Evaluate on test set
    best_model = grid_search.best_estimator_
    y_pred_test = best_model.predict(X_test_scaled)
    # Ensure 2D for consistent indexing
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)
    if y_pred_test.ndim == 1:
        y_pred_test = y_pred_test.reshape(-1, 1)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    logger.info("TEST SET EVALUATION")
    logger.info("  R²: %.4f  |  MSE: %.4f  |  MAE: %.4f", test_r2, test_mse, test_mae)

    logger.info("PER-PARAMETER PERFORMANCE")
    for i, col in enumerate(target_columns):
        r2 = r2_score(y_test[:, i], y_pred_test[:, i])
        mae = mean_absolute_error(y_test[:, i], y_pred_test[:, i])
        logger.info("  %s — R²: %.4f  MAE: %.4f", col, r2, mae)

    # Top 10 parameter combinations
    logger.info("TOP 10 PARAMETER COMBINATIONS")
    results_df = pd.DataFrame(grid_search.cv_results_).sort_values("rank_test_score")
    for _, row in results_df.head(10).iterrows():
        logger.info(
            "  Rank %d — mean R²: %.4f ± %.4f (fit: %.2fs) | %s",
            int(row["rank_test_score"]),
            row["mean_test_score"],
            row["std_test_score"],
            row["mean_fit_time"],
            row["params"],
        )

    # Compare with base model
    logger.info("COMPARISON WITH DEFAULT PARAMETERS")
    try:
        original_model, _, _ = model_io.load_model(prefix="base")
        y_pred_original = original_model.predict(X_test_scaled)
        original_r2 = r2_score(y_test, y_pred_original)
        improvement = test_r2 - original_r2
        pct = (improvement / original_r2 * 100) if original_r2 != 0 else float("inf")
        logger.info("  Original model R²: %.4f", original_r2)
        logger.info("  Tuned model R²:    %.4f", test_r2)
        logger.info("  Improvement:       %.4f (%.2f%%)", improvement, pct)
    except FileNotFoundError:
        logger.warning("Base model not found — skipping comparison.")
    except Exception as exc:
        logger.warning("Could not load base model for comparison: %s", exc)

    # Save tuned model
    tuned_prefix = f"{model_prefix}_tuned" if model_prefix != "base" else "tuned"
    logger.info("Saving tuned model artifacts with prefix '%s'...", tuned_prefix)
    model_io.save_model(best_model, scaler, target_columns, prefix=tuned_prefix)

    logger.info("=" * 60)
    logger.info("TUNING COMPLETE!")
    logger.info("=" * 60)

    per_param = []
    for i, col in enumerate(target_columns):
        r2 = r2_score(y_test[:, i], y_pred_test[:, i])
        mae_val = mean_absolute_error(y_test[:, i], y_pred_test[:, i])
        per_param.append({"name": col, "r2": r2, "mae": mae_val})

    return {
        "test_r2": test_r2,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "best_params": grid_search.best_params_,
        "elapsed_seconds": elapsed,
        "per_param": per_param,
        "tuned_prefix": tuned_prefix,
    }


def main() -> None:
    """Run the full tuning pipeline on the default dataset."""
    tune_on_dataset(config.DATASET_PATH, "base")


if __name__ == "__main__":
    config.setup_logging()
    main()
