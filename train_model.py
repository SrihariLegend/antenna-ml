"""Train a Random Forest model to predict antenna parameters from frequency."""

import logging
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Docker
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config
import data_loader
import model_io

config.setup_logging()
logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics produced by a training run."""

    train_r2: float
    test_r2: float
    train_mse: float
    test_mse: float
    per_param: list[dict]  # each: {"name": str, "r2": float, "mse": float, "mae": float}


def train_on_dataset(dataset_path: str, model_prefix: str) -> dict:
    """Run the full training pipeline on a given dataset.

    Args:
        dataset_path: Path to the CSV file.
        model_prefix: Prefix for saving model artifacts.

    Returns:
        Dict with keys: train_r2, test_r2, train_mse, test_mse,
        per_param (list of dicts with r2, mse, mae per column).

    Raises:
        SystemExit on validation failure, OSError on I/O errors.
    """
    logger.info("=" * 60)
    logger.info("ANTENNA PARAMETER PREDICTION - TRAINING")
    logger.info("=" * 60)

    # [1/7] Load dataset
    logger.info("[1/7] Loading dataset...")
    df = data_loader.load_dataset(dataset_path)
    result = data_loader.validate_dataset(df)
    if not result.valid:
        raise SystemExit(f"Dataset validation failed: {result.errors}")

    # [2/7] Prepare features and targets
    logger.info("[2/7] Preparing features and targets...")
    X, y, target_columns = data_loader.prepare_features_targets(df)

    # [3/7] Split and scale
    logger.info("[3/7] Splitting data (80%% train, 20%% test)...")
    splits = data_loader.split_and_scale(X, y)
    X_train_scaled = splits["X_train_scaled"]
    X_test_scaled = splits["X_test_scaled"]
    y_train = splits["y_train"]
    y_test = splits["y_test"]
    scaler = splits["scaler"]

    # [4/7] Train model
    logger.info("[4/7] Training Random Forest model (n_estimators=100, n_jobs=-1)...")
    rf_model = RandomForestRegressor(**config.DEFAULT_RF_PARAMS)
    rf_model.fit(X_train_scaled, y_train)

    # [5/7] Evaluate
    logger.info("[5/7] Evaluating model...")
    y_pred_train = rf_model.predict(X_train_scaled)
    y_pred_test = rf_model.predict(X_test_scaled)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)

    logger.info("=" * 60)
    logger.info("OVERALL PERFORMANCE")
    logger.info("=" * 60)
    logger.info("Train R²: %.4f  |  Test R²: %.4f", train_r2, test_r2)
    logger.info("Train MSE: %.4f  |  Test MSE: %.4f", train_mse, test_mse)

    logger.info("PER-PARAMETER PERFORMANCE")
    per_param = []
    for i, col in enumerate(target_columns):
        r2 = r2_score(y_test[:, i], y_pred_test[:, i])
        mse = mean_squared_error(y_test[:, i], y_pred_test[:, i])
        mae = mean_absolute_error(y_test[:, i], y_pred_test[:, i])
        logger.info("%s — R²: %.4f  MSE: %.4f  MAE: %.4f", col, r2, mse, mae)
        per_param.append({"name": col, "r2": r2, "mse": mse, "mae": mae})

    # [6/7] Visualize
    logger.info("[6/7] Creating visualization...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, col in enumerate(target_columns[:9]):
        axes[i].scatter(y_test[:, i], y_pred_test[:, i], alpha=0.5)
        axes[i].plot(
            [y_test[:, i].min(), y_test[:, i].max()],
            [y_test[:, i].min(), y_test[:, i].max()],
            "r--",
            lw=2,
        )
        axes[i].set_xlabel("Actual")
        axes[i].set_ylabel("Predicted")
        axes[i].set_title(col)
        axes[i].grid(True, alpha=0.3)

    for i in range(len(target_columns), 9):
        axes[i].axis("off")

    plt.tight_layout()
    try:
        plt.savefig(config.VISUALIZATION_PATH, dpi=300, bbox_inches="tight")
        logger.info("Visualization saved: %s", config.VISUALIZATION_PATH)
    except OSError as exc:
        logger.error("Failed to save visualization: %s", exc)
    finally:
        plt.close(fig)

    # [7/7] Save model
    logger.info("[7/7] Saving model artifacts...")
    model_io.save_model(rf_model, scaler, target_columns, prefix=model_prefix)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)

    return {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "per_param": per_param,
    }


def main() -> None:
    """Run the full training pipeline."""
    train_on_dataset(config.DATASET_PATH, "base")


if __name__ == "__main__":
    main()
