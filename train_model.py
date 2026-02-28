import math

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config

matplotlib.use("Agg")  # Use non-interactive backend for Docker


def load_data(path: str = config.DATASET_FILE):
    """Load dataset and return feature and target arrays."""
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    X = df[[config.FREQUENCY_COLUMN]].values
    target_columns = [
        col for col in df.columns if col.lower() != config.FREQUENCY_COLUMN.lower()
    ]
    y = df[target_columns].values
    return X, y, target_columns


def build_model() -> RandomForestRegressor:
    """Return a configured RandomForestRegressor."""
    return RandomForestRegressor(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )


def print_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_columns: list,
    split_name: str = "Test",
) -> None:
    """Print overall and per-parameter regression metrics."""
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    print(f"\n{'='*60}")
    print(f"OVERALL {split_name.upper()} PERFORMANCE")
    print("=" * 60)
    print(f"R² Score: {r2:.4f}")
    print(f"MSE:      {mse:.4f}")
    print(f"RMSE:     {rmse:.4f}")

    print(f"\n{'='*60}")
    print(f"PER-PARAMETER {split_name.upper()} PERFORMANCE")
    print("=" * 60)
    for i, col in enumerate(target_columns):
        col_r2 = r2_score(y_true[:, i], y_pred[:, i])
        col_mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        col_rmse = math.sqrt(col_mse)
        col_mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        print(f"\n  {col}:")
        print(f"    R²:   {col_r2:.4f}")
        print(f"    RMSE: {col_rmse:.4f}")
        print(f"    MAE:  {col_mae:.4f}")


def save_visualization(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    target_columns: list,
    output_path: str = config.PLOT_FILE,
) -> None:
    """Create and save actual-vs-predicted scatter plots for every target."""
    n = len(target_columns)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(target_columns):
        ax = axes[i]
        ax.scatter(y_test[:, i], y_pred[:, i], alpha=0.5, s=10)
        lo = min(y_test[:, i].min(), y_pred[:, i].min())
        hi = max(y_test[:, i].max(), y_pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="Ideal")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(col, fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplot cells
    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved: {output_path}")


def train() -> None:
    print("=" * 60)
    print("ANTENNA PARAMETER PREDICTION - TRAINING")
    print("=" * 60)

    print("\n[1/7] Loading dataset...")
    X, y, target_columns = load_data()
    print(f"Feature shape: {X.shape}")
    print(f"Target shape:  {y.shape}")
    print(f"Targets ({len(target_columns)}): {target_columns}")

    print("\n[2/7] Splitting data (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}")

    print("\n[3/7] Scaling features...")
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    print("\n[4/7] Training Random Forest model...")
    rf_model = build_model()
    rf_model.fit(X_train_scaled, y_train)

    print("\n[5/7] Evaluating model (train set)...")
    y_pred_train = rf_model.predict(X_train_scaled)
    print_metrics(y_train, y_pred_train, target_columns, split_name="Train")

    print("\n[6/7] Evaluating model (test set)...")
    y_pred_test = rf_model.predict(X_test_scaled)
    print_metrics(y_test, y_pred_test, target_columns, split_name="Test")

    print("\n[7/7] Creating visualization...")
    save_visualization(y_test, y_pred_test, target_columns)

    print("\nSaving model artifacts...")
    joblib.dump(rf_model, config.MODEL_FILE)
    joblib.dump(scaler_X, config.SCALER_FILE)
    joblib.dump(target_columns, config.COLUMNS_FILE)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nSaved files:")
    print(f"  - {config.MODEL_FILE}  (trained model)")
    print(f"  - {config.SCALER_FILE}          (feature scaler)")
    print(f"  - {config.COLUMNS_FILE}    (target names)")
    print(f"  - {config.PLOT_FILE} (visualization)")
    print("\nYou can now run predictions using these files!")
    print("=" * 60)


if __name__ == "__main__":
    train()
