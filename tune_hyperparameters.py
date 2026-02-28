import math
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

import config


def tune() -> None:
    print("=" * 60)
    print("HYPERPARAMETER TUNING - ANTENNA PARAMETERS")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv(config.DATASET_FILE)
    print(f"\nDataset: {len(df)} samples")
    print(
        f"Frequency range: {df[config.FREQUENCY_COLUMN].min():.2f}"
        f" – {df[config.FREQUENCY_COLUMN].max():.2f} GHz"
    )

    # Prepare data
    X = df[[config.FREQUENCY_COLUMN]].values
    target_columns = [
        col
        for col in df.columns
        if col.lower() != config.FREQUENCY_COLUMN.lower()
    ]
    y = df[target_columns].values

    print(f"\nPredicting {len(target_columns)} antenna parameters")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    # Scale
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    print("\n" + "=" * 60)
    print("PARAMETER GRID")
    print("=" * 60)
    for param, values in config.PARAM_GRID.items():
        print(f"  {param}: {values}")

    total_combinations = int(np.prod([len(v) for v in config.PARAM_GRID.values()]))
    print(f"\nTotal combinations to test: {total_combinations}")
    print("This will use 5-fold cross-validation")
    print("=" * 60)

    # Grid search
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

    print("\nStarting grid search...")
    start_time = time.time()
    grid_search.fit(X_train_scaled, y_train)
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("TUNING COMPLETE!")
    print("=" * 60)
    print(f"Time taken: {elapsed / 60:.1f} minutes ({elapsed:.1f} seconds)")

    # Best parameters
    print("\n" + "=" * 60)
    print("BEST PARAMETERS")
    print("=" * 60)
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nBest cross-validation R² score: {grid_search.best_score_:.4f}")

    # Evaluate on test set
    best_model = grid_search.best_estimator_
    y_pred_test = best_model.predict(X_test_scaled)

    test_r2 = r2_score(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    print(f"\nOverall Performance:")
    print(f"  R²:   {test_r2:.4f}")
    print(f"  MSE:  {test_mse:.4f}")
    print(f"  RMSE: {math.sqrt(test_mse):.4f}")
    print(f"  MAE:  {test_mae:.4f}")

    print("\nPer-Parameter Performance:")
    print("-" * 60)
    for i, col in enumerate(target_columns):
        r2 = r2_score(y_test[:, i], y_pred_test[:, i])
        mae = mean_absolute_error(y_test[:, i], y_pred_test[:, i])
        print(f"{col}:")
        print(f"  R²: {r2:.4f}  |  MAE: {mae:.4f}")

    # Top 10 parameter combinations
    print("\n" + "=" * 60)
    print("TOP 10 PARAMETER COMBINATIONS")
    print("=" * 60)
    results_df = pd.DataFrame(grid_search.cv_results_).sort_values("rank_test_score")
    print(
        "\n{:<6} {:<12} {:<12} {:<12}".format(
            "Rank", "Mean CV R²", "Std CV R²", "Mean Fit Time"
        )
    )
    print("-" * 80)
    for _, row in results_df.head(10).iterrows():
        print(
            "{:<6} {:<12.4f} {:<12.4f} {:<12.2f}s".format(
                int(row["rank_test_score"]),
                row["mean_test_score"],
                row["std_test_score"],
                row["mean_fit_time"],
            )
        )
        print(f"       Params: {row['params']}")
        print()

    # Compare with original model
    print("\n" + "=" * 60)
    print("COMPARISON WITH DEFAULT PARAMETERS")
    print("=" * 60)
    try:
        original_model = joblib.load(config.MODEL_FILE)
        y_pred_original = original_model.predict(X_test_scaled)
        original_r2 = r2_score(y_test, y_pred_original)
        improvement = test_r2 - original_r2
        pct = improvement / original_r2 * 100 if original_r2 != 0 else float("nan")
        print(f"Original model R²: {original_r2:.4f}")
        print(f"Tuned model R²:    {test_r2:.4f}")
        print(f"Improvement:       {improvement:.4f} ({pct:.2f}%)")
    except FileNotFoundError:
        print("Original model not found – skipping comparison")

    # Save tuned model
    print("\n" + "=" * 60)
    print("SAVING TUNED MODEL")
    print("=" * 60)
    joblib.dump(best_model, config.TUNED_MODEL_FILE)
    joblib.dump(scaler_X, config.TUNED_SCALER_FILE)
    joblib.dump(target_columns, config.TUNED_COLUMNS_FILE)
    print("✅ Tuned model saved as:")
    print(f"   - {config.TUNED_MODEL_FILE}")
    print(f"   - {config.TUNED_SCALER_FILE}")
    print(f"   - {config.TUNED_COLUMNS_FILE}")
    print("\n" + "=" * 60)
    print("TUNING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    tune()
