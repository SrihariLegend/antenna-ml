import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import time

print("="*60)
print("HYPERPARAMETER TUNING - ANTENNA PARAMETERS")
print("="*60)

# Load dataset
df = pd.read_csv('dataset_WIFI7.csv')
print(f"\nDataset: {len(df)} samples")
print(f"Frequency range: {df['Frequency(GHz)'].min():.2f} - {df['Frequency(GHz)'].max():.2f} GHz")

# Prepare data
X = df[['Frequency(GHz)']].values
target_columns = [col for col in df.columns if col.lower() != 'frequency(ghz)']
y = df[target_columns].values

print(f"\nPredicting {len(target_columns)} antenna parameters")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

print("\n" + "="*60)
print("PARAMETER GRID")
print("="*60)
for param, values in param_grid.items():
    print(f"  {param}: {values}")

total_combinations = np.prod([len(v) for v in param_grid.values()])
print(f"\nTotal combinations to test: {total_combinations}")
print("Estimated time: ~5-10 minutes")
print("\nThis will use 5-fold cross-validation")
print("="*60)

# Create base model
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

# Grid search
print("\nStarting grid search...")
start_time = time.time()

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    verbose=2,
    n_jobs=-1,
    return_train_score=True
)

grid_search.fit(X_train_scaled, y_train)

elapsed_time = time.time() - start_time
elapsed_min = elapsed_time / 60

print("\n" + "="*60)
print("TUNING COMPLETE!")
print("="*60)
print(f"Time taken: {elapsed_min:.1f} minutes ({elapsed_time:.1f} seconds)")

# Best parameters
print("\n" + "="*60)
print("BEST PARAMETERS")
print("="*60)
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest cross-validation R² score: {grid_search.best_score_:.4f}")

# Evaluate on test set
print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

best_model = grid_search.best_estimator_
y_pred_test = best_model.predict(X_test_scaled)

# Overall metrics
test_r2 = r2_score(y_test, y_pred_test)
test_mse = mean_squared_error(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"\nOverall Performance:")
print(f"  R² Score: {test_r2:.4f}")
print(f"  MSE:      {test_mse:.4f}")
print(f"  MAE:      {test_mae:.4f}")

# Per-parameter performance
print("\nPer-Parameter Performance:")
print("-" * 60)
for i, col in enumerate(target_columns):
    r2 = r2_score(y_test[:, i], y_pred_test[:, i])
    mae = mean_absolute_error(y_test[:, i], y_pred_test[:, i])
    print(f"{col}:")
    print(f"  R² Score: {r2:.4f}  |  MAE: {mae:.4f}")

# Show top 10 parameter combinations
print("\n" + "="*60)
print("TOP 10 PARAMETER COMBINATIONS")
print("="*60)

results_df = pd.DataFrame(grid_search.cv_results_)
results_df = results_df.sort_values('rank_test_score')

print("\n{:<6} {:<12} {:<12} {:<12}".format("Rank", "Mean CV R²", "Std CV R²", "Mean Fit Time"))
print("-" * 80)

for idx, row in results_df.head(10).iterrows():
    print("{:<6} {:<12.4f} {:<12.4f} {:<12.2f}s".format(
        int(row['rank_test_score']),
        row['mean_test_score'],
        row['std_test_score'],
        row['mean_fit_time']
    ))
    print(f"       Params: {row['params']}")
    print()

# Compare with original model
print("\n" + "="*60)
print("COMPARISON WITH DEFAULT PARAMETERS")
print("="*60)

# Load original model if it exists
try:
    original_model = joblib.load('rf_antenna_model.pkl')
    y_pred_original = original_model.predict(X_test_scaled)
    original_r2 = r2_score(y_test, y_pred_original)
    
    print(f"Original model R²:     {original_r2:.4f}")
    print(f"Tuned model R²:        {test_r2:.4f}")
    print(f"Improvement:           {(test_r2 - original_r2):.4f} ({((test_r2 - original_r2) / original_r2 * 100):.2f}%)")
except:
    print("Original model not found - skipping comparison")

# Save tuned model
print("\n" + "="*60)
print("SAVING TUNED MODEL")
print("="*60)

joblib.dump(best_model, 'rf_antenna_model_tuned.pkl')
joblib.dump(scaler_X, 'scaler_X_tuned.pkl')
joblib.dump(target_columns, 'target_columns_tuned.pkl')

print("✅ Tuned model saved as:")
print("   - rf_antenna_model_tuned.pkl")
print("   - scaler_X_tuned.pkl")
print("   - target_columns_tuned.pkl")

print("\n" + "="*60)
print("TUNING COMPLETE!")
print("="*60)
print("\nTo use the tuned model in gradio_app.py, replace:")
print("  'rf_antenna_model.pkl' → 'rf_antenna_model_tuned.pkl'")
print("  'scaler_X.pkl' → 'scaler_X_tuned.pkl'")
print("  'target_columns.pkl' → 'target_columns_tuned.pkl'")
print("="*60)
