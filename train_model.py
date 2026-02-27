import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Docker
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

print("="*60)
print("ANTENNA PARAMETER PREDICTION - TRAINING")
print("="*60)

# Load the dataset
print("\n[1/7] Loading dataset...")
df = pd.read_csv('dataset_WIFI7.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Define features and targets
print("\n[2/7] Preparing features and targets...")
X = df[['Frequency(GHz)']].values
target_columns = [col for col in df.columns if col.lower() != 'frequency(ghz)']
y = df[target_columns].values

print(f"Feature shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target columns ({len(target_columns)}): {target_columns}")

# Split the data
print("\n[3/7] Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Scale features
print("\n[4/7] Scaling features...")
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Create and train Random Forest
print("\n[5/7] Training Random Forest model...")
print("Parameters: n_estimators=100, n_jobs=-1 (using all CPU cores)")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_model.fit(X_train_scaled, y_train)

# Make predictions
print("\n[6/7] Evaluating model...")
y_pred_train = rf_model.predict(X_train_scaled)
y_pred_test = rf_model.predict(X_test_scaled)

# Overall metrics
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

print("\n" + "="*60)
print("OVERALL PERFORMANCE")
print("="*60)
print(f"Train R² Score: {train_r2:.4f}")
print(f"Test R² Score:  {test_r2:.4f}")
print(f"Train MSE:      {train_mse:.4f}")
print(f"Test MSE:       {test_mse:.4f}")

# Per-parameter metrics
print("\n" + "="*60)
print("PER-PARAMETER PERFORMANCE")
print("="*60)
for i, col in enumerate(target_columns):
    r2 = r2_score(y_test[:, i], y_pred_test[:, i])
    mse = mean_squared_error(y_test[:, i], y_pred_test[:, i])
    mae = mean_absolute_error(y_test[:, i], y_pred_test[:, i])
    print(f"\n{col}:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MSE:      {mse:.4f}")
    print(f"  MAE:      {mae:.4f}")

# Create visualization
print("\n[7/7] Creating visualization...")
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(target_columns[:9]):
    if i < len(target_columns):
        axes[i].scatter(y_test[:, i], y_pred_test[:, i], alpha=0.5)
        axes[i].plot([y_test[:, i].min(), y_test[:, i].max()], 
                     [y_test[:, i].min(), y_test[:, i].max()], 
                     'r--', lw=2)
        axes[i].set_xlabel('Actual')
        axes[i].set_ylabel('Predicted')
        axes[i].set_title(f'{col}')
        axes[i].grid(True, alpha=0.3)

# Hide empty subplots
for i in range(len(target_columns), 9):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
print("Visualization saved: prediction_results.png")

# Save model and metadata
print("\nSaving model artifacts...")
joblib.dump(rf_model, 'rf_antenna_model.pkl')
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(target_columns, 'target_columns.pkl')

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print("\nSaved files:")
print("  - rf_antenna_model.pkl  (trained model)")
print("  - scaler_X.pkl          (feature scaler)")
print("  - target_columns.pkl    (target names)")
print("  - prediction_results.png (visualization)")
print("\nYou can now run predictions using these files!")
print("="*60)
