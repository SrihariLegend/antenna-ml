"""Shared configuration and constants for the antenna-ml project."""

# Dataset
DATASET_FILE = "dataset_WIFI7.csv"
FREQUENCY_COLUMN = "Frequency(GHz)"

# Frequency range used during training (WiFi 7 bands)
FREQ_MIN = 2.0
FREQ_MAX = 7.0

# Train / test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Base model hyperparameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_SPLIT = 2
RF_MIN_SAMPLES_LEAF = 1

# Hyperparameter search grid (used by tune_hyperparameters.py)
PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}

# Saved artifact file names
MODEL_FILE = "rf_antenna_model.pkl"
SCALER_FILE = "scaler_X.pkl"
COLUMNS_FILE = "target_columns.pkl"
PLOT_FILE = "prediction_results.png"

TUNED_MODEL_FILE = "rf_antenna_model_tuned.pkl"
TUNED_SCALER_FILE = "scaler_X_tuned.pkl"
TUNED_COLUMNS_FILE = "target_columns_tuned.pkl"
