"""Shared pytest fixtures for the antenna-ml test suite."""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Ensure the project root is on the path so imports work when running tests
# from any working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config


@pytest.fixture
def sample_target_columns() -> list[str]:
    """Return the list of target column names (all columns except frequency)."""
    return [c for c in config.EXPECTED_COLUMNS if c != config.FREQUENCY_COL]


@pytest.fixture
def sample_dataset(sample_target_columns) -> pd.DataFrame:
    """Return a small valid DataFrame with the correct schema (30 rows)."""
    rng = np.random.default_rng(0)
    n = 30
    freqs = rng.uniform(config.FREQ_MIN, config.FREQ_MAX, n)
    data = {config.FREQUENCY_COL: freqs}
    for col in sample_target_columns:
        data[col] = rng.uniform(1.0, 100.0, n)
    return pd.DataFrame(data)


@pytest.fixture
def trained_model(sample_dataset, sample_target_columns) -> RandomForestRegressor:
    """Return a RandomForestRegressor fitted on the sample dataset."""
    X = sample_dataset[[config.FREQUENCY_COL]].values
    y = sample_dataset[sample_target_columns].values
    model = RandomForestRegressor(n_estimators=5, random_state=0)
    model.fit(X, y)
    return model


@pytest.fixture
def fitted_scaler(sample_dataset) -> StandardScaler:
    """Return a StandardScaler fitted on the frequency column of the sample dataset."""
    X = sample_dataset[[config.FREQUENCY_COL]].values
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


@pytest.fixture
def tmp_model_dir(tmp_path, trained_model, fitted_scaler, sample_target_columns, monkeypatch):
    """Save model artifacts to a temp directory and patch config to point there.

    Returns the tmp_path directory.
    """
    import joblib

    # Patch MODEL_FILES to use temp paths
    patched = {
        "base": {
            "model": str(tmp_path / "rf_antenna_model.pkl"),
            "scaler": str(tmp_path / "scaler_X.pkl"),
            "target_columns": str(tmp_path / "target_columns.pkl"),
        },
        "tuned": {
            "model": str(tmp_path / "rf_antenna_model_tuned.pkl"),
            "scaler": str(tmp_path / "scaler_X_tuned.pkl"),
            "target_columns": str(tmp_path / "target_columns_tuned.pkl"),
        },
    }
    monkeypatch.setattr(config, "MODEL_FILES", patched)

    # Save base artifacts
    joblib.dump(trained_model, patched["base"]["model"])
    joblib.dump(fitted_scaler, patched["base"]["scaler"])
    joblib.dump(sample_target_columns, patched["base"]["target_columns"])

    return tmp_path
