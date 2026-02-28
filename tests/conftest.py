"""Shared pytest fixtures."""

import sys
import os

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Ensure project root is on the path
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO_ROOT)

import config
from train_model import load_data


@pytest.fixture(scope="session", autouse=True)
def tiny_model_files():
    """Train a tiny model, write pkl files to the repo root, and expose them
    as module-level globals so gradio_app can be imported in tests.

    Returns a dict with the trained objects for use in test assertions.
    """
    X, y, cols = load_data()

    scaler = StandardScaler().fit(X[:300])
    X_scaled = scaler.transform(X[:300])

    m = RandomForestRegressor(n_estimators=5, random_state=0, verbose=0)
    m.fit(X_scaled, y[:300])

    # Write to repo root so gradio_app can find them when imported
    joblib.dump(m, os.path.join(REPO_ROOT, config.MODEL_FILE))
    joblib.dump(scaler, os.path.join(REPO_ROOT, config.SCALER_FILE))
    joblib.dump(cols, os.path.join(REPO_ROOT, config.COLUMNS_FILE))

    yield {
        "model": m,
        "scaler": scaler,
        "target_columns": cols,
    }

    # Clean up pkl files created just for tests
    for f in (config.MODEL_FILE, config.SCALER_FILE, config.COLUMNS_FILE):
        path = os.path.join(REPO_ROOT, f)
        if os.path.exists(path):
            os.remove(path)
