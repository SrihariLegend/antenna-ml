"""Unit tests for the antenna-ml project."""

import math
import sys
import os

import numpy as np
import pandas as pd
import pytest

# Allow importing project modules from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from train_model import build_model, load_data, print_metrics, save_visualization


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_freq_range(self):
        assert config.FREQ_MIN < config.FREQ_MAX

    def test_test_size_valid(self):
        assert 0 < config.TEST_SIZE < 1

    def test_random_state_is_int(self):
        assert isinstance(config.RANDOM_STATE, int)

    def test_param_grid_non_empty(self):
        assert all(len(v) > 0 for v in config.PARAM_GRID.values())


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TestDataset:
    def test_dataset_loads(self):
        df = pd.read_csv(config.DATASET_FILE)
        assert len(df) > 0

    def test_no_duplicate_columns(self):
        df = pd.read_csv(config.DATASET_FILE)
        assert len(df.columns) == len(set(df.columns)), "Duplicate column names found"

    def test_frequency_column_present(self):
        df = pd.read_csv(config.DATASET_FILE)
        assert config.FREQUENCY_COLUMN in df.columns

    def test_no_missing_values(self):
        df = pd.read_csv(config.DATASET_FILE)
        assert df.isnull().sum().sum() == 0, "Dataset contains NaN values"

    def test_frequency_range(self):
        df = pd.read_csv(config.DATASET_FILE)
        assert df[config.FREQUENCY_COLUMN].min() >= config.FREQ_MIN
        assert df[config.FREQUENCY_COLUMN].max() <= config.FREQ_MAX

    def test_radius_column_spelling(self):
        """Column should say 'Radius', not 'Radiaus'."""
        df = pd.read_csv(config.DATASET_FILE)
        for col in df.columns:
            assert "radiaus" not in col.lower(), f"Typo 'Radiaus' found in column: {col}"


# ---------------------------------------------------------------------------
# train_model helpers
# ---------------------------------------------------------------------------

class TestTrainModelHelpers:
    def test_load_data_shapes(self):
        X, y, cols = load_data()
        assert X.shape[1] == 1  # single frequency feature
        assert y.shape[1] == len(cols)
        assert X.shape[0] == y.shape[0]

    def test_load_data_no_frequency_in_targets(self):
        _, _, cols = load_data()
        for col in cols:
            assert col.lower() != config.FREQUENCY_COLUMN.lower()

    def test_build_model_returns_rf(self):
        from sklearn.ensemble import RandomForestRegressor
        m = build_model()
        assert isinstance(m, RandomForestRegressor)

    def test_model_fit_predict(self):
        X, y, _ = load_data()
        m = build_model()
        # Use a small subset so the test runs quickly
        m.set_params(n_estimators=5, verbose=0)
        m.fit(X[:200], y[:200])
        preds = m.predict(X[200:210])
        assert preds.shape == (10, y.shape[1])

    def test_save_visualization(self, tmp_path):
        X, y, cols = load_data()
        m = build_model()
        m.set_params(n_estimators=5, verbose=0)
        m.fit(X[:200], y[:200])
        preds = m.predict(X[200:250])
        out = str(tmp_path / "test_plot.png")
        save_visualization(y[200:250], preds, cols, output_path=out)
        assert os.path.exists(out)


# ---------------------------------------------------------------------------
# Gradio prediction function
# ---------------------------------------------------------------------------

class TestPredictFunction:
    """Smoke-test predict_at_frequency using a tiny in-session trained model."""

    def _get_app(self, tiny_model_files, monkeypatch):
        """Import (or reload) gradio_app with model globals patched."""
        info = tiny_model_files
        # gradio_app may already be imported; patch its globals directly
        import importlib
        import gradio_app as app_module

        monkeypatch.setattr(app_module, "model", info["model"])
        monkeypatch.setattr(app_module, "scaler", info["scaler"])
        monkeypatch.setattr(app_module, "target_columns", info["target_columns"])
        monkeypatch.setattr(app_module, "model_label", "test")
        return app_module

    def test_valid_frequency(self, tiny_model_files, monkeypatch):
        app = self._get_app(tiny_model_files, monkeypatch)
        table, fig = app.predict_at_frequency(5.0)
        assert "5.000 GHz" in table
        assert fig is not None

    def test_out_of_range_frequency_warns(self, tiny_model_files, monkeypatch):
        app = self._get_app(tiny_model_files, monkeypatch)
        table, _ = app.predict_at_frequency(1.0)  # below FREQ_MIN
        assert "outside the training range" in table or "⚠️" in table
