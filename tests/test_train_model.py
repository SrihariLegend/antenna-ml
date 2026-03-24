"""Tests for train_model.py."""

import os

import numpy as np
import pytest

import config


def _run_main(tmp_path, sample_dataset, monkeypatch):
    """Helper: patch config to use tmp_path, write sample CSV, run main()."""
    import train_model

    csv_path = str(tmp_path / "dataset_WIFI7.csv")
    sample_dataset.to_csv(csv_path, index=False)

    patched_files = {
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
    monkeypatch.setattr(config, "DATASET_PATH", csv_path)
    monkeypatch.setattr(config, "MODEL_FILES", patched_files)
    monkeypatch.setattr(config, "VISUALIZATION_PATH", str(tmp_path / "prediction_results.png"))
    # Use fewer trees for speed
    monkeypatch.setattr(
        config,
        "DEFAULT_RF_PARAMS",
        {**config.DEFAULT_RF_PARAMS, "n_estimators": 5, "verbose": 0},
    )

    train_model.main()
    return patched_files


def test_main_produces_model_files(tmp_path, sample_dataset, monkeypatch):
    patched_files = _run_main(tmp_path, sample_dataset, monkeypatch)
    for path in patched_files["base"].values():
        assert os.path.exists(path), f"Missing artifact: {path}"


def test_main_produces_visualization(tmp_path, sample_dataset, monkeypatch):
    _run_main(tmp_path, sample_dataset, monkeypatch)
    viz = tmp_path / "prediction_results.png"
    assert viz.exists()


def test_main_model_can_predict(tmp_path, sample_dataset, monkeypatch):
    import joblib
    patched_files = _run_main(tmp_path, sample_dataset, monkeypatch)
    model = joblib.load(patched_files["base"]["model"])
    scaler = joblib.load(patched_files["base"]["scaler"])
    freq_scaled = scaler.transform([[5.0]])
    pred = model.predict(freq_scaled)
    assert pred.shape[1] == len(config.EXPECTED_COLUMNS) - 1  # all targets


def test_main_predictions_valid(tmp_path, sample_dataset, monkeypatch):
    """Trained model predictions must have correct shape and contain no NaN values."""
    import joblib

    patched_files = _run_main(tmp_path, sample_dataset, monkeypatch)
    model = joblib.load(patched_files["base"]["model"])
    scaler = joblib.load(patched_files["base"]["scaler"])
    target_columns = joblib.load(patched_files["base"]["target_columns"])

    X = sample_dataset[[config.FREQUENCY_COL]].values
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    assert y_pred.shape == (len(sample_dataset), len(target_columns))
    assert not np.isnan(y_pred).any(), "Predictions contain NaN values"
