"""Tests for tune_hyperparameters.py."""

import os

import numpy as np
import pytest

import config


def _run_tune(tmp_path, sample_dataset, monkeypatch):
    """Helper: patch config, write sample CSV, run main() with a tiny param grid."""
    import tune_hyperparameters

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
    # Minimal param grid so grid search finishes quickly
    monkeypatch.setattr(
        config,
        "PARAM_GRID",
        {"n_estimators": [5], "max_depth": [3]},
    )

    tune_hyperparameters.main()
    return patched_files


def test_tune_produces_tuned_model_files(tmp_path, sample_dataset, monkeypatch):
    patched_files = _run_tune(tmp_path, sample_dataset, monkeypatch)
    for path in patched_files["tuned"].values():
        assert os.path.exists(path), f"Missing tuned artifact: {path}"


def test_tune_model_can_predict(tmp_path, sample_dataset, monkeypatch):
    import joblib

    patched_files = _run_tune(tmp_path, sample_dataset, monkeypatch)
    model = joblib.load(patched_files["tuned"]["model"])
    scaler = joblib.load(patched_files["tuned"]["scaler"])
    target_columns = joblib.load(patched_files["tuned"]["target_columns"])

    freq_scaled = scaler.transform([[5.0]])
    pred = model.predict(freq_scaled)
    assert pred.shape == (1, len(target_columns))
    assert not np.isnan(pred).any()


def test_tune_skips_comparison_when_no_base_model(tmp_path, sample_dataset, monkeypatch):
    """main() should log a warning and continue when no base model exists."""
    _run_tune(tmp_path, sample_dataset, monkeypatch)
    # If it reached here without raising, the missing-base-model path was handled


def test_tune_comparison_with_base_model(tmp_path, sample_dataset, monkeypatch, trained_model, fitted_scaler, sample_target_columns):
    """main() should log comparison metrics when a base model exists."""
    import joblib

    # Pre-create base model artifacts so comparison branch runs
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
    joblib.dump(trained_model, patched_files["base"]["model"])
    joblib.dump(fitted_scaler, patched_files["base"]["scaler"])
    joblib.dump(sample_target_columns, patched_files["base"]["target_columns"])

    monkeypatch.setattr(config, "DATASET_PATH", csv_path)
    monkeypatch.setattr(config, "MODEL_FILES", patched_files)
    monkeypatch.setattr(config, "VISUALIZATION_PATH", str(tmp_path / "prediction_results.png"))
    monkeypatch.setattr(config, "PARAM_GRID", {"n_estimators": [5], "max_depth": [3]})

    import tune_hyperparameters
    tune_hyperparameters.main()

    assert os.path.exists(patched_files["tuned"]["model"])
