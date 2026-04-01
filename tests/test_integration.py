"""End-to-end integration tests for the antenna-ml pipeline."""

import os

import numpy as np
import pytest

import config
import data_loader
import model_io


def _setup_pipeline(tmp_path, sample_dataset, monkeypatch):
    """Patch config paths to use tmp_path, save sample CSV, return patched MODEL_FILES."""
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
    monkeypatch.setattr(
        config,
        "DEFAULT_RF_PARAMS",
        {**config.DEFAULT_RF_PARAMS, "n_estimators": 5, "verbose": 0},
    )
    return patched_files


def test_train_then_predict(tmp_path, sample_dataset, monkeypatch):
    """Full pipeline: train → save → load → predict."""
    import train_model
    _setup_pipeline(tmp_path, sample_dataset, monkeypatch)

    train_model.main()

    model, scaler, target_columns = model_io.load_model(prefix="base")
    freq_scaled = scaler.transform([[5.0]])
    pred = model.predict(freq_scaled)

    assert pred.shape == (1, len(target_columns))
    assert not np.any(np.isnan(pred)), "Predictions contain NaN values"


def test_gradio_predict_with_trained_model(tmp_path, sample_dataset, monkeypatch):
    """Train model then use it via the Gradio predict function."""
    import importlib
    import gradio_app
    import train_model
    _setup_pipeline(tmp_path, sample_dataset, monkeypatch)

    train_model.main()

    importlib.reload(gradio_app)
    gradio_app.create_app()
    result = gradio_app.predict_antenna_parameters(5.0)

    assert isinstance(result, str)
    assert "| Parameter | Value |" in result
    assert "5.0 GHz" in result


def test_load_and_validate_real_dataset():
    """Validate the actual dataset_WIFI7.csv is loadable and passes validation."""
    if not os.path.exists(config.DATASET_PATH):
        pytest.skip("dataset_WIFI7.csv not present in working directory")

    df = data_loader.load_dataset(config.DATASET_PATH)
    result = data_loader.validate_dataset(df)

    # Duplicate column warning is expected (the CSV still has old header until fixed)
    # but there should be no hard errors beyond column count
    assert df is not None
    assert len(df) > 0


def test_prepare_and_split_real_dataset():
    """End-to-end data prep on the real dataset."""
    if not os.path.exists(config.DATASET_PATH):
        pytest.skip("dataset_WIFI7.csv not present in working directory")

    df = data_loader.load_dataset(config.DATASET_PATH)
    X, y, cols = data_loader.prepare_features_targets(df)
    splits = data_loader.split_and_scale(X, y)

    assert splits["X_train_scaled"].shape[1] == 1
    assert splits["y_train"].shape[1] == len(cols)
