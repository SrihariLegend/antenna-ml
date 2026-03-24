"""Tests for model_io.py."""

import os

import joblib
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import config
import model_io


# ---------------------------------------------------------------------------
# save_model / load_model roundtrip
# ---------------------------------------------------------------------------

def test_save_and_load_roundtrip(tmp_model_dir, trained_model, fitted_scaler, sample_target_columns):
    model, scaler, cols = model_io.load_model(prefix="base")
    assert hasattr(model, "predict")
    assert hasattr(scaler, "transform")
    assert cols == sample_target_columns


def test_load_model_returns_correct_types(tmp_model_dir):
    model, scaler, cols = model_io.load_model(prefix="base")
    assert isinstance(model, RandomForestRegressor)
    assert isinstance(scaler, StandardScaler)
    assert isinstance(cols, list)
    assert all(isinstance(c, str) for c in cols)


# ---------------------------------------------------------------------------
# save_model errors
# ---------------------------------------------------------------------------

def test_save_model_custom_prefix(tmp_path, trained_model, fitted_scaler, sample_target_columns, monkeypatch):
    """Arbitrary prefixes now generate convention-based paths instead of raising KeyError."""
    monkeypatch.chdir(tmp_path)
    model_io.save_model(trained_model, fitted_scaler, sample_target_columns, prefix="custom")
    assert os.path.exists(tmp_path / "rf_antenna_model_custom.pkl")
    assert os.path.exists(tmp_path / "scaler_X_custom.pkl")
    assert os.path.exists(tmp_path / "target_columns_custom.pkl")


def test_save_model_creates_all_files(tmp_model_dir, trained_model, fitted_scaler, sample_target_columns):
    files = config.MODEL_FILES["base"]
    for path in files.values():
        assert os.path.exists(path), f"Missing artifact: {path}"


# ---------------------------------------------------------------------------
# load_model errors
# ---------------------------------------------------------------------------

def test_load_model_file_not_found(tmp_path, monkeypatch):
    patched = {
        "base": {
            "model": str(tmp_path / "missing_model.pkl"),
            "scaler": str(tmp_path / "missing_scaler.pkl"),
            "target_columns": str(tmp_path / "missing_cols.pkl"),
        },
        "tuned": {
            "model": str(tmp_path / "missing_model_t.pkl"),
            "scaler": str(tmp_path / "missing_scaler_t.pkl"),
            "target_columns": str(tmp_path / "missing_cols_t.pkl"),
        },
    }
    monkeypatch.setattr(config, "MODEL_FILES", patched)
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        model_io.load_model(prefix="base")


def test_load_model_corrupt_pickle(tmp_path, monkeypatch):
    corrupt = tmp_path / "corrupt.pkl"
    corrupt.write_bytes(b"this is not a valid pickle")
    patched = {
        "base": {
            "model": str(corrupt),
            "scaler": str(corrupt),
            "target_columns": str(corrupt),
        },
        "tuned": config.MODEL_FILES.get("tuned", {}),
    }
    monkeypatch.setattr(config, "MODEL_FILES", patched)
    with pytest.raises(ValueError, match="Failed to load"):
        model_io.load_model(prefix="base")


def test_load_model_unknown_prefix_file_not_found():
    """Unknown prefixes generate convention-based paths; raises FileNotFoundError if files missing."""
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        model_io.load_model(prefix="nonexistent")


# ---------------------------------------------------------------------------
# find_best_model
# ---------------------------------------------------------------------------

def test_find_best_model_returns_base_when_only_base_exists(tmp_model_dir):
    result = model_io.find_best_model()
    assert result == "base"


def test_find_best_model_prefers_tuned(
    tmp_model_dir, trained_model, fitted_scaler, sample_target_columns
):
    # Save tuned artifacts too
    model_io.save_model(trained_model, fitted_scaler, sample_target_columns, prefix="tuned")
    result = model_io.find_best_model()
    assert result == "tuned"


def test_find_best_model_raises_when_none(tmp_path, monkeypatch):
    patched = {
        "base": {
            "model": str(tmp_path / "no_model.pkl"),
            "scaler": str(tmp_path / "no_scaler.pkl"),
            "target_columns": str(tmp_path / "no_cols.pkl"),
        },
        "tuned": {
            "model": str(tmp_path / "no_model_t.pkl"),
            "scaler": str(tmp_path / "no_scaler_t.pkl"),
            "target_columns": str(tmp_path / "no_cols_t.pkl"),
        },
    }
    monkeypatch.setattr(config, "MODEL_FILES", patched)
    monkeypatch.setattr(config, "DATASET_REGISTRY_PATH", str(tmp_path / "empty_registry.json"))
    with pytest.raises(FileNotFoundError, match="No trained model found"):
        model_io.find_best_model()
