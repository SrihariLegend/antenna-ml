"""Tests for gradio_app.py."""

import numpy as np
import pytest

import config
import model_io


def _setup_model(trained_model, fitted_scaler, sample_target_columns, monkeypatch):
    """Patch model loading and create the app so predict function is usable."""
    monkeypatch.setattr(model_io, "find_best_model", lambda: "base")
    monkeypatch.setattr(
        model_io,
        "load_model",
        lambda prefix="base": (trained_model, fitted_scaler, sample_target_columns),
    )
    import gradio_app
    import importlib
    importlib.reload(gradio_app)
    gradio_app.create_app()
    return gradio_app.predict_antenna_parameters


# ---------------------------------------------------------------------------
# predict_antenna_parameters behaviour
# ---------------------------------------------------------------------------

def test_predict_valid_frequency(
    trained_model, fitted_scaler, sample_target_columns, monkeypatch
):
    fn = _setup_model(trained_model, fitted_scaler, sample_target_columns, monkeypatch)
    result = fn(5.0)
    assert "5.0 GHz" in result
    assert "| Parameter | Value |" in result


def test_predict_returns_all_parameters(
    trained_model, fitted_scaler, sample_target_columns, monkeypatch
):
    fn = _setup_model(trained_model, fitted_scaler, sample_target_columns, monkeypatch)
    result = fn(2.4)
    for col in sample_target_columns:
        assert col in result, f"Column '{col}' missing from output"


def test_predict_out_of_range_low(
    trained_model, fitted_scaler, sample_target_columns, monkeypatch
):
    fn = _setup_model(trained_model, fitted_scaler, sample_target_columns, monkeypatch)
    result = fn(0.5)
    assert "out of training range" in result.lower() or "⚠️" in result


def test_predict_out_of_range_high(
    trained_model, fitted_scaler, sample_target_columns, monkeypatch
):
    fn = _setup_model(trained_model, fitted_scaler, sample_target_columns, monkeypatch)
    result = fn(10.5)
    assert "out of training range" in result.lower() or "⚠️" in result


def test_predict_boundary_freq_min(
    trained_model, fitted_scaler, sample_target_columns, monkeypatch
):
    fn = _setup_model(trained_model, fitted_scaler, sample_target_columns, monkeypatch)
    result = fn(config.FREQ_MIN)
    assert "| Parameter | Value |" in result


def test_predict_boundary_freq_max(
    trained_model, fitted_scaler, sample_target_columns, monkeypatch
):
    fn = _setup_model(trained_model, fitted_scaler, sample_target_columns, monkeypatch)
    result = fn(config.FREQ_MAX)
    assert "| Parameter | Value |" in result


def test_predict_result_is_string(
    trained_model, fitted_scaler, sample_target_columns, monkeypatch
):
    fn = _setup_model(trained_model, fitted_scaler, sample_target_columns, monkeypatch)
    assert isinstance(fn(5.0), str)


# ---------------------------------------------------------------------------
# Graceful startup with no model
# ---------------------------------------------------------------------------

def test_predict_no_model_available(monkeypatch):
    """When no model is loaded, predict should prompt user to train."""
    import gradio_app
    import importlib

    monkeypatch.setattr(
        model_io, "find_best_model",
        lambda: (_ for _ in ()).throw(FileNotFoundError("No model")),
    )
    importlib.reload(gradio_app)
    gradio_app.create_app()
    result = gradio_app.predict_antenna_parameters(5.0)
    assert "no trained model" in result.lower() or "train" in result.lower()


# ---------------------------------------------------------------------------
# create_app
# ---------------------------------------------------------------------------

def test_create_app_returns_blocks(
    trained_model, fitted_scaler, sample_target_columns, monkeypatch
):
    import gradio as gr
    monkeypatch.setattr(model_io, "find_best_model", lambda: "base")
    monkeypatch.setattr(
        model_io,
        "load_model",
        lambda prefix="base": (trained_model, fitted_scaler, sample_target_columns),
    )
    import gradio_app
    import importlib
    importlib.reload(gradio_app)
    app = gradio_app.create_app()
    assert isinstance(app, gr.Blocks)


def test_import_gradio_app_has_no_side_effects(monkeypatch):
    """Importing gradio_app must NOT load models or call sys.exit."""
    loaded = []
    monkeypatch.setattr(model_io, "find_best_model", lambda: loaded.append(1) or "base")
    import importlib
    import gradio_app
    importlib.reload(gradio_app)
    assert loaded == [], "find_best_model was called at import time — global state detected"
