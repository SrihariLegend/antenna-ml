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
from generate_dataset import patch_dimensions, generate


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


# ---------------------------------------------------------------------------
# patch_dimensions (physics equations)
# ---------------------------------------------------------------------------

class TestPatchDimensions:
    def test_returns_positive_dimensions(self):
        result = patch_dimensions(5e9, 4.4, 1.6e-3)
        assert result is not None
        W, L = result
        assert W > 0
        assert L > 0

    def test_higher_eps_gives_smaller_patch(self):
        """Higher dielectric constant → shorter electrical wavelength → smaller patch."""
        _, L_low = patch_dimensions(5e9, 2.0, 1.6e-3)
        _, L_high = patch_dimensions(5e9, 10.0, 1.6e-3)
        assert L_high < L_low

    def test_lower_frequency_gives_larger_patch(self):
        """Lower frequency → longer wavelength → larger patch."""
        _, L_high_freq = patch_dimensions(10e9, 4.4, 1.6e-3)
        _, L_low_freq = patch_dimensions(2e9, 4.4, 1.6e-3)
        assert L_low_freq > L_high_freq

    def test_known_reference_design(self):
        """Cross-check against a textbook 2.45 GHz patch on FR4 (ε_r=4.4, h=1.6mm).

        Expected patch length ≈ 29 mm, width ≈ 37 mm (±15% tolerance for the
        Hammerstad approximation).
        """
        W, L = patch_dimensions(2.45e9, 4.4, 1.6e-3)
        assert 25 <= L * 1e3 <= 35, f"Patch length out of expected range: {L*1e3:.1f} mm"
        assert 30 <= W * 1e3 <= 45, f"Patch width  out of expected range: {W*1e3:.1f} mm"


# ---------------------------------------------------------------------------
# generate_dataset
# ---------------------------------------------------------------------------

class TestGenerateDataset:
    def test_returns_dataframe(self):
        df = generate(n_freqs=3, n_heights=3, n_eps=3)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_required_columns_present(self):
        df = generate(n_freqs=2, n_heights=2, n_eps=2)
        expected = {
            "Frequency(GHz)", "Substrate_Height(mm)", "Dielectric_Constant",
            "Patch_Length(mm)", "Patch_Width(mm)",
            "Substrate_Length(mm)", "Substrate_Width(mm)",
        }
        assert expected.issubset(set(df.columns))

    def test_all_positive_dimensions(self):
        df = generate(n_freqs=5, n_heights=5, n_eps=5)
        for col in ("Patch_Length(mm)", "Patch_Width(mm)",
                    "Substrate_Length(mm)", "Substrate_Width(mm)"):
            assert (df[col] > 0).all(), f"Non-positive value found in {col}"

    def test_substrate_larger_than_patch(self):
        df = generate(n_freqs=5, n_heights=5, n_eps=5)
        assert (df["Substrate_Length(mm)"] > df["Patch_Length(mm)"]).all()
        assert (df["Substrate_Width(mm)"] > df["Patch_Width(mm)"]).all()

    def test_no_missing_values(self):
        df = generate(n_freqs=5, n_heights=5, n_eps=5)
        assert df.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# Constraint-based design function
# ---------------------------------------------------------------------------

class TestConstraintBasedDesign:
    def _get_app_with_reverse(self, tiny_model_files, monkeypatch):
        import gradio_app as app_module
        monkeypatch.setattr(app_module, "model", tiny_model_files["model"])
        monkeypatch.setattr(app_module, "scaler", tiny_model_files["scaler"])
        monkeypatch.setattr(app_module, "target_columns", tiny_model_files["target_columns"])
        monkeypatch.setattr(app_module, "model_label", "test")
        # Disable reverse model so tests run without pre-trained artefacts
        monkeypatch.setattr(app_module, "reverse_model_available", False)
        return app_module

    def test_valid_constraints_return_designs(self, tiny_model_files, monkeypatch):
        app = self._get_app_with_reverse(tiny_model_files, monkeypatch)
        table, fig = app.find_designs_for_constraints(5.0, 50.0, 50.0)
        assert "5.000 GHz" in table
        assert "valid designs" in table.lower() or "Constraint" in table
        assert fig is not None

    def test_impossible_constraints_return_warning(self, tiny_model_files, monkeypatch):
        app = self._get_app_with_reverse(tiny_model_files, monkeypatch)
        # Extremely tight constraint – no patch can be ≤ 1 mm at 1 GHz
        table, _ = app.find_designs_for_constraints(1.0, 1.0, 1.0)
        assert "⚠️" in table or "No valid designs" in table

    def test_designs_respect_constraints(self, tiny_model_files, monkeypatch):
        app = self._get_app_with_reverse(tiny_model_files, monkeypatch)
        max_L, max_W = 60.0, 70.0
        table, _ = app.find_designs_for_constraints(5.0, max_L, max_W)
        # All reported patch lengths and widths must be ≤ the constraints
        for patch_l, patch_w in _extract_patch_dims_from_table(table):
            assert patch_l <= max_L + 0.01, f"Patch length {patch_l} exceeds max {max_L}"
            assert patch_w <= max_W + 0.01, f"Patch width {patch_w} exceeds max {max_W}"


def _extract_patch_dims_from_table(table: str):
    """Yield (patch_length, patch_width) tuples from a constraint design table.

    The markdown table produced by ``find_designs_for_constraints`` has
    columns: Substrate Height | Dielectric Constant | Patch Length | Patch Width | …
    This helper skips header and separator rows and parses numeric data rows.
    """
    for line in table.split("\n"):
        if not line.startswith("|"):
            continue
        if "---" in line or "Substrate" in line or "Parameter" in line:
            continue
        cols_vals = [c.strip() for c in line.split("|") if c.strip()]
        if len(cols_vals) >= 4:
            try:
                yield float(cols_vals[2]), float(cols_vals[3])
            except ValueError:
                pass
