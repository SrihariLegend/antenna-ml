"""Tests for data_loader.py."""

import re

import numpy as np
import pandas as pd
import pytest

import config
import data_loader


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------

def test_load_dataset_success(sample_dataset, tmp_path, monkeypatch):
    path = str(tmp_path / "test.csv")
    sample_dataset.to_csv(path, index=False)
    df = data_loader.load_dataset(path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(sample_dataset)


def test_load_dataset_file_not_found():
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        data_loader.load_dataset("nonexistent_file.csv")


def test_load_dataset_empty_file(tmp_path):
    path = str(tmp_path / "empty.csv")
    path_obj = tmp_path / "empty.csv"
    path_obj.write_text("")
    with pytest.raises(ValueError, match="empty"):
        data_loader.load_dataset(path)


def test_load_dataset_malformed_csv(tmp_path):
    path = tmp_path / "bad.csv"
    # Write a file that pandas cannot parse as CSV (binary noise)
    path.write_bytes(b"\xff\xfe" * 1000)
    # May raise ValueError or succeed depending on pandas version; just ensure no crash
    try:
        data_loader.load_dataset(str(path))
    except ValueError:
        pass  # Expected


# ---------------------------------------------------------------------------
# validate_dataset
# ---------------------------------------------------------------------------

def test_validate_dataset_valid(sample_dataset):
    result = data_loader.validate_dataset(sample_dataset)
    assert result.valid is True
    assert result.errors == []


def test_validate_dataset_missing_frequency_column(sample_dataset):
    df = sample_dataset.drop(columns=[config.FREQUENCY_COL])
    result = data_loader.validate_dataset(df)
    assert result.valid is False
    assert any("Frequency" in e or "column" in e.lower() for e in result.errors)


def test_validate_dataset_wrong_column_count():
    df = pd.DataFrame({"Frequency(GHz)": [2.0, 3.0], "x": [1, 2]})
    result = data_loader.validate_dataset(df)
    assert result.valid is False
    assert any("missing" in e.lower() for e in result.errors)


def test_validate_dataset_duplicate_columns():
    # Pandas handles duplicates by suffixing; we pass a df with manually set dupe cols
    df = pd.DataFrame([[2.0, 1.0, 1.0]], columns=[config.FREQUENCY_COL, "a", "a"])
    result = data_loader.validate_dataset(df)
    # Should warn about duplicates
    assert any("duplicate" in w.lower() for w in result.warnings)


def test_validate_dataset_nan_values(sample_dataset):
    df = sample_dataset.copy()
    df.iloc[0, 1] = float("nan")
    result = data_loader.validate_dataset(df)
    assert result.valid is False
    assert any("nan" in e.lower() or "NaN" in e for e in result.errors)


def test_validate_dataset_non_numeric(sample_dataset):
    df = sample_dataset.copy()
    df["length of patch in mm"] = "not_a_number"
    result = data_loader.validate_dataset(df)
    assert result.valid is False


def test_validate_dataset_frequency_out_of_range(sample_dataset):
    df = sample_dataset.copy()
    df[config.FREQUENCY_COL] = 100.0  # Way out of range
    result = data_loader.validate_dataset(df)
    assert any("frequency" in w.lower() or "GHz" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# prepare_features_targets
# ---------------------------------------------------------------------------

def test_prepare_features_targets_shape(sample_dataset, sample_target_columns):
    X, y, cols = data_loader.prepare_features_targets(sample_dataset)
    assert X.shape == (len(sample_dataset), 1)
    assert y.shape == (len(sample_dataset), len(sample_target_columns))
    assert cols == sample_target_columns


def test_prepare_features_targets_missing_freq_col(sample_dataset):
    df = sample_dataset.drop(columns=[config.FREQUENCY_COL])
    with pytest.raises(KeyError, match=re.escape(config.FREQUENCY_COL)):
        data_loader.prepare_features_targets(df)


def test_prepare_features_targets_values(sample_dataset):
    X, y, _ = data_loader.prepare_features_targets(sample_dataset)
    np.testing.assert_array_equal(X[:, 0], sample_dataset[config.FREQUENCY_COL].values)


# ---------------------------------------------------------------------------
# split_and_scale
# ---------------------------------------------------------------------------

def test_split_and_scale_keys(sample_dataset, sample_target_columns):
    X, y, _ = data_loader.prepare_features_targets(sample_dataset)
    splits = data_loader.split_and_scale(X, y)
    expected_keys = {"X_train_scaled", "X_test_scaled", "y_train", "y_test", "scaler"}
    assert expected_keys.issubset(splits)


def test_split_and_scale_sizes(sample_dataset, sample_target_columns):
    X, y, _ = data_loader.prepare_features_targets(sample_dataset)
    splits = data_loader.split_and_scale(X, y)
    total = len(sample_dataset)
    test_n = int(total * config.TEST_SIZE)
    train_n = total - test_n
    assert len(splits["y_train"]) == train_n
    assert len(splits["y_test"]) == test_n


def test_split_and_scale_scaler_fitted(sample_dataset):
    X, y, _ = data_loader.prepare_features_targets(sample_dataset)
    splits = data_loader.split_and_scale(X, y)
    # Scaled training data should have approximately zero mean
    assert abs(splits["X_train_scaled"].mean()) < 0.5


# ---------------------------------------------------------------------------
# validate_uploaded_dataset
# ---------------------------------------------------------------------------

def test_validate_uploaded_dataset_valid(sample_dataset):
    result = data_loader.validate_uploaded_dataset(sample_dataset)
    assert result.valid is True
    assert result.errors == []


def test_validate_uploaded_dataset_missing_frequency_column(sample_dataset):
    df = sample_dataset.drop(columns=[config.FREQUENCY_COL])
    result = data_loader.validate_uploaded_dataset(df)
    assert result.valid is False
    assert any(config.FREQUENCY_COL in e for e in result.errors)


def test_validate_uploaded_dataset_non_numeric(sample_dataset):
    df = sample_dataset.copy()
    df["length of patch in mm"] = "text"
    result = data_loader.validate_uploaded_dataset(df)
    assert result.valid is False
    assert any("non-numeric" in e.lower() or "Non-numeric" in e for e in result.errors)


def test_validate_uploaded_dataset_nan_values(sample_dataset):
    df = sample_dataset.copy()
    df.iloc[0, 1] = float("nan")
    result = data_loader.validate_uploaded_dataset(df)
    assert result.valid is False
    assert any("nan" in e.lower() or "NaN" in e for e in result.errors)


def test_validate_uploaded_dataset_too_few_rows():
    df = pd.DataFrame({
        config.FREQUENCY_COL: [2.0, 3.0],
        "col_a": [1.0, 2.0],
    })
    result = data_loader.validate_uploaded_dataset(df)
    assert result.valid is False
    assert any("rows" in e.lower() for e in result.errors)


def test_validate_uploaded_dataset_collects_all_errors():
    """Ensure all violations are reported, not just the first one."""
    df = pd.DataFrame({
        "bad_col": ["a", "b"],
    })
    result = data_loader.validate_uploaded_dataset(df)
    assert result.valid is False
    # Should have errors for: missing frequency col, non-numeric, too few rows, no targets
    assert len(result.errors) >= 3


def test_validate_uploaded_dataset_frequency_only():
    """A dataset with only the frequency column and no targets should be rejected."""
    df = pd.DataFrame({
        config.FREQUENCY_COL: [float(i) for i in range(20)],
    })
    result = data_loader.validate_uploaded_dataset(df)
    assert result.valid is False
    assert any("target column" in e.lower() for e in result.errors)
