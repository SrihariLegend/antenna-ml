"""Unit tests for dataset_registry module."""

import json
import os

import numpy as np
import pandas as pd
import pytest

import config
import dataset_registry


@pytest.fixture
def tmp_registry(tmp_path, monkeypatch):
    """Patch registry path and datasets dir to use a temp directory."""
    registry_path = str(tmp_path / "dataset_registry.json")
    datasets_dir = str(tmp_path / "datasets")
    monkeypatch.setattr(config, "DATASET_REGISTRY_PATH", registry_path)
    monkeypatch.setattr(config, "DATASETS_DIR", datasets_dir)
    return tmp_path


@pytest.fixture
def small_df():
    """Return a small valid DataFrame for registration."""
    rng = np.random.default_rng(42)
    n = 15
    data = {config.FREQUENCY_COL: rng.uniform(1.0, 10.0, n)}
    for col in config.EXPECTED_COLUMNS:
        if col != config.FREQUENCY_COL:
            data[col] = rng.uniform(1.0, 100.0, n)
    return pd.DataFrame(data)


class TestLoadRegistry:
    """Tests for load_registry()."""

    def test_returns_empty_when_file_missing(self, tmp_registry):
        result = dataset_registry.load_registry()
        assert result == {"datasets": {}}

    def test_reads_existing_registry(self, tmp_registry):
        registry_path = config.DATASET_REGISTRY_PATH
        data = {"datasets": {"test": {"path": "test.csv", "columns": [], "row_count": 0, "freq_min": 0.0, "freq_max": 0.0, "added_at": "2024-01-01T00:00:00"}}}
        with open(registry_path, "w") as f:
            json.dump(data, f)

        result = dataset_registry.load_registry()
        assert result == data

    def test_returns_empty_on_corrupted_json(self, tmp_registry):
        with open(config.DATASET_REGISTRY_PATH, "w") as f:
            f.write("{invalid json")

        result = dataset_registry.load_registry()
        assert result == {"datasets": {}}

    def test_adds_datasets_key_if_missing(self, tmp_registry):
        with open(config.DATASET_REGISTRY_PATH, "w") as f:
            json.dump({"other_key": 123}, f)

        result = dataset_registry.load_registry()
        assert "datasets" in result


class TestSaveRegistry:
    """Tests for save_registry()."""

    def test_writes_json_file(self, tmp_registry):
        data = {"datasets": {"ds1": {"path": "ds1.csv", "columns": ["a"], "row_count": 5, "freq_min": 1.0, "freq_max": 5.0, "added_at": "2024-01-01T00:00:00"}}}
        dataset_registry.save_registry(data)

        with open(config.DATASET_REGISTRY_PATH, "r") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_round_trip(self, tmp_registry):
        data = {"datasets": {"alpha": {"path": "a.csv", "columns": ["x", "y"], "row_count": 10, "freq_min": 2.0, "freq_max": 8.0, "added_at": "2024-06-01T12:00:00"}}}
        dataset_registry.save_registry(data)
        result = dataset_registry.load_registry()
        assert result == data


class TestListDatasets:
    """Tests for list_datasets()."""

    def test_empty_registry_returns_empty_list(self, tmp_registry):
        assert dataset_registry.list_datasets() == []

    def test_returns_sorted_names(self, tmp_registry):
        data = {"datasets": {
            "Zebra": {"path": "z.csv", "columns": [], "row_count": 0, "freq_min": 0.0, "freq_max": 0.0, "added_at": ""},
            "Alpha": {"path": "a.csv", "columns": [], "row_count": 0, "freq_min": 0.0, "freq_max": 0.0, "added_at": ""},
            "Middle": {"path": "m.csv", "columns": [], "row_count": 0, "freq_min": 0.0, "freq_max": 0.0, "added_at": ""},
        }}
        dataset_registry.save_registry(data)
        assert dataset_registry.list_datasets() == ["Alpha", "Middle", "Zebra"]


class TestGetDatasetPath:
    """Tests for get_dataset_path()."""

    def test_returns_path_for_registered_dataset(self, tmp_registry):
        data = {"datasets": {"myds": {"path": "datasets/myds.csv", "columns": [], "row_count": 0, "freq_min": 0.0, "freq_max": 0.0, "added_at": ""}}}
        dataset_registry.save_registry(data)
        assert dataset_registry.get_dataset_path("myds") == "datasets/myds.csv"

    def test_raises_key_error_for_unknown_name(self, tmp_registry):
        with pytest.raises(KeyError, match="not found"):
            dataset_registry.get_dataset_path("nonexistent")


class TestRegisterDataset:
    """Tests for register_dataset()."""

    def test_registers_new_dataset(self, tmp_registry, small_df):
        dataset_registry.register_dataset("TestDS", "datasets/test.csv", small_df)

        names = dataset_registry.list_datasets()
        assert "TestDS" in names

        path = dataset_registry.get_dataset_path("TestDS")
        assert path == "datasets/test.csv"

    def test_stores_correct_metadata(self, tmp_registry, small_df):
        dataset_registry.register_dataset("Meta", "datasets/meta.csv", small_df)

        registry = dataset_registry.load_registry()
        entry = registry["datasets"]["Meta"]
        assert entry["columns"] == list(small_df.columns)
        assert entry["row_count"] == len(small_df)
        assert entry["freq_min"] == float(small_df[config.FREQUENCY_COL].min())
        assert entry["freq_max"] == float(small_df[config.FREQUENCY_COL].max())
        assert "added_at" in entry

    def test_raises_value_error_on_duplicate(self, tmp_registry, small_df):
        dataset_registry.register_dataset("Dup", "datasets/dup.csv", small_df)
        with pytest.raises(ValueError, match="already exists"):
            dataset_registry.register_dataset("Dup", "datasets/dup2.csv", small_df)

    def test_handles_df_without_frequency_col(self, tmp_registry):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        dataset_registry.register_dataset("NoFreq", "datasets/nofreq.csv", df)

        registry = dataset_registry.load_registry()
        entry = registry["datasets"]["NoFreq"]
        assert entry["freq_min"] == 0.0
        assert entry["freq_max"] == 0.0


class TestInitDefaultDataset:
    """Tests for init_default_dataset()."""

    def test_registers_default_on_first_run(self, tmp_registry, monkeypatch):
        # Point DATASET_PATH to a real CSV
        csv_path = str(tmp_registry / "default.csv")
        rng = np.random.default_rng(0)
        n = 20
        data = {config.FREQUENCY_COL: rng.uniform(2.0, 7.0, n)}
        for col in config.EXPECTED_COLUMNS:
            if col != config.FREQUENCY_COL:
                data[col] = rng.uniform(1.0, 50.0, n)
        pd.DataFrame(data).to_csv(csv_path, index=False)
        monkeypatch.setattr(config, "DATASET_PATH", csv_path)

        dataset_registry.init_default_dataset()

        names = dataset_registry.list_datasets()
        assert config.DEFAULT_DATASET_NAME in names

    def test_skips_if_already_registered(self, tmp_registry, monkeypatch, small_df):
        csv_path = str(tmp_registry / "default.csv")
        small_df.to_csv(csv_path, index=False)
        monkeypatch.setattr(config, "DATASET_PATH", csv_path)

        dataset_registry.init_default_dataset()
        # Get the added_at timestamp from first registration
        registry = dataset_registry.load_registry()
        first_ts = registry["datasets"][config.DEFAULT_DATASET_NAME]["added_at"]

        # Call again — should not re-register
        dataset_registry.init_default_dataset()
        registry = dataset_registry.load_registry()
        assert registry["datasets"][config.DEFAULT_DATASET_NAME]["added_at"] == first_ts

    def test_skips_if_csv_missing(self, tmp_registry, monkeypatch):
        monkeypatch.setattr(config, "DATASET_PATH", str(tmp_registry / "missing.csv"))
        dataset_registry.init_default_dataset()
        assert dataset_registry.list_datasets() == []

    def test_creates_datasets_directory(self, tmp_registry):
        datasets_dir = config.DATASETS_DIR
        assert not os.path.exists(datasets_dir)
        # Even if default CSV is missing, the directory should be created
        dataset_registry.init_default_dataset()
        assert os.path.isdir(datasets_dir)
