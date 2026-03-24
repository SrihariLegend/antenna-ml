"""Tests for config.py."""

import logging

import config


def test_freq_min_less_than_max():
    assert config.FREQ_MIN < config.FREQ_MAX


def test_expected_columns_length():
    # 1 frequency column + 7 target columns
    assert len(config.EXPECTED_COLUMNS) == 8


def test_frequency_col_in_expected_columns():
    assert config.FREQUENCY_COL in config.EXPECTED_COLUMNS


def test_model_files_has_base_and_tuned():
    assert "base" in config.MODEL_FILES
    assert "tuned" in config.MODEL_FILES


def test_model_files_keys():
    for variant in config.MODEL_FILES.values():
        assert "model" in variant
        assert "scaler" in variant
        assert "target_columns" in variant


def test_param_grid_not_empty():
    assert len(config.PARAM_GRID) > 0


def test_default_rf_params_has_required_keys():
    required = {"n_estimators", "random_state", "n_jobs"}
    assert required.issubset(config.DEFAULT_RF_PARAMS)


def test_test_size_between_0_and_1():
    assert 0.0 < config.TEST_SIZE < 1.0


def test_setup_logging_configures_root_logger():
    config.setup_logging(level=logging.DEBUG)
    root = logging.getLogger()
    assert root.level == logging.DEBUG


def test_setup_logging_with_file(tmp_path):
    log_file = str(tmp_path / "test.log")
    config.setup_logging(log_file=log_file)
    logging.getLogger("test_setup").info("hello")
    import os
    assert os.path.exists(log_file)


def test_no_duplicate_expected_columns():
    assert len(config.EXPECTED_COLUMNS) == len(set(config.EXPECTED_COLUMNS))


def test_width_of_patch_column_present():
    # Ensures the renamed column (was a duplicate) is correctly defined
    assert "width of patch in mm" in config.EXPECTED_COLUMNS
