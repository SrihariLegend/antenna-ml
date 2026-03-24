# Implementation Plan: Expanded Prediction Pipeline

## Overview

Incrementally expand the antenna-ml application: update config constants, add new modules (constraint_checker, dataset_registry), extend existing modules (data_loader, train_model, model_io), migrate the Gradio UI to a tabbed Blocks layout, and add PyInstaller packaging. Each task builds on the previous, with property-based tests validating correctness properties from the design.

## Tasks

- [x] 1. Update `config.py` with expanded constants
  - [x] 1.1 Update frequency range and add dataset/constraint/packaging constants
    - Change `FREQ_MIN` from `2.0` to `1.0` and `FREQ_MAX` from `7.0` to `10.0`
    - Add `DATASETS_DIR`, `DATASET_REGISTRY_PATH`, `MIN_DATASET_ROWS`, `DEFAULT_DATASET_NAME`
    - Add `CONSTRAINT_PARAMS` dict mapping parameter names to units
    - Add `PYINSTALLER_ENTRY`, `PYINSTALLER_NAME`, `PYINSTALLER_BUNDLE_DATA`
    - Include descriptive comments for each new constant
    - _Requirements: 1.5, 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ]* 1.2 Write unit tests for new config constants
    - Verify `FREQ_MIN == 1.0` and `FREQ_MAX == 10.0`
    - Verify `DATASETS_DIR`, `DATASET_REGISTRY_PATH`, `MIN_DATASET_ROWS` exist and have correct types
    - Verify `CONSTRAINT_PARAMS` contains all 7 antenna parameter keys
    - _Requirements: 1.5, 8.1, 8.2, 8.3_

- [x] 2. Create `constraint_checker.py` module
  - [x] 2.1 Implement `ConstraintResult` dataclass, `validate_constraints`, and `apply_constraints`
    - Define `ConstraintResult` dataclass with fields: `parameter`, `predicted_value`, `min_bound`, `max_bound`, `in_bounds`
    - Implement `validate_constraints(constraints) -> list[str]` that checks min <= max for each constraint
    - Implement `apply_constraints(predictions, constraints) -> list[ConstraintResult]` that checks each prediction against bounds
    - Parameters with no constraint entry are marked `in_bounds=True`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.6_

  - [ ]* 2.2 Write property test: Empty constraints preserve unconstrained prediction
    - **Property 3: Empty constraints preserve unconstrained prediction**
    - Use `st.floats(min_value=1.0, max_value=10.0)` to generate frequencies
    - Verify all `ConstraintResult` entries have `in_bounds=True` when constraints dict is empty
    - **Validates: Requirements 2.2**

  - [ ]* 2.3 Write property test: Constraint bound checking correctness
    - **Property 4: Constraint bound checking correctness**
    - Use `st.dictionaries` with parameter names and `st.tuples(optional_float, optional_float)` for bounds
    - Verify `in_bounds=True` iff predicted value satisfies all supplied bounds
    - **Validates: Requirements 2.3, 2.4**

  - [ ]* 2.4 Write property test: Invalid constraint bounds are rejected
    - **Property 5: Invalid constraint bounds are rejected**
    - Generate float pairs where min > max
    - Verify `validate_constraints` returns non-empty error list referencing the offending parameter
    - **Validates: Requirements 2.6**

- [x] 3. Create `dataset_registry.py` module
  - [x] 3.1 Implement registry CRUD functions
    - Implement `load_registry() -> dict` — reads JSON file, creates empty registry if missing
    - Implement `save_registry(registry: dict) -> None` — writes registry dict to JSON
    - Implement `list_datasets() -> list[str]` — returns sorted dataset names
    - Implement `get_dataset_path(name: str) -> str` — returns path, raises `KeyError` if not found
    - Implement `register_dataset(name, path, df) -> None` — adds entry with metadata, raises `ValueError` on duplicate name
    - Implement `init_default_dataset() -> None` — ensures default dataset is registered on first run
    - _Requirements: 3.1, 3.2, 3.3, 4.3, 4.5_

  - [ ]* 3.2 Write property test: Registry serialization round-trip
    - **Property 6: Dataset registry serialization round-trip**
    - Generate valid registry dicts with `st.dictionaries`
    - Verify `save_registry` then `load_registry` produces equal dict
    - **Validates: Requirements 3.2**

  - [ ]* 3.3 Write property test: Dataset registration and retrieval
    - **Property 7: Dataset registration and retrieval**
    - Generate random unique dataset names and valid DataFrames
    - Verify registered name appears in `list_datasets()` and `get_dataset_path` returns correct path
    - **Validates: Requirements 4.3**

- [x] 4. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Extend `data_loader.py` with upload validation
  - [x] 5.1 Add `validate_uploaded_dataset` function
    - Implement `validate_uploaded_dataset(df: pd.DataFrame) -> ValidationResult`
    - Check: frequency column exists, all columns numeric, no NaN values, at least `config.MIN_DATASET_ROWS` rows
    - Return `ValidationResult` with all detected errors collected (not fail-fast)
    - Update existing `validate_dataset` frequency range check to use new `FREQ_MIN`/`FREQ_MAX` values
    - _Requirements: 4.2, 4.4, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

  - [ ]* 5.2 Write property test: Dataset validation catches all violations
    - **Property 8: Dataset validation catches all violations**
    - Generate DataFrames with randomly injected violations (missing frequency col, non-numeric cols, NaN values, too few rows)
    - Verify errors list contains entry for each present violation; clean DataFrames produce `valid=True`
    - **Validates: Requirements 4.2, 4.4, 5.1, 5.2, 5.3, 5.4, 5.5**

- [x] 6. Refactor `train_model.py` with reusable training function
  - [x] 6.1 Extract `train_on_dataset` function and `TrainingMetrics` dataclass
    - Create `TrainingMetrics` dataclass with `train_r2`, `test_r2`, `train_mse`, `test_mse`, `per_param`
    - Extract `train_on_dataset(dataset_path: str, model_prefix: str) -> dict` from existing `main()`
    - Returns dict with overall and per-parameter metrics (R², MSE, MAE)
    - Refactor `main()` to be a thin wrapper calling `train_on_dataset(config.DATASET_PATH, "base")`
    - _Requirements: 3.4, 7.2, 7.4, 7.6_

  - [ ]* 6.2 Write property test: Training produces complete metrics
    - **Property 9: Training produces complete metrics**
    - Use a valid synthetic dataset with sufficient rows
    - Verify returned dict contains `train_r2`, `test_r2`, `train_mse`, `test_mse` (all floats) and `per_param` list with correct structure
    - **Validates: Requirements 3.4, 7.2, 7.4**

- [x] 7. Extend `model_io.py` with dynamic prefix support
  - [x] 7.1 Implement `_artifact_paths` and update `save_model`/`load_model`/`find_best_model`
    - Add `_artifact_paths(prefix: str) -> dict[str, str]` — uses `config.MODEL_FILES` for known prefixes, generates convention-based paths otherwise
    - Update `save_model` and `load_model` to use `_artifact_paths` instead of direct `config.MODEL_FILES` lookup
    - Update `find_best_model` to also search for dataset-specific prefixes
    - _Requirements: 7.6_

  - [ ]* 7.2 Write property test: Dataset-specific prefixes produce unique artifact paths
    - **Property 10: Dataset-specific model prefixes produce unique artifact paths**
    - Generate pairs of distinct dataset name strings
    - Verify `_artifact_paths` returns completely disjoint file path sets for each
    - **Validates: Requirements 7.6**

- [x] 8. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Migrate `gradio_app.py` to `gr.Blocks` with tabbed layout
  - [x] 9.1 Build Prediction Tab
    - Replace `gr.Interface` with `gr.Blocks` and `gr.Tabs`
    - Create Prediction tab with frequency slider (1.0–10.0 GHz range)
    - Add collapsible `gr.Accordion` for dimension constraints (one row per parameter: label, min `gr.Number`, max `gr.Number`)
    - Implement predict callback that calls `constraint_checker.validate_constraints`, then `apply_constraints`
    - Render Markdown output with predictions table; flagged parameters show ⚠️ icon
    - Handle graceful startup when no model is available (prompt user to train)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 6.5_

  - [x] 9.2 Build Dataset Management Tab
    - Add Dataset Management tab with dataset dropdown populated from `dataset_registry.list_datasets()`
    - Add `gr.File` upload component (CSV only) with Upload button
    - Wire upload callback: validate via `data_loader.validate_uploaded_dataset`, register via `dataset_registry.register_dataset`, refresh dropdown
    - Add "Train Model" button that calls `train_model.train_on_dataset` with selected dataset
    - Add status textbox for progress/errors and metrics display (R², MSE, MAE per parameter)
    - Hot-reload model artifacts after training completes
    - _Requirements: 3.1, 3.3, 3.4, 3.5, 3.6, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ]* 9.3 Write property tests for prediction function frequency handling
    - **Property 1: In-range frequency produces complete predictions**
    - Use `st.floats(min_value=1.0, max_value=10.0)` to generate frequencies
    - Verify prediction returns a value for every target parameter with no missing/null values
    - **Validates: Requirements 1.1, 1.2**

  - [ ]* 9.4 Write property test for out-of-range frequency warning
    - **Property 2: Out-of-range frequency produces warning**
    - Use `st.floats().filter(lambda f: f < 1.0 or f > 10.0)` to generate out-of-range frequencies
    - Verify result is a warning string containing the supported range boundaries
    - **Validates: Requirements 1.3**

- [x] 10. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Add PyInstaller binary packaging
  - [x] 11.1 Create `build_exe.py` script and update `run.sh`
    - Create `build_exe.py` (or `antenna-ml.spec`) configuring PyInstaller in single-directory mode
    - Set entry point to `gradio_app.py`, output name `antenna-ml`
    - Configure hidden imports: `sklearn`, `gradio`, `pandas`, `numpy`, `joblib`
    - Bundle data files: default dataset CSV, `dataset_registry.json`, `datasets/` dir, `*.pkl` model files, Gradio static assets
    - Add `build-exe` command to `run.sh`
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.6, 6.7_

  - [x] 11.2 Initialize default dataset registry on startup
    - Call `dataset_registry.init_default_dataset()` at app startup in `gradio_app.py`
    - Ensure `datasets/` directory is created if missing
    - _Requirements: 3.2, 3.3, 6.4_

- [x] 12. Add `hypothesis` to `requirements.txt`
  - Add `hypothesis` library to `requirements.txt` for property-based testing
  - _Requirements: Testing strategy from design_

- [x] 13. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation after major milestones
- Property tests validate universal correctness properties from the design document
- The design uses Python throughout, so all implementation tasks use Python
- All new modules follow the existing flat structure convention with `logging` and type hints
