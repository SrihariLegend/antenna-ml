# Project Structure

```
antenna-ml/
├── config.py                  # Centralized constants, paths, hyperparams, logging setup
├── data_loader.py             # Dataset loading, validation, feature/target prep, train/test split
├── constraint_checker.py      # Dimension constraint validation and bound checking
├── dataset_registry.py        # Dataset CRUD operations (JSON-backed registry)
├── model_io.py                # Model save/load with integrity checks, best-model selection
├── train_model.py             # Training pipeline entry point (load → train → evaluate → save)
├── tune_hyperparameters.py    # GridSearchCV tuning pipeline entry point
├── gradio_app.py              # Tabbed Gradio web UI (Prediction + Dataset Management)
├── build_exe.py               # PyInstaller packaging script
├── dataset_WIFI7.csv          # Source dataset (~3000 samples)
├── datasets/                  # Uploaded user datasets
├── dataset_registry.json      # Dataset registry (auto-generated)
├── run.sh                     # Docker helper script for all operations
├── Dockerfile                 # Python 3.11-slim container definition
├── docker-compose.yml         # Single-service compose config
├── requirements.txt           # Pinned Python dependencies
├── pytest.ini                 # pytest config (test paths, coverage flags)
├── .coveragerc                # Coverage settings (80% minimum)
├── tests/
│   ├── conftest.py            # Shared fixtures (sample data, models, temp dirs)
│   ├── test_config.py         # Tests for config constants and logging
│   ├── test_constraint_checker.py  # Tests for constraint validation
│   ├── test_data_loader.py    # Tests for loading, validation, splitting
│   ├── test_dataset_registry.py    # Tests for dataset registry CRUD
│   ├── test_model_io.py       # Tests for save/load/find_best_model
│   ├── test_train_model.py    # Tests for training pipeline
│   ├── test_tune_hyperparameters.py  # Tests for tuning pipeline
│   ├── test_gradio_app.py     # Tests for Gradio app creation and prediction
│   └── test_integration.py    # End-to-end integration tests
└── Generated artifacts (git-ignored):
    ├── rf_antenna_model.pkl        # Base trained model
    ├── rf_antenna_model_tuned.pkl  # Tuned model (optional)
    ├── rf_antenna_model_<name>.pkl # Dataset-specific models
    ├── scaler_X.pkl                # Fitted StandardScaler
    ├── target_columns.pkl          # Target column name list
    └── prediction_results.png      # Actual vs predicted scatter plots
```

## Architecture

The codebase is a flat Python module structure (no packages/subdirectories for source code):

- `config.py` is the single source of truth for all constants, file paths, and hyperparameters
- `data_loader.py` and `model_io.py` are utility modules imported by the pipeline scripts
- `train_model.py` and `tune_hyperparameters.py` are standalone entry points with `main()` functions
- `gradio_app.py` uses a factory pattern (`create_app()`) to avoid side effects on import

## Conventions

- All modules use Python `logging` configured via `config.setup_logging()`
- Model artifacts use a prefix system ("base" / "tuned") defined in `config.MODEL_FILES`
- Dataclasses are used for structured return types (e.g., `ValidationResult`)
- Type hints are used throughout (Python 3.11+ syntax like `list[str]`, `str | None`)
- Tests use `monkeypatch` and `tmp_path` fixtures to isolate file I/O from the real filesystem
- Matplotlib uses the `Agg` backend for headless rendering in Docker
