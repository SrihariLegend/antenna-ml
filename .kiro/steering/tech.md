# Tech Stack & Build System

## Language & Runtime

- Python 3.11 (via `python:3.11-slim` Docker image)

## Key Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| scikit-learn | 1.3.2 | RandomForestRegressor, StandardScaler, GridSearchCV, metrics |
| pandas | 2.1.4 | Dataset loading and manipulation |
| numpy | 1.26.2 | Numerical arrays |
| gradio | 4.16.0 | Web UI for predictions |
| matplotlib | 3.8.2 | Visualization (actual vs predicted plots) |
| seaborn | 0.13.0 | Plot styling |
| joblib | 1.3.2 | Model serialization (.pkl files) |
| pytest | 7.4.4 | Test framework |
| pytest-cov | 4.1.0 | Coverage reporting |
| hypothesis | 6.112.1 | Property-based testing |
| pyinstaller | 6.11.1 | Standalone executable packaging |

## Containerization

- Docker with Docker Compose
- Single service `ml` (container name: `antenna_ml`)
- Project root mounted as volume at `/app`
- Port 7860 exposed for Gradio
- `run.sh` wraps all Docker operations

## Common Commands

All commands run inside Docker via `run.sh`:

```bash
# Container lifecycle
./run.sh start          # Build + start container
./run.sh stop           # Stop container
./run.sh rebuild        # Rebuild after dependency changes

# ML workflow
./run.sh train          # Train base model (~1 min)
./run.sh tune           # Hyperparameter tuning (~5-10 min)
./run.sh app            # Launch Gradio UI at localhost:7860
./run.sh build-exe      # Package as standalone executable

# Testing
./run.sh test           # pytest with coverage
# Or directly:
python -m pytest tests/ -v --cov

# Dev access
./run.sh shell          # Bash inside container
./run.sh exec "cmd"     # Run arbitrary command in container
```

## Testing

- Framework: pytest
- Test directory: `tests/`
- Config: `pytest.ini` — auto-runs with `--cov` and `--cov-report=term-missing`
- Coverage config: `.coveragerc` — minimum 80% coverage, excludes `tests/` and `setup.py`
- Fixtures in `tests/conftest.py` provide sample data, trained models, and temp artifact directories

## Dependency Management

- `requirements.txt` — pinned versions, no lock file
- To update: edit `requirements.txt`, then `./run.sh rebuild`
