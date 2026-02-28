# 🛜 antenna-ml

A machine learning pipeline for predicting antenna physical parameters from operating frequency, using a Random Forest regression model with a Gradio web interface, and fully containerized with Docker.

## What it does

Antennas must be physically designed for specific frequency bands. Instead of running expensive EM simulations for every design iteration, this model learns the relationship between **frequency → antenna geometry** from simulation data and predicts design parameters instantly.

**Input:** Frequency in GHz (2.0 – 7.0 GHz)

**Output:** Predicted antenna design parameters:
| Parameter | Description |
|---|---|
| Length of patch (mm) | Radiating patch length |
| Width of patch (mm) | Radiating patch width |
| Length of substrate (mm) | Dielectric substrate length |
| Width of substrate (mm) | Dielectric substrate width |
| Area of slots (mm²) | Total slot area on patch |
| Radius of circular slot (mm) | Circular slot radius |
| S11 (dB) | Return loss |

## Project structure

```
antenna-ml/
├── dataset_WIFI7.csv          # ~3000-sample WiFi 7 antenna simulation dataset
├── config.py                  # Shared constants (frequency range, file paths, hyperparameters)
├── train_model.py             # Train base Random Forest model
├── tune_hyperparameters.py    # GridSearchCV hyperparameter tuning
├── gradio_app.py              # Interactive web UI with sweep plot
├── run.sh                     # Docker helper script (all operations)
├── Dockerfile                 # Python 3.11-slim container (non-root user)
├── docker-compose.yml         # Service definition (port 7860)
├── requirements.txt           # Python dependencies
└── tests/                     # Pytest unit tests
    ├── conftest.py
    └── test_antenna_ml.py
```

## Quickstart

**Prerequisites:** Docker and Docker Compose installed.

```bash
# 1. Build and start the container
./run.sh start

# 2. Train the model (takes ~1 min)
./run.sh train

# 3. Launch the web app
./run.sh app
# → Open http://localhost:7860
```

Optionally, run hyperparameter tuning after the base model is trained (takes 5-10 min):
```bash
./run.sh tune
```
The app automatically picks up the tuned model (`rf_antenna_model_tuned.pkl`) if it exists.

## run.sh command reference

```
Container management:
  ./run.sh start          Build + start container (detached)
  ./run.sh stop           Stop container
  ./run.sh restart        Restart container
  ./run.sh rebuild        Rebuild image (after changing Dockerfile/requirements)
  ./run.sh status         Container status + list saved model files
  ./run.sh logs           Follow container logs
  ./run.sh info           Full project info + Docker disk usage

ML workflow:
  ./run.sh train          Train base Random Forest model
  ./run.sh tune           GridSearchCV hyperparameter tuning (~5-10 min)
  ./run.sh app            Start Gradio web interface at :7860

Development:
  ./run.sh shell          bash shell inside container
  ./run.sh python         Python REPL inside container
  ./run.sh exec "cmd"     Run arbitrary command in container

File management:
  ./run.sh upload <file>    Copy local file into container
  ./run.sh download <file>  Copy file out of container
  ./run.sh backup           Timestamped backup of models + plots

Cleanup:
  ./run.sh clean          Remove .pkl and .png output files
  ./run.sh clean-all      Remove containers, images, and outputs
```

## Running tests

```bash
# Inside the container or with dependencies installed locally:
pytest tests/ -v
```

## Model details

- **Algorithm:** `RandomForestRegressor` (multi-output)
- **Feature:** `Frequency(GHz)` — single input
- **Targets:** 7 antenna design parameters
- **Split:** 80% train / 20% test, `random_state=42`
- **Feature scaling:** `StandardScaler`
- **Evaluation:** R², RMSE, MAE per parameter + overall

### Hyperparameter search space

| Parameter | Values |
|---|---|
| `n_estimators` | 50, 100, 200 |
| `max_depth` | None, 10, 20, 30 |
| `min_samples_split` | 2, 5, 10 |
| `min_samples_leaf` | 1, 2, 4 |
| `max_features` | sqrt, log2, None |

Total combinations: 324 × 5-fold CV, selected by R² score.

## Saved artifacts

After training, the following files are created in the project directory (mounted as a Docker volume, so they persist on the host):

| File | Contents |
|---|---|
| `rf_antenna_model.pkl` | Trained base Random Forest |
| `rf_antenna_model_tuned.pkl` | Tuned model (if `tune` was run) |
| `scaler_X.pkl` | Fitted `StandardScaler` |
| `target_columns.pkl` | List of target column names |
| `prediction_results.png` | Actual vs predicted scatter plots |

## Stack

- Python 3.11
- scikit-learn 1.3.2
- pandas 2.1.4 · numpy 1.26.2
- Gradio 4.16
- matplotlib 3.8.2 · seaborn 0.13.0
- joblib 1.3.2
- Docker (python:3.11-slim, non-root user)
