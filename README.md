# 🛜 antenna-ml

A machine learning pipeline for predicting WiFi 7 antenna physical parameters from operating frequency, using a Random Forest regression model with a Gradio web interface — fully containerized with Docker.

## What it does

WiFi 7 antennas must be physically designed for specific frequency bands (2.4, 5, and 6 GHz). Instead of running expensive EM simulations for every design iteration, this model learns the relationship between **frequency → antenna geometry** from simulation data and predicts design parameters instantly.

<<<<<<< HEAD
**Input:** Frequency in GHz (1.0 – 10.0 GHz), with optional dimension constraints (min/max bounds per parameter)
||||||| 78942b5
**Input:** Frequency in GHz (2.0 – 7.0 GHz)
=======
**Input:** Frequency in GHz (1.0 – 10.0 GHz)
>>>>>>> 356719de886ac371c45797dac38db998a5d2a8fc

**Output:** Predicted antenna design parameters with constraint status:
| Parameter | Description |
|---|---|
| Length of patch (mm) | Radiating patch length |
| Width of patch (mm) | Radiating patch width |
| Length of substrate (mm) | Dielectric substrate length |
| Width of substrate (mm) | Dielectric substrate width |
| Area of slots (mm²) | Total slot area on patch |
| Radius of circular slot (mm) | Circular slot dimensions |
| S11 (dB) | Return loss — measures how much signal is reflected |

Parameters that fall outside user-defined constraints are flagged with ⚠️ in the results.

## Project structure

```
antenna-ml/
├── config.py                  # Centralized constants, paths, hyperparams, logging
├── data_loader.py             # Dataset loading, validation, feature/target prep
├── constraint_checker.py      # Dimension constraint validation and checking
├── dataset_registry.py        # Dataset CRUD operations (JSON-backed registry)
├── model_io.py                # Model save/load with integrity checks
├── train_model.py             # Train base Random Forest model
├── tune_hyperparameters.py    # GridSearchCV hyperparameter tuning
├── gradio_app.py              # Tabbed Gradio web UI (Prediction + Dataset Management)
├── build_exe.py               # PyInstaller packaging script
├── dataset_WIFI7.csv          # ~3000-sample WiFi 7 antenna simulation dataset
├── datasets/                  # Uploaded user datasets
├── dataset_registry.json      # Dataset registry (auto-generated)
├── run.sh                     # Docker helper script (all operations)
├── Dockerfile                 # Python 3.11-slim container
├── docker-compose.yml         # Service definition (port 7860)
├── requirements.txt           # Python dependencies
└── tests/                     # pytest test suite (106 tests, 87%+ coverage)
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

## Web UI

The Gradio app at `http://localhost:7860` has two tabs:

- **Prediction** — Slide the frequency (1.0–10.0 GHz), optionally set min/max dimension constraints per parameter, and click Predict. Results show a table with predicted values and constraint status.
- **Dataset Management** — Upload custom CSV datasets, select a registered dataset, and train a new model. Training metrics (R², MSE, MAE per parameter) display inline. The model hot-reloads after training.

## Packaging

Build a standalone executable with PyInstaller:
```bash
./run.sh build-exe
```
Output goes to `dist/antenna-ml/`.

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
  ./run.sh build-exe      Package as standalone executable via PyInstaller

Development:
  ./run.sh test           Run pytest with coverage
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

## Model details

- **Algorithm:** `RandomForestRegressor` (multi-output)
- **Feature:** `Frequency(GHz)` — single input
- **Targets:** 6 antenna design parameters
- **Split:** 80% train / 20% test, `random_state=42`
- **Feature scaling:** `StandardScaler`
- **Evaluation:** R², MSE, MAE per parameter + overall

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
| `rf_antenna_model_<dataset>.pkl` | Dataset-specific model (if trained from UI) |
| `scaler_X.pkl` | Fitted `StandardScaler` |
| `target_columns.pkl` | List of target column names |
| `prediction_results.png` | Actual vs predicted scatter plots |
| `dataset_registry.json` | Registered datasets metadata |

## Stack

- Python 3.11
- scikit-learn 1.3.2
- pandas 2.1.4 · numpy 1.26.2
- Gradio 4.16
- matplotlib 3.8.2 · seaborn 0.13.0
- joblib 1.3.2
- hypothesis 6.112.1 (property-based testing)
- PyInstaller 6.11.1 (packaging)
- Docker (python:3.11-slim)
