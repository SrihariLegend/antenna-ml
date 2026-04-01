# 📡 antenna-ml

Machine learning pipeline for predicting rectangular patch antenna S11 (return loss) from operating frequency and geometry constraints. Uses a Random Forest model trained on CST Studio simulation data, augmented with physics-informed interpolation. Includes a Gradio web interface with interactive 3D antenna visualisation and CST project generation.

## What it does

Given a target frequency (1.5–10.5 GHz) and optional size constraints, the model predicts the optimal antenna geometry (patch length, substrate height) that minimises S11, then displays the full design with an interactive 3D preview.

**Input:** Frequency (GHz), optional max patch length / substrate height constraints

**Output:**
| Parameter | Description |
|---|---|
| S11 (dB) | Predicted return loss at target frequency |
| Patch length (mm) | Radiating element length |
| Patch width (mm) | 38 mm (fixed, per CST simulation setup) |
| Substrate height (mm) | Dielectric substrate thickness |
| Substrate length (mm) | 2 × patch length |
| Substrate width (mm) | 76 mm (2 × patch width) |

## Quick start

```bash
git clone --recurse-submodules https://github.com/SrihariLegend/antenna-ml.git
cd antenna-ml
pip install -r requirements.txt

python augment_dataset.py   # generate augmented dataset (25k → 81k rows)
python train_model.py       # train the model (~1 min)
python gradio_app.py        # launch web UI at http://localhost:7860
```

Or with Docker:
```bash
./run.sh rebuild
./run.sh train
./run.sh app
```

## Project structure

```
antenna-ml/
├── config.py                  # Constants, paths, hyperparameters
├── data_loader.py             # Dataset loading and validation
├── train_model.py             # Train Random Forest model
├── tune_hyperparameters.py    # GridSearchCV hyperparameter tuning
├── augment_dataset.py         # Multi-fidelity data augmentation
├── balanis.py                 # Antenna dimension helpers (CST geometry rules)
├── antenna_3d.py              # Interactive 3D Plotly visualiser (extensible)
├── gradio_app.py              # Gradio web UI (prediction + dataset management)
├── model_io.py                # Model save/load with integrity checks
├── dataset_registry.py        # Dataset CRUD (JSON-backed registry)
├── build_exe.py               # PyInstaller packaging
├── clean_dataset_rect.csv     # CST-simulated dataset (25,025 rows, 3 heights)
├── CST_Linker/                # Git submodule — CST project generator
├── dataset_generator/         # Rust-based Balanis analytical dataset generator
├── run.sh                     # Docker helper script
├── Dockerfile / docker-compose.yml
├── requirements.txt
└── tests/
```

## Dataset

The base dataset (`clean_dataset_rect.csv`) contains 25,025 CST Studio simulations with 4 columns:

| Column | Range | Description |
|---|---|---|
| `freq` | 1.5–10.5 GHz | Operating frequency |
| `S11` | -49 to 0 dB | Return loss |
| `patch_length` | 8–34.1 mm | Patch length (9 values) |
| `substrate_height` | 0.8, 2.0, 3.2 mm | Substrate thickness (3 values) |

### Multi-fidelity augmentation

The CST dataset only covers 3 substrate heights. `augment_dataset.py` uses quadratic interpolation between the CST data points to generate synthetic samples at 7 intermediate heights (1.0, 1.2, 1.4, 1.6, 1.8, 2.4, 2.8 mm), producing 81,081 total rows across 10 heights.

This approach is based on multi-fidelity surrogate modelling techniques from:
- Pietrenko-Dabrowska et al., "Two-stage variable-fidelity modeling of antennas with domain confinement," *Sci. Rep.* 12, 17275 (2022)

## Model

- **Algorithm:** Random Forest Regressor
- **Features:** freq, patch_length, substrate_height
- **Target:** S11 (dB)
- **Performance:** R² = 0.9959, MAE = 0.08 dB on test set
- **Validated against CST ground truth:** R² = 0.9980, MAE = 0.056 dB

### Known limitation

The training data uses 9 MHz frequency steps. Narrow resonance dips may be deeper in actual CST simulation than the model predicts. The predicted S11 is a conservative estimate — actual performance will be equal or better.

## 3D Visualiser

The app includes an interactive 3D antenna preview (Plotly) showing ground plane, substrate, and patch with dimension annotations. The visualiser uses a registry pattern for extensibility:

```python
@antenna_3d.register("circular")
def _render_circular(dims: dict, **kw) -> go.Figure:
    ...
```

Currently registered shapes: `rectangular`, `circular`.

## CST Linker

Included as a git submodule (`CST_Linker/`). After prediction, click "Launch CST Linker" to generate a `.cstprj` file with the predicted dimensions. Requires CST Studio Suite Python API on the target machine.

## run.sh commands

```
./run.sh start       Start Docker container
./run.sh stop        Stop container
./run.sh rebuild     Rebuild and start
./run.sh train       Train the model
./run.sh tune        Hyperparameter tuning (~15-30 min)
./run.sh app         Launch Gradio at :7860
./run.sh test        Run pytest with coverage
./run.sh build-exe   Package with PyInstaller
./run.sh shell       Bash shell in container
./run.sh clean       Remove generated .pkl/.png files
```

## Stack

- Python 3.11 / scikit-learn 1.3 / pandas 2.1 / numpy 1.26
- Gradio 4.16 / Plotly 5.18
- Docker (python:3.11-slim)
- Rust (dataset generator)
