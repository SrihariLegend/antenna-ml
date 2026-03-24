# Product Overview

antenna-ml is a machine learning pipeline that predicts antenna physical design parameters from an operating frequency input. It replaces expensive electromagnetic simulations with a trained Random Forest regression model, enabling instant antenna geometry predictions.

## What It Does

- Takes a single input: frequency in GHz (1.0–10.0)
- Supports optional dimension constraints (min/max bounds per parameter)
- Predicts 7 antenna design parameters: patch length/width, substrate dimensions, slot area, circular slot radius, and S11 return loss
- Serves predictions through a tabbed Gradio web UI on port 7860
- Supports custom dataset upload, validation, and per-dataset model training

## Core Workflow

1. Train a base Random Forest model on antenna simulation samples (`train_model.py`)
2. Optionally tune hyperparameters via GridSearchCV (`tune_hyperparameters.py`)
3. Serve predictions through an interactive tabbed web interface (`gradio_app.py`)
4. Upload custom datasets and train dataset-specific models from the UI
5. Optionally package as standalone executable via PyInstaller (`build_exe.py`)

## Key Concepts

- The dataset (`dataset_WIFI7.csv`) contains simulation data mapping frequency to antenna geometry
- Two model variants exist: "base" (default params) and "tuned" (GridSearchCV optimized)
- The app auto-selects the tuned model if available, falling back to base
- All model artifacts (.pkl files) are generated, not source-controlled
