# Product Overview

antenna-ml is a machine learning pipeline that predicts antenna physical design parameters from an operating frequency input. It replaces expensive electromagnetic simulations with a trained Random Forest regression model, enabling instant antenna geometry predictions.

## What It Does

- Takes a single input: frequency in GHz (2.0–7.0)
- Will later expand to taking more inputs, e.g Length constraints.
- Predicts 6 antenna design parameters: patch length, substrate dimensions, slot area, circular slot radius, and S11 return loss
- Serves predictions through a Gradio web UI on port 7860

## Core Workflow

1. Train a base Random Forest model on antenna simulation samples (`train_model.py`)
2. Optionally tune hyperparameters via GridSearchCV (`tune_hyperparameters.py`)
3. Serve predictions through an interactive web interface (`gradio_app.py`)

## Key Concepts

- The dataset (`dataset_WIFI7.csv`) contains simulation data mapping frequency to antenna geometry
- Two model variants exist: "base" (default params) and "tuned" (GridSearchCV optimized)
- The app auto-selects the tuned model if available, falling back to base
- All model artifacts (.pkl files) are generated, not source-controlled
