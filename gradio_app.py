"""Gradio web interface for predicting WiFi 7 antenna parameters."""

import logging
import os
import shutil

import gradio as gr
import numpy as np
import pandas as pd

import config
import constraint_checker
import data_loader
import dataset_registry
import model_io
import train_model

config.setup_logging()
logger = logging.getLogger(__name__)

# Module-level state populated by create_app(); exposed for testing.
_model = None
_scaler = None
_target_columns = None
_model_loaded = False


def predict_antenna_parameters(frequency: float, *constraint_values) -> str:
    """Predict all antenna parameters for a given frequency with optional constraints.

    Args:
        frequency: Input frequency in GHz.
        *constraint_values: Pairs of (min, max) floats for each constraint parameter,
            flattened in the order of config.CONSTRAINT_PARAMS keys.

    Returns:
        Markdown-formatted table of predicted parameters, or a warning/error string.
    """
    global _model, _scaler, _target_columns, _model_loaded

    if not _model_loaded:
        return (
            "⚠️ No trained model available. "
            "Please go to the **Dataset Management** tab to select a dataset and train a model."
        )

    if frequency < config.FREQ_MIN or frequency > config.FREQ_MAX:
        return (
            f"⚠️ Frequency out of training range "
            f"({config.FREQ_MIN}–{config.FREQ_MAX} GHz)"
        )

    # Collect constraints from the flat list of min/max values
    param_names = list(config.CONSTRAINT_PARAMS.keys())
    constraints: dict[str, tuple[float | None, float | None]] = {}
    for i, param in enumerate(param_names):
        min_val = constraint_values[i * 2] if i * 2 < len(constraint_values) else None
        max_val = constraint_values[i * 2 + 1] if i * 2 + 1 < len(constraint_values) else None
        constraints[param] = (min_val, max_val)

    # Validate constraints (min <= max)
    errors = constraint_checker.validate_constraints(constraints)
    if errors:
        return "⚠️ **Constraint validation errors:**\n\n" + "\n".join(f"- {e}" for e in errors)

    # Run prediction
    try:
        freq_scaled = _scaler.transform([[frequency]])
        predictions_array = _model.predict(freq_scaled)[0]
    except Exception as exc:
        logger.error("Prediction failed for frequency %.3f GHz: %s", frequency, exc)
        return f"Error during prediction: {exc}"

    # Build predictions dict
    predictions = {col: float(predictions_array[i]) for i, col in enumerate(_target_columns)}

    # Apply constraints
    results = constraint_checker.apply_constraints(predictions, constraints)

    # Render markdown table
    output = f"### Predictions for {frequency} GHz\n\n"
    output += "| Parameter | Value | Status |\n"
    output += "|-----------|-------|--------|\n"
    for r in results:
        flag = "" if r.in_bounds else " ⚠️"
        output += f"| {r.parameter} | {r.predicted_value:.4f} | {flag} |\n"

    return output


def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks app with tabbed layout.

    Loads the best available model at call time (not at import time) so
    that importing this module does not trigger file I/O or side effects.

    Returns:
        Configured gr.Blocks instance ready to launch.
    """
    global _model, _scaler, _target_columns, _model_loaded

    # Ensure default dataset is registered and datasets/ directory exists
    dataset_registry.init_default_dataset()

    # Attempt to load model; graceful fallback if none available
    try:
        prefix = model_io.find_best_model()
        _model, _scaler, _target_columns = model_io.load_model(prefix)
        _model_loaded = True
        logger.info("Using %s model with %d target columns.", prefix, len(_target_columns))
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("No model available at startup: %s", exc)
        _model_loaded = False

    with gr.Blocks(
        title="🛜 WiFi 7 Antenna Parameter Predictor",
    ) as demo:
        gr.Markdown(
            "# 🛜 WiFi 7 Antenna Parameter Predictor\n\n"
            "Predict antenna parameters from operating frequency. "
            "This model uses Random Forest regression trained on WiFi 7 antenna simulation data."
        )

        with gr.Tabs():
            with gr.Tab("Prediction"):
                freq_slider = gr.Slider(
                    minimum=config.FREQ_MIN,
                    maximum=config.FREQ_MAX,
                    value=2.4,
                    step=0.01,
                    label="Frequency (GHz)",
                    info="WiFi 7 operates at 2.4, 5, and 6 GHz bands",
                )

                # Build constraint inputs inside a collapsible accordion
                constraint_inputs = []
                with gr.Accordion("Dimension Constraints (optional)", open=False):
                    param_names = list(config.CONSTRAINT_PARAMS.keys())
                    for param in param_names:
                        unit = config.CONSTRAINT_PARAMS[param]
                        with gr.Row():
                            gr.Markdown(f"**{param}** ({unit})")
                            min_input = gr.Number(
                                label=f"Min",
                                value=None,
                                minimum=0,
                            )
                            max_input = gr.Number(
                                label=f"Max",
                                value=None,
                                minimum=0,
                            )
                            constraint_inputs.extend([min_input, max_input])

                predict_btn = gr.Button("Predict", variant="primary")
                output_md = gr.Markdown(label="Predicted Antenna Parameters")

                predict_btn.click(
                    fn=predict_antenna_parameters,
                    inputs=[freq_slider] + constraint_inputs,
                    outputs=output_md,
                )

            with gr.Tab("Dataset Management"):
                gr.Markdown("### Dataset Management\nUpload new datasets, select a dataset, and train models.")

                # Dataset selection dropdown
                dataset_dropdown = gr.Dropdown(
                    choices=dataset_registry.list_datasets(),
                    label="Select Dataset",
                    info="Choose a registered dataset for training",
                )

                # File upload section
                with gr.Row():
                    file_upload = gr.File(
                        label="Upload CSV Dataset",
                        file_types=[".csv"],
                        type="filepath",
                    )
                    upload_btn = gr.Button("Upload", variant="secondary")

                # Train button
                train_btn = gr.Button("Train Model", variant="primary")

                # Status and metrics display
                status_box = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=3,
                )
                metrics_box = gr.Textbox(
                    label="Training Metrics",
                    interactive=False,
                    lines=10,
                )

                # --- Upload callback ---
                def _upload_dataset(file_path):
                    if file_path is None:
                        return (
                            gr.update(),  # dropdown
                            "⚠️ No file selected.",  # status
                        )
                    try:
                        df = pd.read_csv(file_path)
                    except Exception as exc:
                        logger.error("Failed to read uploaded file: %s", exc)
                        return (
                            gr.update(),
                            f"❌ Failed to read CSV: {exc}",
                        )

                    # Validate
                    result = data_loader.validate_uploaded_dataset(df)
                    if not result.valid:
                        error_msg = "❌ Validation failed:\n" + "\n".join(f"  • {e}" for e in result.errors)
                        return (
                            gr.update(),
                            error_msg,
                        )

                    # Copy to datasets dir and register
                    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
                    dest_path = os.path.join(config.DATASETS_DIR, os.path.basename(file_path))
                    try:
                        os.makedirs(config.DATASETS_DIR, exist_ok=True)
                        shutil.copy2(file_path, dest_path)
                        dataset_registry.register_dataset(dataset_name, dest_path, df)
                    except ValueError as exc:
                        # Duplicate name
                        logger.warning("Dataset registration failed: %s", exc)
                        return (
                            gr.update(),
                            f"⚠️ {exc}",
                        )
                    except Exception as exc:
                        logger.error("Dataset registration error: %s", exc)
                        return (
                            gr.update(),
                            f"❌ Registration error: {exc}",
                        )

                    updated_choices = dataset_registry.list_datasets()
                    return (
                        gr.update(choices=updated_choices, value=dataset_name),
                        f"✅ Dataset '{dataset_name}' uploaded and registered ({len(df)} rows).",
                    )

                upload_btn.click(
                    fn=_upload_dataset,
                    inputs=[file_upload],
                    outputs=[dataset_dropdown, status_box],
                )

                # --- Train callback ---
                def _train_model(selected_dataset):
                    global _model, _scaler, _target_columns, _model_loaded

                    if not selected_dataset:
                        return "⚠️ Please select a dataset first.", ""

                    try:
                        ds_path = dataset_registry.get_dataset_path(selected_dataset)
                    except KeyError as exc:
                        return f"❌ {exc}", ""

                    model_prefix = selected_dataset
                    try:
                        status_msg = f"Training model on '{selected_dataset}'..."
                        logger.info(status_msg)
                        metrics = train_model.train_on_dataset(ds_path, model_prefix)
                    except Exception as exc:
                        logger.error("Training failed: %s", exc)
                        return f"❌ Training failed: {exc}", ""

                    # Hot-reload model artifacts
                    try:
                        _model, _scaler, _target_columns = model_io.load_model(model_prefix)
                        _model_loaded = True
                        logger.info("Hot-reloaded model from prefix '%s'.", model_prefix)
                    except Exception as exc:
                        logger.error("Failed to reload model after training: %s", exc)
                        return (
                            f"⚠️ Training succeeded but model reload failed: {exc}",
                            "",
                        )

                    # Format metrics
                    metrics_lines = [
                        f"Overall Metrics:",
                        f"  Train R²: {metrics['train_r2']:.4f}",
                        f"  Test  R²: {metrics['test_r2']:.4f}",
                        f"  Train MSE: {metrics['train_mse']:.4f}",
                        f"  Test  MSE: {metrics['test_mse']:.4f}",
                        "",
                        "Per-Parameter Metrics:",
                    ]
                    for p in metrics["per_param"]:
                        metrics_lines.append(
                            f"  {p['name']}: R²={p['r2']:.4f}  MSE={p['mse']:.4f}  MAE={p['mae']:.4f}"
                        )

                    return (
                        f"✅ Training complete for '{selected_dataset}'.",
                        "\n".join(metrics_lines),
                    )

                train_btn.click(
                    fn=_train_model,
                    inputs=[dataset_dropdown],
                    outputs=[status_box, metrics_box],
                )

    return demo


if __name__ == "__main__":
    logger.info("Starting Gradio app on %s:%d...", config.GRADIO_HOST, config.GRADIO_PORT)
    app = create_app()
    app.launch(
        server_name=config.GRADIO_HOST,
        server_port=config.GRADIO_PORT,
        share=False,
    )
