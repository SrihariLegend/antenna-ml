"""Gradio web interface for antenna design prediction and CST generation."""

import logging
import os
import shutil
import sys

import gradio as gr
import numpy as np
import pandas as pd

import antenna_3d
import balanis
import config
import data_loader
import dataset_registry
import model_io
import train_model
import tune_hyperparameters

logger = logging.getLogger(__name__)

_model = None
_scaler = None
_target_columns = None
_model_loaded = False
_geometry_grid = None
# Last prediction result, used by CST launch button
_last_design = None


def _load_geometry_grid():
    global _geometry_grid
    try:
        df = pd.read_csv(config.DATASET_PATH)
        _geometry_grid = df[["patch_length", "substrate_height"]].drop_duplicates().values
        logger.info("Loaded geometry grid: %d candidates.", len(_geometry_grid))
    except Exception as exc:
        logger.warning("Could not load geometry grid: %s", exc)
        _geometry_grid = None


def predict_best_design(frequency, max_patch_length, max_substrate_height):
    global _last_design
    _last_design = None

    if not _model_loaded or _geometry_grid is None:
        return "⚠️ No trained model. Go to **Dataset Management** tab to train.", gr.update(interactive=False), None

    if frequency < config.FREQ_MIN or frequency > config.FREQ_MAX:
        return f"⚠️ Frequency out of range ({config.FREQ_MIN}–{config.FREQ_MAX} GHz)", gr.update(interactive=False), None

    candidates = _geometry_grid.copy()
    if max_patch_length > 0:
        candidates = candidates[candidates[:, 0] <= max_patch_length]
    if max_substrate_height > 0:
        candidates = candidates[candidates[:, 1] <= max_substrate_height]

    if len(candidates) == 0:
        return "⚠️ No geometries satisfy constraints. Try relaxing them.", gr.update(interactive=False), None

    try:
        X = np.column_stack([np.full(len(candidates), frequency), candidates])
        s11_pred = _model.predict(_scaler.transform(X))
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        return f"Error: {exc}", gr.update(interactive=False), None

    best_idx = s11_pred.argmin()
    best_pl = candidates[best_idx, 0]
    best_sh = candidates[best_idx, 1]
    best_s11 = s11_pred[best_idx]

    dims = balanis.full_dimensions(frequency, best_pl, best_sh)
    _last_design = dims

    output = f"### Best Design for {frequency:.3f} GHz\n\n"
    output += "| Parameter | Value |\n|---|---|\n"
    output += f"| Predicted S11 | {best_s11:.2f} dB |\n"
    output += f"| Patch length | {dims['patch_length_mm']:.2f} mm |\n"
    output += f"| Patch width | {dims['patch_width_mm']:.2f} mm |\n"
    output += f"| Substrate height | {dims['substrate_height_mm']:.2f} mm |\n"
    output += f"| Substrate length | {dims['substrate_length_mm']:.2f} mm |\n"
    output += f"| Substrate width | {dims['substrate_width_mm']:.2f} mm |\n"

    ranked = s11_pred.argsort()
    if len(ranked) > 1:
        output += "\n### Top Alternatives\n\n"
        output += "| Rank | Patch Length | Substrate Height | S11 |\n|---|---|---|---|\n"
        for rank, idx in enumerate(ranked[:5], 1):
            output += f"| {rank} | {candidates[idx, 0]:.2f} mm | {candidates[idx, 1]:.2f} mm | {s11_pred[idx]:.2f} dB |\n"

    try:
        fig_3d = antenna_3d.render("rectangular", dims)
    except Exception:
        fig_3d = None

    return output, gr.update(interactive=True), fig_3d


def launch_cst(output_path):
    """Launch CST Linker with the last predicted design."""
    if _last_design is None:
        return "⚠️ Run a prediction first."

    if not output_path or not output_path.strip():
        output_path = "."

    # Locate the CST Linker script relative to this file or bundled path
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    linker_script = os.path.join(base, "cst_linker", "example_script.py")
    if not os.path.exists(linker_script):
        # Try sibling directory layout (non-bundled)
        linker_script = os.path.join(os.path.dirname(base), "CST_Linker", "example_script.py")
    if not os.path.exists(linker_script):
        return f"❌ CST Linker script not found. Expected at:\n{linker_script}"

    d = _last_design

    # Import and call directly
    try:
        sys.path.insert(0, os.path.dirname(linker_script))
        from example_script import create_patch_antenna_external
        create_patch_antenna_external(
            patch_length_mm=d["patch_length_mm"],
            patch_width_mm=d["patch_width_mm"],
            substrate_thickness_mm=d["substrate_height_mm"],
            output_path=output_path,
        )
        return f"✅ CST project generated at: {os.path.abspath(output_path)}/example_patch_antenna.cstprj"
    except ImportError:
        return (
            "❌ CST Studio Suite Python API not installed on this machine.\n"
            "Install it on a machine with CST, then run:\n\n"
            f"python example_script.py with:\n"
            f"  patch_length = {d['patch_length_mm']:.2f}\n"
            f"  patch_width  = {d['patch_width_mm']:.2f}\n"
            f"  substrate_h  = {d['substrate_height_mm']:.2f}"
        )
    except Exception as exc:
        return f"❌ CST generation failed: {exc}"


def create_app() -> gr.Blocks:
    global _model, _scaler, _target_columns, _model_loaded

    dataset_registry.init_default_dataset()
    _load_geometry_grid()

    try:
        prefix = model_io.find_best_model()
        _model, _scaler, _target_columns = model_io.load_model(prefix)
        _model_loaded = True
        logger.info("Using %s model.", prefix)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("No model available at startup: %s", exc)
        _model_loaded = False

    with gr.Blocks(title="📡 Antenna Design Predictor") as demo:
        gr.Markdown(
            "# 📡 Antenna Design Predictor\n\n"
            "Predict optimal rectangular patch antenna geometry for a target frequency, "
            "then generate a CST Studio project with one click."
        )

        with gr.Tabs():
            with gr.Tab("Prediction"):
                with gr.Row():
                    with gr.Column(scale=2):
                        freq_slider = gr.Slider(
                            minimum=config.FREQ_MIN,
                            maximum=config.FREQ_MAX,
                            value=2.4,
                            step=0.001,
                            label="Target Frequency (GHz)",
                        )
                        with gr.Row():
                            max_pl = gr.Number(value=0, label="Max Patch Length (mm)", info="0 = no constraint")
                            max_sh = gr.Number(value=0, label="Max Substrate Height (mm)", info="0 = no constraint")

                        predict_btn = gr.Button("🔍 Find Best Design", variant="primary", size="lg")
                        output_md = gr.Markdown(label="Results")

                    with gr.Column(scale=1):
                        gr.Markdown("### 3D Preview")
                        plot_3d = gr.Plot(label="Antenna Geometry")
                        gr.Markdown("### Build in CST")
                        cst_output_path = gr.Textbox(
                            value=".",
                            label="CST Project Output Path",
                            info="Directory where .cstprj file will be saved",
                        )
                        cst_btn = gr.Button("🚀 Launch CST Linker", variant="secondary", size="lg", interactive=False)
                        cst_status = gr.Textbox(label="CST Status", interactive=False, lines=5)

                predict_btn.click(
                    fn=predict_best_design,
                    inputs=[freq_slider, max_pl, max_sh],
                    outputs=[output_md, cst_btn, plot_3d],
                )
                cst_btn.click(
                    fn=launch_cst,
                    inputs=[cst_output_path],
                    outputs=[cst_status],
                )

            with gr.Tab("Dataset Management"):
                gr.Markdown("### Dataset Management\nUpload new datasets, select a dataset, and train models.")

                dataset_dropdown = gr.Dropdown(
                    choices=dataset_registry.list_datasets(),
                    label="Select Dataset",
                )

                with gr.Row():
                    file_upload = gr.File(label="Upload CSV Dataset", file_types=[".csv"], type="filepath")
                    upload_btn = gr.Button("Upload", variant="secondary")

                with gr.Row():
                    train_btn = gr.Button("Train Model", variant="primary")
                    tune_btn = gr.Button("Tune Hyperparameters", variant="secondary")

                status_box = gr.Textbox(label="Status", interactive=False, lines=3)
                metrics_box = gr.Textbox(label="Training Metrics", interactive=False, lines=10)

                def _upload_dataset(file_path):
                    if file_path is None:
                        return gr.update(), "⚠️ No file selected."
                    try:
                        df = pd.read_csv(file_path)
                    except Exception as exc:
                        return gr.update(), f"❌ Failed to read CSV: {exc}"
                    if config.FREQUENCY_COL not in df.columns:
                        return gr.update(), f"❌ Missing required column '{config.FREQUENCY_COL}'."

                    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
                    dest_path = os.path.join(config.DATASETS_DIR, os.path.basename(file_path))
                    try:
                        os.makedirs(config.DATASETS_DIR, exist_ok=True)
                        shutil.copy2(file_path, dest_path)
                        dataset_registry.register_dataset(dataset_name, dest_path, df)
                    except ValueError as exc:
                        return gr.update(), f"⚠️ {exc}"
                    except Exception as exc:
                        return gr.update(), f"❌ Registration error: {exc}"

                    return (
                        gr.update(choices=dataset_registry.list_datasets(), value=dataset_name),
                        f"✅ Dataset '{dataset_name}' uploaded ({len(df)} rows).",
                    )

                upload_btn.click(fn=_upload_dataset, inputs=[file_upload], outputs=[dataset_dropdown, status_box])

                def _train_model(selected_dataset):
                    global _model, _scaler, _target_columns, _model_loaded
                    if not selected_dataset:
                        return "⚠️ Please select a dataset first.", ""
                    try:
                        ds_path = dataset_registry.get_dataset_path(selected_dataset)
                    except KeyError as exc:
                        return f"❌ {exc}", ""
                    try:
                        metrics = train_model.train_on_dataset(ds_path, selected_dataset)
                    except Exception as exc:
                        return f"❌ Training failed: {exc}", ""
                    try:
                        _model, _scaler, _target_columns = model_io.load_model(selected_dataset)
                        _model_loaded = True
                        _load_geometry_grid()
                    except Exception as exc:
                        return f"⚠️ Training OK but reload failed: {exc}", ""

                    lines = [
                        f"Train R²: {metrics['train_r2']:.4f}  |  Test R²: {metrics['test_r2']:.4f}",
                        f"Train MSE: {metrics['train_mse']:.4f}  |  Test MSE: {metrics['test_mse']:.4f}",
                        "",
                    ]
                    for p in metrics["per_param"]:
                        lines.append(f"  {p['name']}: R²={p['r2']:.4f}  MSE={p['mse']:.4f}  MAE={p['mae']:.4f}")
                    return f"✅ Training complete for '{selected_dataset}'.", "\n".join(lines)

                train_btn.click(fn=_train_model, inputs=[dataset_dropdown], outputs=[status_box, metrics_box])

                def _tune_model(selected_dataset):
                    global _model, _scaler, _target_columns, _model_loaded
                    if not selected_dataset:
                        return "⚠️ Please select a dataset first.", ""
                    try:
                        ds_path = dataset_registry.get_dataset_path(selected_dataset)
                    except KeyError as exc:
                        return f"❌ {exc}", ""
                    try:
                        result = tune_hyperparameters.tune_on_dataset(ds_path, selected_dataset)
                    except Exception as exc:
                        return f"❌ Tuning failed: {exc}", ""

                    tuned_prefix = result["tuned_prefix"]
                    try:
                        _model, _scaler, _target_columns = model_io.load_model(tuned_prefix)
                        _model_loaded = True
                    except Exception as exc:
                        return f"⚠️ Tuning OK but reload failed: {exc}", ""

                    elapsed = result["elapsed_seconds"]
                    lines = [
                        f"Completed in {elapsed / 60:.1f} min",
                        f"Test R²: {result['test_r2']:.4f}  |  MSE: {result['test_mse']:.4f}",
                        "", "Best Parameters:",
                    ]
                    for k, v in result["best_params"].items():
                        lines.append(f"  {k}: {v}")
                    return f"✅ Tuning complete (saved as '{tuned_prefix}').", "\n".join(lines)

                tune_btn.click(fn=_tune_model, inputs=[dataset_dropdown], outputs=[status_box, metrics_box])

    return demo


if __name__ == "__main__":
    import socket

    def _find_free_port(start=7860, end=7880):
        for port in range(start, end):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    return port
        return start

    config.setup_logging()
    app = create_app()
    port = _find_free_port()
    logger.info("Starting Antenna Design Predictor on port %d...", port)

    # Try native desktop window, fall back to browser
    use_native = False
    try:
        import webview
        from webview.guilib import initialize
        initialize()
        use_native = True
    except Exception:
        use_native = False

    if use_native:
        import threading, time

        def _start_gradio():
            app.launch(
                server_name="127.0.0.1",
                server_port=port,
                share=False,
                prevent_thread_lock=True,
            )

        threading.Thread(target=_start_gradio, daemon=True).start()
        time.sleep(2)
        webview.create_window(
            "Antenna Design Predictor",
            f"http://127.0.0.1:{port}",
            width=1200, height=800,
        )
        webview.start()
    else:
        logger.info("Native window not available — opening in browser")
        app.launch(
            server_name=config.GRADIO_HOST,
            server_port=port,
            share=False,
        )
