import math
import os

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import joblib

import config
from generate_dataset import patch_dimensions

matplotlib.use("Agg")

print("Loading model...")

# Prefer tuned model when available
if os.path.exists(config.TUNED_MODEL_FILE):
    model_file = config.TUNED_MODEL_FILE
    scaler_file = config.TUNED_SCALER_FILE
    target_file = config.TUNED_COLUMNS_FILE
    model_label = "tuned"
    print("Using tuned model.")
elif os.path.exists(config.MODEL_FILE):
    model_file = config.MODEL_FILE
    scaler_file = config.SCALER_FILE
    target_file = config.COLUMNS_FILE
    model_label = "base"
    print("Tuned model not found, using base model.")
else:
    print("ERROR: No model found! Run train_model.py or tune_hyperparameters.py first.")
    raise SystemExit(1)

model = joblib.load(model_file)
scaler = joblib.load(scaler_file)
target_columns = joblib.load(target_file)

print(f"Model loaded ({model_label}) – predicting {len(target_columns)} antenna parameters")

# ---------------------------------------------------------------------------
# Load reverse (constraint-based) model if available
# ---------------------------------------------------------------------------
if os.path.exists(config.REVERSE_MODEL_FILE):
    reverse_model = joblib.load(config.REVERSE_MODEL_FILE)
    reverse_scaler = joblib.load(config.REVERSE_SCALER_FILE)
    reverse_target_columns = joblib.load(config.REVERSE_TARGET_FILE)
    reverse_model_available = True
    print("Reverse model loaded.")
else:
    reverse_model = None
    reverse_scaler = None
    reverse_target_columns = []
    reverse_model_available = False
    print(
        "Reverse model not found – constraint-based design will use analytical "
        "equations only.  Run train_reverse_model.py to enable ML predictions."
    )


def predict_at_frequency(frequency: float) -> tuple[str, plt.Figure]:
    """Return a markdown table and a sweep plot for the given frequency.

    Args:
        frequency: Desired frequency in GHz.

    Returns:
        (markdown_table, matplotlib_figure)
    """
    try:
        if frequency < config.FREQ_MIN or frequency > config.FREQ_MAX:
            msg = (
                f"⚠️ Frequency {frequency:.3f} GHz is outside the training range "
                f"({config.FREQ_MIN}–{config.FREQ_MAX} GHz). "
                "Extrapolated predictions may be unreliable."
            )
            # Still proceed so users can see extrapolation behaviour
        else:
            msg = None

        # --- Point prediction ---
        freq_scaled = scaler.transform([[frequency]])
        predictions = model.predict(freq_scaled)[0]

        table = f"### Predictions for {frequency:.3f} GHz"
        if msg:
            table += f"\n\n{msg}"
        table += "\n\n| Parameter | Value |\n|-----------|-------|\n"
        for param_name, value in zip(target_columns, predictions):
            table += f"| {param_name} | {value:.4f} |\n"
        table += f"\n*Model: {model_label}*"

        # --- Sweep plot ---
        freqs = np.linspace(config.FREQ_MIN, config.FREQ_MAX, 300)
        freqs_scaled = scaler.transform(freqs.reshape(-1, 1))
        sweep = model.predict(freqs_scaled)  # shape (300, n_targets)

        n = len(target_columns)
        ncols = 3
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
        axes = np.array(axes).flatten()

        for i, col in enumerate(target_columns):
            ax = axes[i]
            ax.plot(freqs, sweep[:, i], lw=2, color="#1f77b4")
            ax.axvline(frequency, color="red", lw=1.5, linestyle="--", label=f"{frequency:.2f} GHz")
            ax.scatter([frequency], [predictions[i]], color="red", zorder=5, s=40)
            ax.set_xlabel("Frequency (GHz)", fontsize=8)
            ax.set_ylabel(col, fontsize=7)
            ax.set_title(col, fontsize=8)
            ax.grid(True, alpha=0.3)

        for i in range(n, len(axes)):
            axes[i].axis("off")

        fig.suptitle(
            f"Antenna Parameter Sweep – {config.FREQ_MIN}–{config.FREQ_MAX} GHz",
            fontsize=11,
            y=1.01,
        )
        plt.tight_layout()

        return table, fig

    except Exception as exc:
        return f"Error: {exc}", plt.figure()


# ---------------------------------------------------------------------------
# Constraint-based (reverse) prediction
# ---------------------------------------------------------------------------

def find_designs_for_constraints(
    target_freq: float,
    max_patch_length: float,
    max_patch_width: float,
) -> tuple[str, plt.Figure]:
    """Return antenna designs that satisfy the given size constraints.

    The function sweeps a grid of substrate heights and dielectric constants,
    computes the resulting patch dimensions using the classical microstrip
    patch equations, keeps only designs that fit within the user's size
    limits, and presents them ranked by patch area (most compact first).

    If the trained reverse ML model is available it also shows its single-shot
    prediction alongside the analytical results.

    Args:
        target_freq:      Desired resonant frequency in GHz.
        max_patch_length: Maximum allowable patch length in mm.
        max_patch_width:  Maximum allowable patch width in mm.

    Returns:
        (markdown_table, matplotlib_figure)
    """
    try:
        # --- Analytical sweep ---
        heights_mm = np.linspace(0.5, 5.0, 40)
        eps_values = np.linspace(2.0, 12.0, 40)
        valid_designs = []

        for h_mm in heights_mm:
            for eps in eps_values:
                result = patch_dimensions(target_freq * 1e9, eps, h_mm * 1e-3)
                if result is None:
                    continue
                W_m, L_m = result
                L_mm = L_m * 1e3
                W_mm = W_m * 1e3
                if L_mm <= max_patch_length and W_mm <= max_patch_width:
                    margin = 6.0 * h_mm
                    valid_designs.append(
                        {
                            "Substrate Height (mm)": round(h_mm, 2),
                            "Dielectric Constant": round(eps, 2),
                            "Patch Length (mm)": round(L_mm, 2),
                            "Patch Width (mm)": round(W_mm, 2),
                            "Substrate Length (mm)": round(L_mm + margin, 2),
                            "Substrate Width (mm)": round(W_mm + margin, 2),
                            "Patch Area (mm²)": round(L_mm * W_mm, 2),
                        }
                    )

        if not valid_designs:
            return (
                f"⚠️ No valid designs found for {target_freq:.3f} GHz with patch "
                f"≤ {max_patch_length:.1f} mm × {max_patch_width:.1f} mm.\n\n"
                "Try relaxing the size constraints or choosing a lower frequency.",
                plt.figure(),
            )

        # Sort by patch area – most compact first
        valid_designs.sort(key=lambda d: d["Patch Area (mm²)"])
        top = valid_designs[:10]

        table = (
            f"### Constraint-Based Designs for {target_freq:.3f} GHz\n\n"
            f"Constraints: patch ≤ **{max_patch_length:.1f} mm** (length) "
            f"× **{max_patch_width:.1f} mm** (width)\n\n"
            f"Found **{len(valid_designs)}** valid designs (top 10 by compactness shown):\n\n"
        )
        headers = list(top[0].keys())
        table += "| " + " | ".join(headers) + " |\n"
        table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for d in top:
            table += "| " + " | ".join(str(v) for v in d.values()) + " |\n"

        # --- ML reverse model prediction (if available) ---
        if reverse_model_available:
            inp = np.array([[target_freq, max_patch_length, max_patch_width]])
            inp_s = reverse_scaler.transform(inp)
            pred = reverse_model.predict(inp_s)[0]
            table += "\n#### ML Model Suggestion\n"
            table += "| Parameter | Value |\n|-----------|-------|\n"
            for col, val in zip(reverse_target_columns, pred):
                table += f"| {col} | {val:.3f} |\n"
            table += "\n*ML model prediction may differ from analytical sweep results.*"

        # --- Scatter plot: valid designs in (ε_r, h) space ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        all_eps = [d["Dielectric Constant"] for d in valid_designs]
        all_h = [d["Substrate Height (mm)"] for d in valid_designs]
        all_area = [d["Patch Area (mm²)"] for d in valid_designs]

        sc = axes[0].scatter(
            all_eps, all_h, c=all_area, cmap="viridis_r", alpha=0.7, s=20
        )
        plt.colorbar(sc, ax=axes[0], label="Patch Area (mm²)")
        axes[0].set_xlabel("Dielectric Constant (ε_r)")
        axes[0].set_ylabel("Substrate Height (mm)")
        axes[0].set_title(
            f"Valid Designs – {target_freq:.2f} GHz\n"
            f"patch ≤ {max_patch_length:.0f} × {max_patch_width:.0f} mm"
        )
        axes[0].grid(True, alpha=0.3)

        # Pareto-ish: patch length vs patch width
        all_L = [d["Patch Length (mm)"] for d in valid_designs]
        all_W = [d["Patch Width (mm)"] for d in valid_designs]
        axes[1].scatter(all_L, all_W, c=all_area, cmap="viridis_r", alpha=0.7, s=20)
        axes[1].axvline(max_patch_length, color="red", lw=1.5, linestyle="--", label="Max length")
        axes[1].axhline(max_patch_width, color="orange", lw=1.5, linestyle="--", label="Max width")
        if top:
            axes[1].scatter(
                [top[0]["Patch Length (mm)"]],
                [top[0]["Patch Width (mm)"]],
                color="red",
                marker="*",
                s=200,
                zorder=5,
                label="Most compact",
            )
        axes[1].set_xlabel("Patch Length (mm)")
        axes[1].set_ylabel("Patch Width (mm)")
        axes[1].set_title("Patch Dimensions of Valid Designs")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return table, fig

    except Exception as exc:
        return f"Error: {exc}", plt.figure()


# Build Gradio interface
with gr.Blocks(title="Antenna Predictor") as demo:
    gr.Markdown(
        """
        # 🛜 Antenna Parameter Predictor
        Two modes are available:

        * **Forward Prediction** – enter a frequency and get all antenna physical parameters.
        * **Constraint-Based Design** – enter a target frequency *and* size constraints to find
          valid antenna designs (patch dimensions, substrate parameters) that fit within your limits.
        """
    )

    with gr.Tabs():
        # ------------------------------------------------------------------ #
        # Tab 1 – Forward prediction (original behaviour)
        # ------------------------------------------------------------------ #
        with gr.TabItem("📡 Forward Prediction"):
            gr.Markdown(
                "Predict antenna physical parameters from a target operating frequency "
                "using a Random Forest model trained on WiFi 7 simulation data "
                "(2–7 GHz, three patch configurations)."
            )

            with gr.Row():
                freq_slider = gr.Slider(
                    minimum=config.FREQ_MIN,
                    maximum=config.FREQ_MAX,
                    value=2.4,
                    step=0.005,
                    label="Frequency (GHz)",
                    info="WiFi 7 operates at 2.4, 5, and 6 GHz bands",
                )
                freq_number = gr.Number(
                    value=2.4,
                    minimum=config.FREQ_MIN,
                    maximum=config.FREQ_MAX,
                    label="Frequency (GHz) – type exact value",
                )

            with gr.Row():
                predict_btn = gr.Button("Predict", variant="primary")

            with gr.Row():
                table_out = gr.Markdown(label="Predicted Values")
                plot_out = gr.Plot(label="Parameter vs Frequency Sweep")

            # Keep slider and number input in sync.
            freq_slider.change(fn=lambda v: v, inputs=freq_slider, outputs=freq_number)
            freq_number.change(fn=lambda v: v, inputs=freq_number, outputs=freq_slider)

            predict_btn.click(
                fn=predict_at_frequency,
                inputs=freq_slider,
                outputs=[table_out, plot_out],
            )

            # Auto-predict on slider release
            freq_slider.release(
                fn=predict_at_frequency,
                inputs=freq_slider,
                outputs=[table_out, plot_out],
            )

            gr.Examples(
                examples=[[2.4], [5.0], [5.5], [6.0], [6.5]],
                inputs=freq_slider,
                outputs=[table_out, plot_out],
                fn=predict_at_frequency,
                cache_examples=False,
            )

            gr.Markdown(
                f"*Training range: {config.FREQ_MIN}–{config.FREQ_MAX} GHz · "
                f"Model: {model_label} Random Forest*"
            )

        # ------------------------------------------------------------------ #
        # Tab 2 – Constraint-based (reverse) design
        # ------------------------------------------------------------------ #
        with gr.TabItem("🔧 Constraint-Based Design"):
            gr.Markdown(
                """
                ### Find an antenna design that meets your requirements

                Enter your **target resonant frequency** and the **maximum patch dimensions**
                you can accommodate.  The tool sweeps the substrate parameter space and
                returns all antenna designs that satisfy your constraints, ranked by
                compactness (smallest patch area first).

                *Example:* for 5 GHz and a board where the patch must fit inside 50 mm × 50 mm,
                enter Frequency = 5, Max Patch Length = 50, Max Patch Width = 50.
                """
            )

            with gr.Row():
                cb_freq = gr.Slider(
                    minimum=config.REVERSE_FREQ_MIN,
                    maximum=config.REVERSE_FREQ_MAX,
                    value=5.0,
                    step=0.1,
                    label="Target Frequency (GHz)",
                    info="Desired resonant frequency of the antenna",
                )

            with gr.Row():
                cb_max_length = gr.Slider(
                    minimum=5.0,
                    maximum=150.0,
                    value=50.0,
                    step=1.0,
                    label="Max Patch Length (mm)",
                    info="Maximum allowable patch length",
                )
                cb_max_width = gr.Slider(
                    minimum=5.0,
                    maximum=150.0,
                    value=50.0,
                    step=1.0,
                    label="Max Patch Width (mm)",
                    info="Maximum allowable patch width",
                )

            with gr.Row():
                cb_btn = gr.Button("Find Designs", variant="primary")

            with gr.Row():
                cb_table_out = gr.Markdown(label="Valid Designs")
                cb_plot_out = gr.Plot(label="Design Space")

            cb_btn.click(
                fn=find_designs_for_constraints,
                inputs=[cb_freq, cb_max_length, cb_max_width],
                outputs=[cb_table_out, cb_plot_out],
            )

            gr.Examples(
                examples=[
                    [5.0, 50.0, 50.0],
                    [2.4, 80.0, 80.0],
                    [6.0, 30.0, 30.0],
                    [2.4, 40.0, 40.0],
                    [5.0, 25.0, 25.0],
                ],
                inputs=[cb_freq, cb_max_length, cb_max_width],
                outputs=[cb_table_out, cb_plot_out],
                fn=find_designs_for_constraints,
                cache_examples=False,
            )

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Starting Gradio app...")
    print("Access at: http://localhost:7860")
    print("=" * 60 + "\n")

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
