import math
import os

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import joblib

import config

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


# Build Gradio interface
with gr.Blocks(title="WiFi 7 Antenna Predictor") as demo:
    gr.Markdown(
        """
        # 🛜 WiFi 7 Antenna Parameter Predictor
        Predict antenna physical design parameters from operating frequency using a
        Random Forest regression model trained on WiFi 7 simulation data.
        """
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
    # Gradio suppresses re-entrant events so these bindings don't loop.
    freq_slider.change(fn=lambda v: v, inputs=freq_slider, outputs=freq_number)
    freq_number.change(fn=lambda v: v, inputs=freq_number, outputs=freq_slider)

    predict_btn.click(
        fn=predict_at_frequency,
        inputs=freq_slider,
        outputs=[table_out, plot_out],
    )

    # Auto-predict on slider move
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

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Starting Gradio app...")
    print("Access at: http://localhost:7860")
    print("=" * 60 + "\n")

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
