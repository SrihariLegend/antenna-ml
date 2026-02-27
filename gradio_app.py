import gradio as gr
import numpy as np
import joblib
import os

print("Loading model...")

# Check if tuned model exists, fall back to base model
if os.path.exists('rf_antenna_model_tuned.pkl'):
    model_file = 'rf_antenna_model_tuned.pkl'
    scaler_file = 'scaler_X_tuned.pkl'
    target_file = 'target_columns_tuned.pkl'
    print("Using tuned model.")
elif os.path.exists('rf_antenna_model.pkl'):
    model_file = 'rf_antenna_model.pkl'
    scaler_file = 'scaler_X.pkl'
    target_file = 'target_columns.pkl'
    print("Tuned model not found, using base model.")
else:
    print("ERROR: No model found! Run train_model.py or tune_hyperparameters.py first.")
    exit(1)

# Load model, scaler, and target columns
model = joblib.load(model_file)
scaler = joblib.load(scaler_file)
target_columns = joblib.load(target_file)

print(f"Model loaded successfully!")
print(f"Predicting {len(target_columns)} antenna parameters")

def predict_antenna_parameters(frequency):
    """
    Predict all antenna parameters from frequency
    
    Args:
        frequency: Input frequency in GHz
    
    Returns:
        Dictionary of predicted parameters
    """
    try:
        # Validate input
        if frequency < 2.0 or frequency > 7.0:
            return "⚠️ Frequency out of training range (2-7 GHz)"
        
        # Scale input
        freq_scaled = scaler.transform([[frequency]])
        
        # Predict
        predictions = model.predict(freq_scaled)[0]
        
        # Format output as markdown table
        result = f"### Predictions for {frequency} GHz\n\n"
        result += "| Parameter | Value |\n"
        result += "|-----------|-------|\n"
        
        for i, param_name in enumerate(target_columns):
            result += f"| {param_name} | {predictions[i]:.4f} |\n"
        
        return result
        
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
demo = gr.Interface(
    fn=predict_antenna_parameters,
    inputs=gr.Slider(
        minimum=2.0,
        maximum=7.0,
        value=2.4,
        step=0.01,
        label="Frequency (GHz)",
        info="WiFi 7 operates at 2.4, 5, and 6 GHz bands"
    ),
    outputs=gr.Markdown(label="Predicted Antenna Parameters"),
    title="🛜 WiFi 7 Antenna Parameter Predictor",
    description="""
    **Predict antenna parameters from operating frequency**
    
    This model uses Random Forest regression trained on WiFi 7 antenna simulation data.
    Slide the frequency to see how antenna characteristics change across the spectrum.
    """,
    examples=[
        [2.4],   # WiFi 2.4 GHz
        [5.0],   # WiFi 5 GHz  
        [5.5],
        [6.0],   # WiFi 6E
        [6.5],
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Gradio app...")
    print("Access at: http://localhost:7860")
    print("="*60 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
