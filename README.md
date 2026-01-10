---
title: MLOPs-BC
emoji: 🍚
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.0.2
python_version: "3.13"
app_file: app.py
pinned: false
license: mit
short_description: Demo for mlops practice
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# 🍚 Rice Classification UI

A modern Gradio frontend for the MLOps Binary Classification project. This application provides an interactive interface to explore the capabilities of the Rice Grain Classification system.

## ✨ Features

-   **Interactive Inputs**: Sliders for 10 morphological features (Area, Perimeter, Axis Lengths, etc.).
-   **Model Switching**: Toggle between **XGBoost** and **TabNet** models in real-time.
-   **Dynamic Visualization**:
    -   **P5.js Integration**: Generates a real-time 2D visual representation of the rice grain based on your input parameters.
    -   **Audio Feedback**: Ambient background music enhances the user experience.
-   **Real-time Inference**: Sends requests to the deployed FastAPI backend on Render.

## 🎮 How to Use

The interface offers two modes designed for different user needs:

### 1. Simple Mode (Default)
Ideal for quick demonstrations.
- **Input**: You only control the *independent* physical properties (Rotation, Major Axis, Minor Axis).
- **Automation**: The system automatically calculates complex derived features (like Roundness or Eccentricity) based on your inputs.
- **Visualizer**: See the grain shape evolve naturally as you adjust the core dimensions.

### 2. Advanced Mode
For data scientists and deeper exploration.
- **Full Control**: Unlocks manual sliders for all 10 feature inputs, including mathematical ratios.
- **Edge Cases**: Allows you to test unusual or physically impossible combinations (e.g., high area with low perimeter) to see how the model behaves.

### Steps
1.  **Select Mode**: Toggle the "Advanced Mode" checkbox to reveal all features.
2.  **Configure Parameters**: Use the sliders to adjust the morphological properties.
3.  **Select Model**: Choose between `XGBoost` or `TabNet` using the dropdown.
4.  **Classify**: The system predicts whether the grain is **Gonen** or **Jasmine**.

## 🔧 How it Works

This interface connects to a separate AI backend running on cloud infrastructure. When you change settings, this app:
1.  Collects your 10 measurement points.
2.  Sends them securely to our Inference API.
3.  Displays the returned prediction and confidence.

*Note: The visualizer uses P5.js to approximate what a rice grain with your specific measurements might look like!*

## 🔗 Links

-   **Live Demo**: [Hugging Face Space](https://huggingface.co/spaces/Yngvine/MLOPs-BC)
-   **Source Code**: [GitHub Repository](https://github.com/Yngvine/MLOPs-Binary-Classificator)

---

### 👨‍💻 For Developers

#### Local Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set Backend URL (Optional):**
    The app defaults to the production endpoint (`https://mlops-bc-latest.onrender.com`). To target a local API:
    ```bash
    # PowerShell
    $env:API_URL="http://localhost:8000"
    ```

3.  **Run the App:**
    ```bash
    python app.py
    ```

