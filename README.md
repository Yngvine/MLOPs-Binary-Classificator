# MLOps Binary Classificator - Rice Grain Classification

This project is a complete MLOps implementation for a binary classification problem (classifying rice grains as either "Gonen" or "Jasmine"). It demonstrates a full lifecycle including automated training, model serialization (ONNX), containerization, continuous deployment (CD), and production monitoring.

## 📂 Project Structure

```
MLOPs-Binary-Classificator/
├── api/                 # FastAPI application
│   └── api_main.py      # Main entry point, exposes /classify and /metrics
├── custom_lib/          # Core logic and orchestration
│   ├── data.py          # Data loading and visualization
│   ├── model.py         # ONNX model inference logic
│   └── train.py         # Orchestrator for training, serialization, and metrics
├── data/                # Dataset storage
├── graf/                # Grafana configuration (Docker, Datasource, Dashboard)
├── ml_modules/          # MLflow training & serialization scripts
├── prom/                # Prometheus configuration
├── templates/           # HTML templates for the API UI
├── tests/               # Unit tests
├── .github/workflows/   # CI/CD Pipeline definition
├── Dockerfile           # Dockerfile for the API
└── pyproject.toml       # Python dependencies (managed by uv)
```

## 🚀 Features

*   **Model:** XGBoost classifier optimized with Optuna and serialized to ONNX.
*   **API:** FastAPI service providing real-time predictions and a web UI.
*   **CI/CD:** GitHub Actions pipeline for automated testing, training, and deployment to Render.
*   **Monitoring:** Full stack monitoring with Prometheus (metrics collection) and Grafana (visualization).
*   **Metric Propagation:** Automatically propagates training metrics (F1 Score) from the training pipeline to the production monitoring dashboard.

## 🛠️ Local Setup

1.  **Install Dependencies:**
    This project uses `uv` for dependency management.
    ```bash
    uv sync
    ```

2.  **Run API Locally:**
    ```bash
    uv run uvicorn api.api_main:app --reload
    ```
    Access the UI at `http://localhost:8000`.

3.  **Run Tests:**
    ```bash
    uv run python -m pytest tests/
    ```

## 🐳 Docker Usage

To build and run the API container locally:

```bash
# 1. Create a dummy metrics file (required for build if training hasn't run)
echo {"model_accuracy": 0.0} > metrics.json

# 2. Build the image
docker build -t mlops-bc .

# 3. Run the container
docker run -p 8000:8000 mlops-bc
```

## 🔄 CI/CD Pipeline

The project uses a **Conditional Training** pipeline defined in `.github/workflows/cd.yml`.

### Workflow Logic
1.  **Standard Push:** If you push code *without* the tag, the pipeline skips training. It builds the Docker image using the *existing* model and metrics in the repo.
2.  **Training Push:** If your commit message contains **`#TRAIN`**, the pipeline:
    *   Runs `custom_lib/train.py`.
    *   Optimizes hyperparameters (Optuna) and logs to MLflow.
    *   Serializes the best model to ONNX.
    *   Saves the best metric (e.g., `val_f1`) to `metrics.json`.
    *   Builds the Docker image *including* the new model and `metrics.json`.
    *   Deploys the new image to Render.

**Example Commit:**
```bash
git commit -m "Updated hyperparameters and retrained model #TRAIN"
```

## 📊 Monitoring Stack

The monitoring architecture consists of three services running on Render:

1.  **API Service (`mlops-bc`):** Exposes a `/metrics` endpoint containing:
    *   `rice_predictions_total`: Counter for predicted classes.
    *   `model_accuracy`: Gauge showing the F1 score of the deployed model (read from `metrics.json`).
2.  **Prometheus (`mlops-prometheus`):** Scrapes the API every 15 seconds.
3.  **Grafana (`mlops-grafana`):** Visualizes the data with a pre-provisioned dashboard.

### How to Deploy/Redeploy Monitoring

To avoid data loss and redundant builds, Prometheus and Grafana are **not** deployed in the CD pipeline. You deploy them manually once (or when configs change).

#### 1. Deploy Prometheus
Ensure `prom/prometheus.yml` points to your actual Render API URL.

```bash
docker build -t <your-user>/mlops-prometheus:latest -f prom/Dockerfile.prometheus .
docker push <your-user>/mlops-prometheus:latest
# Deploy this image on Render (Port 9090)
```

#### 2. Deploy Grafana
The Grafana image comes pre-configured with the Prometheus datasource and a Dashboard.

```bash
docker build -t <your-user>/mlops-grafana:latest -f graf/Dockerfile.grafana graf/
docker push <your-user>/mlops-grafana:latest
# Deploy this image on Render (Port 3000)
```

## 🔗 API Endpoints

*   `GET /`: Web Interface for manual testing.
*   `POST /classify/`: JSON endpoint for predictions.
    *   Input: Rice features (Area, Perimeter, etc.).
    *   Output: `{"predicted_class": "Jasmine"}`.
*   `GET /metrics`: Prometheus metrics endpoint.