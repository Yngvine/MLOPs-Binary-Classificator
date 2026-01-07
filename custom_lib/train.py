import os
import sys
from pathlib import Path

# Add the project root to sys.path so imports work correctly
sys.path.append(str(Path(__file__).parent.parent / "mlflow"))

import mlflow
from mlflow.tracking import MlflowClient
# Note: Assuming your local folder is named 'mlflow' and contains training.py
# If this conflicts with the library 'mlflow', you might need to rename the folder to 'ml_modules'
from training import run_optimization, EXPERIMENT_NAME #type: ignore
from serialization import serialize_best_model #type: ignore
import json

def get_best_metric_from_mlflow(experiment_name: str, metric_name: str = "val_f1") -> float:
    """
    Queries MLflow to find the best run in the experiment and returns the specific metric.
    """
    print(f"Querying MLflow for best {metric_name} in experiment '{experiment_name}'...")
    
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        print(f"Experiment '{experiment_name}' not found.")
        return 0.0

    # Search for the run with the highest value for the given metric
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1,
        order_by=[f"metrics.{metric_name} DESC"] # Get the highest accuracy
    )

    if not runs:
        print("No runs found")
        return 0.0

    best_run = runs[0]
    best_value = best_run.data.metrics.get(metric_name, 0.0)
    print(f"Found best run ({best_run.info.run_id}) with {metric_name}: {best_value}")
    
    return float(best_value)


def main():
    print("=== Step 1: Starting Training & Optimization ===")
    # This runs the Optuna search and registers the best model in MLflow
    run_optimization()
    
    print("\n=== Step 2: Serializing Best Model to ONNX ===")
    # This converts the registered XGBoost model to ONNX
    serialize_best_model()

    print("\n=== Step 3: Saving Metrics for API ===")
    # Retrieve the metric from MLflow instead of modifying training.py
    # Ensure 'accuracy' matches exactly what you log in training.py (e.g. 'accuracy', 'f1_score')
    best_model_f1 = get_best_metric_from_mlflow(EXPERIMENT_NAME, metric_name="val_f1")
    best_model_auc_pr = get_best_metric_from_mlflow(EXPERIMENT_NAME, metric_name="val_auc_pr")
    best_model_mean_recall = get_best_metric_from_mlflow(EXPERIMENT_NAME, metric_name="val_mean_recall")
    
    # Save to a file that the API will read
    metrics_data = {"f1_score": best_model_f1, "auc_pr": best_model_auc_pr, "mean_recall": best_model_mean_recall}
    with open("metrics.json", "w") as f:
        json.dump(metrics_data, f)
    
    print(f"Saved metrics to metrics.json: {metrics_data}") 

if __name__ == "__main__":
    main()