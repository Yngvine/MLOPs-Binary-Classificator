"""
Script to query MLflow for the best REGISTERED XGBoost model and serialize it to ONNX.
"""
from mlflow.tracking import MlflowClient
from mlflow import xgboost as mlflow_xgboost
import os
import shutil
import json
import numpy as np

# ONNX Libraries for XGBoost
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import xgboost as xgb
import re

# Monkeypatch XGBoost save_config to fix JSON format issue with onnxmltools
# XGBoost > 1.6 (and especially 2.x/3.x) saves base_score as a list in JSON dump,
# which causes onnxmltools to fail with "could not convert string to float".
if hasattr(xgb.Booster, 'save_config'):
    original_save_config = xgb.Booster.save_config
    def patched_save_config(self):
        config_str = original_save_config(self)
        # Fix base_score array format to scalar for onnxmltools compatibility
        # Pattern: "base_score":"[0.5]" -> "base_score":"0.5"
        new_config_str = re.sub(r'"base_score":"\[(.*?)\]"', r'"base_score":"\1"', config_str)
        return new_config_str
    xgb.Booster.save_config = patched_save_config

def serialize_best_model():
    # Setup Client and Model Name
    client = MlflowClient()
    model_name = "XGBoost_Best_Model" # Must match the name in train.py
    
    print(f"Searching for registered versions of model: '{model_name}'...")

    # Query Registered Models
    try:
        registered_models = client.search_model_versions(f"name='{model_name}'")
    except Exception as e:
        print(f"Error searching models: {e}")
        return

    if not registered_models:
        print(f"No registered models found for name '{model_name}'.")
        return

    print(f"Found {len(registered_models)} registered versions.")

    # 3. Compare models to select the best one
    best_run_id = None
    best_metric_val = -1.0
    best_version_num = None
    
    # The metric we optimized for in train.py
    target_metric = "val_auc_pr" 

    for model_version in registered_models:
        run_id = model_version.run_id
        version = model_version.version
        
        if not run_id:
            print(f"Skipping version {version}: No run_id found.")
            continue

        # Get run info to access metrics
        try:
            run = client.get_run(run_id)
            # Note: In the training script, we logged metrics to child runs, 
            # but the final model is logged to the Parent run. 
            # We usually log the best score to the parent run manually or 
            # check the metrics associated with the specific run ID where the model is stored.
            
            # Assuming the final model run has the metric logged (or we look at the optimization value)
            # If the metric isn't found, default to -1
            metric_val = run.data.metrics.get(target_metric, -1.0)
            
            # If metric is missing in the final model run, we might need to look up the best trial value
            # For this script, we assume the metric exists.
            
            print(f"Version {version} (Run {run_id}): {target_metric} = {metric_val:.4f}")
            
            if metric_val > best_metric_val:
                best_metric_val = metric_val
                best_run_id = run_id
                best_version_num = version
        except Exception as e:
            print(f"Could not fetch metrics for version {version}: {e}")

    if best_run_id is None:
        # Fallback: If metrics aren't comparable, pick the latest version
        print("Could not determine best model by metric. Selecting latest version.")
        best_run_id = registered_models[0].run_id
        best_version_num = registered_models[0].version

    if best_run_id is None:
        print("Error: No run_id found for the selected model version.")
        return

    print(f"\nBest Model: Version {best_version_num} (Run {best_run_id})")

    # Load the best model
    model_uri = f"runs:/{best_run_id}/model"
    print(f"Loading model from {model_uri}...")
    
    # LOAD XGBOOST MODEL
    xgb_model = mlflow_xgboost.load_model(model_uri)


    # Download Metadata (to get feature count for ONNX)
    print("Downloading metadata...")
    local_artifacts_path = "mlflow/model"
    os.makedirs(local_artifacts_path, exist_ok=True)
    
    n_features = 20 # Default fallback
    try:
        downloaded_path = client.download_artifacts(run_id=str(best_run_id), path="feature_metadata.json", dst_path=local_artifacts_path)
        with open(downloaded_path, 'r') as f:
            meta = json.load(f)
            n_features = meta.get("n_features", 20)
            print(f"Detected {n_features} features from metadata.")
    except Exception as e:
        print(f"Warning: Could not load feature metadata ({e}). Using default n_features={n_features}")

    # Serialize to ONNX
    output_path = os.path.join(local_artifacts_path, "xgboost_binary.onnx")
    print(f"Exporting model to {output_path}...")

    try:
        # Define input type: (Name, Type(Shape))
        # None in shape means dynamic batch size
        initial_type = [('float_input', FloatTensorType([None, n_features]))]
        
        # Convert
        onnx_model = onnxmltools.convert_xgboost(xgb_model, initial_types=initial_type)
        
        # Save
        onnxmltools.utils.save_model(onnx_model, output_path)
        print("Model serialized successfully.")
        
    except Exception as e:
        print(f"Error during ONNX export: {e}")

if __name__ == "__main__":
    serialize_best_model()