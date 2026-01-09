"""
Script to query MLflow for the best REGISTERED XGBoost model and serialize it to ONNX.
"""
from mlflow.tracking import MlflowClient
from mlflow import xgboost as mlflow_xgboost
from mlflow import sklearn as mlflow_sklearn
import os
import shutil
import json
import numpy as np
import sys

# ONNX Libraries for XGBoost
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import xgboost as xgb
import re

# ONNX Libraries for TabNet (Method specific)
import torch

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

class TabNetOnnxWrapper(torch.nn.Module):
    """Wrapper to make TabNet output labels and probabilities like XGBoost."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, x):
        # TabNet network returns (logits, M_loss)
        logits, _ = self.model(x)
        probs = self.softmax(logits)
        preds = torch.argmax(probs, dim=1)
        return preds, probs

def serialize_best_model(model_type="xgboost"):
    # Setup Client and Model Name
    client = MlflowClient()
    
    # Needs to match constants in training.py
    if model_type.lower() == "xgboost":
        model_name = "Rice_XGBoost_Best"
    elif model_type.lower() == "tabnet":
        model_name = "Rice_TabNet_Best"
    else:
        print(f"Unknown model type: {model_type}")
        return
    
    print(f"Searching for best registered model: '{model_name}' (Type: {model_type})...")

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

    print(f"\nBest Model: Version {best_version_num} (Run {best_run_id})")

    # Load the best model
    model_uri = f"runs:/{best_run_id}/model"
    print(f"Loading model from {model_uri}...")
    
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

    # Output path
    # We overwrite the same file so the API picks it up automatically, regardless of the model typ
    
    if model_type.lower() == "xgboost":
        output_path = os.path.join(local_artifacts_path, "xgboost_binary.onnx")
        # LOAD XGBOOST MODEL
        xgb_model = mlflow_xgboost.load_model(model_uri)

        # Serialize to ONNX
        print(f"Exporting XGBoost model to {output_path}...")

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
            
    elif model_type.lower() == "tabnet":
        output_path = os.path.join(local_artifacts_path, "tabnet_binary.onnx")
        # LOAD TABNET MODEL
        # It was saved as sklearn flavor
        print("Loading TabNet model...")
        tabnet_clf = mlflow_sklearn.load_model(model_uri)
        
        print(f"Exporting TabNet model to {output_path}...")
        
        try:
            # Wrap the network
            network = getattr(tabnet_clf, "network")
            model_wrapper = TabNetOnnxWrapper(network)
            model_wrapper.eval()
            model_wrapper.cpu()
            
            # Dummy Input
            dummy_input = torch.randn(1, n_features)
            
            # Export
            # WE REMOVE dynamic_axes due to torch.onnx Dynamo issues
            # We will use static batch size (1) for now, or need to configure dynamic_shapes properly
            print("Exporting model to ONNX...")
            torch.onnx.export(
                model_wrapper,
                (dummy_input,),
                output_path,
                export_params=True,
                opset_version=18, # Using 18 to match available opset and avoid conversion errors
                do_constant_folding=True,
                input_names=['float_input'],
                output_names=['label', 'probabilities']
                # dynamic_axes={...}  <-- REMOVED to avoid Dynamo constraints error
            )
            print("TabNet model serialized successfully.")
            
            # Post-processing: Check and repack if external data file was created
            if os.path.exists(output_path + ".data"):
                print("External data file detected. Attempting to repack into single ONNX file...")
                try:
                    import onnx
                    # Load model (loads external data automatically)
                    model_proto = onnx.load(output_path)
                    # Save model (defaults to single file if < 2GB)
                    onnx.save(model_proto, output_path)
                    print("Repack successful. Removing external .data file...")
                    os.remove(output_path + ".data")
                except Exception as rep_e:
                    print(f"Warning: Could not repack model to single file: {rep_e}")
            
        except Exception as e:
            # Handle potential encoding errors on Windows if the exception message contains special chars (like emojis)
            print("Error during TabNet ONNX export. See traceback below:")
            import traceback
            traceback.print_exc()
            
            # Safe print of exception message
            try:
                print(f"Exception message: {e}")
            except Exception:
                 print(f"Exception message (ascii): {str(e).encode('ascii', 'replace').decode('ascii')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Serialize best model to ONNX')
    parser.add_argument('--model_type', type=str, default='xgboost', help='Model type to serialize: xgboost or tabnet')
    args = parser.parse_args()
    
    serialize_best_model(model_type=args.model_type)