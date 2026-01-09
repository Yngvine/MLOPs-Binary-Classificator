import mlflow
from mlflow import xgboost as mlflow_xgboost
from mlflow import pytorch as mlflow_pytorch
from mlflow import models as mlflow_models
from mlflow import sklearn as mlflow_sklearn
import optuna
import xgboost as xgb
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import cast

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score, 
    f1_score, 
    recall_score, 
    confusion_matrix
)
from imblearn.metrics import geometric_mean_score
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURATION ---
# Experiment Names
EXP_XGBOOST = "Rice_Classification_XGBoost"
EXP_TABNET = "Rice_Classification_TabNet"
EXP_COMPARISON = "Rice_Classification_Comparison"

# Registered Model Names
MODEL_XGBOOST = "Rice_XGBoost_Best"
MODEL_TABNET = "Rice_TabNet_Best"

N_TRIALS = 20  # Number of Optuna trials
N_FOLDS = 5    # K-Fold Cross Validation

# Load actual data
def load_data():
    # Load the actual dataset
    df = pd.read_csv("data/umbalanced/riceClassification_imbalanced.csv")
    
    # Separate features and target
    X = df.drop(columns=['Class', 'id'], errors='ignore') 
    y = df['Class']
    
    # If 'Class' is text (Gonen/Jasmine), encode it to 0/1
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"Classes encoded: {le.classes_}")

    return X.values, y

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculates the specific metrics requested."""
    
    # 1. Independent Metric: AUC-PR
    auc_pr = average_precision_score(y_true, y_prob)
    
    # 2. Dependent Metrics
    f1 = f1_score(y_true, y_pred)
    
    # Mean Per Class Recall (Macro Recall)
    mean_recall = recall_score(y_true, y_pred, average='macro')
    
    # Geometric Mean
    g_mean = geometric_mean_score(y_true, y_pred)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # CM Percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    metrics = {
        "auc_pr": auc_pr,
        "f1_score": f1,
        "mean_recall": mean_recall,
        "g_mean": g_mean,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }
    
    return metrics, cm, cm_norm

def objective_xgboost(trial, X, y, prefix: str = ""):
    """Optuna Objective Function for XGBoost.

    prefix: used to namespace parameter names when running mixed studies.
    """
    p = lambda name: f"{prefix}{name}"

    # Define Hyperparameter Search Space
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['aucpr', 'logloss'],
        'booster': 'gbtree',
        'lambda': trial.suggest_float(p('lambda'), 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float(p('alpha'), 1e-8, 1.0, log=True),
        'max_depth': trial.suggest_int(p('max_depth'), 1, 9),
        'eta': trial.suggest_float(p('eta'), 1e-8, 1.0, log=True),
        'gamma': trial.suggest_float(p('gamma'), 1e-8, 1.0, log=True),
        'grow_policy': trial.suggest_categorical(p('grow_policy'), ['depthwise', 'lossguide']),
        'scale_pos_weight': trial.suggest_float(p('scale_pos_weight'), 1, 100), # Important for imbalance
        'verbosity': 0
    }

    # K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    # Storage for evolution metrics across folds
    # Structure: {'train': {'aucpr': [[fold1], [fold2]...], 'logloss': ...}, 'val': ...}
    metrics_history = {
        'train': {'aucpr': [], 'logloss': []},
        'val': {'aucpr': [], 'logloss': []}
    }
    
    fold_metrics_agg = {k: [] for k in ["auc_pr", "f1_score", "mean_recall", "g_mean"]}
    
    
    # We start a nested run for the trial
    with mlflow.start_run(nested=True):
        # Log params for this trial
        mlflow.log_params(params)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            evals_result = {}
            
            # Train Model - evals to capture history
            model = xgb.train(
                params, 
                dtrain, 
                num_boost_round=100,
                evals=[(dtrain, 'train'), (dval, 'val')],
                evals_result=evals_result,
                verbose_eval=False
            )

            # Collect history for this fold
            for dataset in ['train', 'val']:
                for metric in ['aucpr', 'logloss']:
                    # Check if metric exists (it should)
                    if metric in evals_result.get(dataset, {}):
                        metrics_history[dataset][metric].append(evals_result[dataset][metric])

            
            # Predictions
            y_prob = model.predict(dval)
            y_pred = (y_prob > 0.5).astype(int) # Threshold can be tuned, using 0.5 for now
            
            # Calculate Metrics
            metrics, cm, cm_norm = calculate_metrics(y_val, y_pred, y_prob)
            
            # Aggregate
            for k in fold_metrics_agg:
                fold_metrics_agg[k].append(metrics[k])
        
        # Average metrics across folds
        avg_auc_pr = cast(float, np.mean(fold_metrics_agg["auc_pr"]))
        avg_f1 = cast(float, np.mean(fold_metrics_agg["f1_score"]))
        avg_g_mean = cast(float, np.mean(fold_metrics_agg["g_mean"]))
        avg_recall = cast(float, np.mean(fold_metrics_agg["mean_recall"]))
        
        # Log averaged metrics to MLflow
        mlflow.log_metric("val_auc_pr", avg_auc_pr)
        mlflow.log_metric("val_f1", avg_f1)
        mlflow.log_metric("val_g_mean", avg_g_mean)
        mlflow.log_metric("val_mean_recall", avg_recall)

        # 2. Log Averaged Evolution Metrics (Curves over epochs)
        # We average the history across the 5 folds to get a smooth representative curve
        for dataset in ['train', 'val']:
            for metric in ['aucpr', 'logloss']:
                histories = metrics_history[dataset][metric]
                if histories:
                    # Convert list of lists to numpy array -> (n_folds, n_rounds)
                    # Calculate mean across folds -> (n_rounds,)
                    avg_curve = np.mean(np.array(histories), axis=0)
                    
                    # Log each step to MLflow
                    for step, value in enumerate(avg_curve):
                        mlflow.log_metric(f"{dataset}_{metric}_mean", value, step=step)

        # Store secondary metrics in Optuna trial so we can log them to Parent later
        trial.set_user_attr("val_f1", avg_f1)
        trial.set_user_attr("val_g_mean", avg_g_mean)
        trial.set_user_attr("val_mean_recall", avg_recall)
        
        # Return the metric we want to OPTIMIZE (AUC-PR)
        return avg_auc_pr

def objective_tabnet(trial, X, y, prefix: str = ""):
    """Optuna Objective Function for TabNet.

    prefix: used to namespace parameter names when running mixed studies.
    """
    p = lambda name: f"{prefix}{name}"
    
    # Define Hyperparameter Search Space
    n_d = trial.suggest_int(p('n_d'), 8, 64)
    # n_a is usually equal to n_d
    n_a = n_d
    
    params = {
        'n_d': n_d,
        'n_a': n_a, 
        'n_steps': trial.suggest_int(p('n_steps'), 3, 10),
        'gamma': trial.suggest_float(p('gamma'), 1.0, 2.0),
        'lambda_sparse': trial.suggest_float(p('lambda_sparse'), 1e-6, 1e-1, log=True),
        'optimizer_params': dict(lr=trial.suggest_float(p('lr'), 2e-2, 2e-1)),
        'mask_type': trial.suggest_categorical(p('mask_type'), ['entmax', 'sparsemax']),
        'verbose': 0,
        'seed': 42
    }
    
    batch_size = trial.suggest_categorical(p('batch_size'), [256, 512, 1024, 2048])
    virtual_batch_size = trial.suggest_categorical(p('virtual_batch_size'), [128, 256])
    max_epochs = 100 

    # K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    fold_metrics_agg = {k: [] for k in ["auc_pr", "f1_score", "mean_recall", "g_mean"]}
    
    # We start a nested run for the trial
    with mlflow.start_run(nested=True):
        # Log params for this trial
        mlflow.log_params(params)
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('virtual_batch_size', virtual_batch_size)
        mlflow.log_param('model_type', 'tabnet')
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # TabNetClassifier initialization
            model = TabNetClassifier(**params)
            
            # Fit
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_name=['train', 'val'],
                eval_metric=['auc'], # Using inbuilt ROC-AUC for early stopping
                max_epochs=max_epochs,
                patience=15,
                batch_size=batch_size,
                virtual_batch_size=virtual_batch_size,
                num_workers=0,
                drop_last=False
            )
            
            # Predictions
            y_prob = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)
            
            # Calculate Metrics (reuse existing function)
            metrics, cm, cm_norm = calculate_metrics(y_val, y_pred, y_prob)
            
            # Aggregate
            for k in fold_metrics_agg:
                fold_metrics_agg[k].append(metrics[k])
        
        # Average metrics across folds
        avg_auc_pr = cast(float, np.mean(fold_metrics_agg["auc_pr"]))
        avg_f1 = cast(float, np.mean(fold_metrics_agg["f1_score"]))
        avg_g_mean = cast(float, np.mean(fold_metrics_agg["g_mean"]))
        avg_recall = cast(float, np.mean(fold_metrics_agg["mean_recall"]))
        
        # Log averaged metrics to MLflow
        mlflow.log_metric("val_auc_pr", avg_auc_pr)
        mlflow.log_metric("val_f1", avg_f1)
        mlflow.log_metric("val_g_mean", avg_g_mean)
        mlflow.log_metric("val_mean_recall", avg_recall)

        # Store secondary metrics in Optuna trial so we can log them to Parent later
        trial.set_user_attr("val_f1", avg_f1)
        trial.set_user_attr("val_g_mean", avg_g_mean)
        trial.set_user_attr("val_mean_recall", avg_recall)
        
        # Return the metric we want to OPTIMIZE (AUC-PR)
        return avg_auc_pr

def objective_combined(trial, X, y):
    """Optuna Objective Function that selects between XGBoost and TabNet"""
    classifier_name = trial.suggest_categorical("classifier", ["xgboost", "tabnet"])
    
    if classifier_name == "xgboost":
        return objective_xgboost(trial, X, y, prefix="xgb_")
    else:
        return objective_tabnet(trial, X, y, prefix="tab_")

def run_optimization(model_type="xgboost"):
    X, y = load_data()
    
    # Determine Experiment and Model Name
    if model_type == "xgboost":
        experiment_name = EXP_XGBOOST
        model_name = MODEL_XGBOOST
        objective_func = lambda trial: objective_xgboost(trial, X, y)
    elif model_type == "tabnet":
        experiment_name = EXP_TABNET
        model_name = MODEL_TABNET
        objective_func = lambda trial: objective_tabnet(trial, X, y)
    elif model_type == "comparison":
        experiment_name = EXP_COMPARISON
        model_name = None # We don't register models in comparison mode
        objective_func = lambda trial: objective_combined(trial, X, y)
    else:
         raise ValueError("Invalid model_type. Choose 'xgboost', 'tabnet', or 'comparison'")

    mlflow.set_experiment(experiment_name)

    # Parent Run: The Optimization Process
    run_name = f"Optuna_Optimization_{model_type.upper()}"
    with mlflow.start_run(run_name=run_name):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_func, n_trials=N_TRIALS)
        
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        # Log best params to parent run
        mlflow.log_params(trial.params)

        # Log best metrics to parent run (So they show up in the main UI table)
        mlflow.log_metric("val_auc_pr", cast(float, trial.value))
        mlflow.log_metric("val_f1", cast(float, trial.user_attrs["val_f1"]))
        mlflow.log_metric("val_g_mean", cast(float, trial.user_attrs["val_g_mean"]))
        mlflow.log_metric("val_mean_recall", cast(float, trial.user_attrs["val_mean_recall"]))

        # If comparison mode, we stop here (no retraining/registration)
        if model_type == "comparison":
            print("Comparison experiment complete. Check MLflow UI for results.")
            return

        # --- RETRAIN BEST MODEL ON FULL DATA ---
        print(f"Retraining best {model_type} model on full dataset...")
        best_params = trial.params
        
        if model_type == "xgboost":
            best_params['objective'] = 'binary:logistic' # Ensure objective is set
            
            dtrain = xgb.DMatrix(X, label=y)
            final_model = xgb.train(best_params, dtrain, num_boost_round=100)
            
            # Log the final model
            signature = mlflow_models.infer_signature(X, final_model.predict(dtrain))
            
            mlflow_xgboost.log_model(
                final_model, 
                "model", 
                signature=signature,
                registered_model_name=model_name
            )
            
        elif model_type == "tabnet":
            # Reconstruct params for TabNet
            tabnet_params = {
                'n_d': best_params['n_d'],
                'n_a': best_params['n_d'], # n_a = n_d
                'n_steps': best_params['n_steps'],
                'gamma': best_params['gamma'],
                'lambda_sparse': best_params['lambda_sparse'],
                'optimizer_params': dict(lr=best_params['lr']),
                'mask_type': best_params['mask_type'],
                'verbose': 0,
                'seed': 42
            }
            batch_size = best_params['batch_size']
            virtual_batch_size = best_params['virtual_batch_size']
            
            final_model = TabNetClassifier(**tabnet_params)
            final_model.fit(
                X, y,
                eval_set=[(X, y)],
                eval_name=['train'],
                eval_metric=['auc'],
                max_epochs=200,
                patience=20,
                batch_size=batch_size,
                virtual_batch_size=virtual_batch_size,
                num_workers=0,
                drop_last=False
            )
            
            signature = mlflow_models.infer_signature(X, final_model.predict(X))
            
            mlflow_sklearn.log_model(
                final_model, 
                "model", 
                signature=signature,
                registered_model_name=model_name
            )

        
        # Save feature count for serialization script
        with open("feature_metadata.json", "w") as f:
            json.dump({"n_features": X.shape[1]}, f)
        mlflow.log_artifact("feature_metadata.json")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run Optuna optimization for XGBoost or TabNet')
    parser.add_argument('--model_type', type=str, default='xgboost', 
                        choices=['xgboost', 'tabnet', 'comparison'],
                        help='Model type to optimize: xgboost, tabnet, or comparison')
    args = parser.parse_args()
    
    run_optimization(model_type=args.model_type)