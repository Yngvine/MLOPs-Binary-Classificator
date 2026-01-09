"""
Rice classification model inference.
"""

from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort

from .data import CLASS_LABELS, FEATURE_COLUMNS

# Assuming the model is stored in the mlflow/model directory (DEFAULT)
MODEL_DIR = Path(__file__).parent.parent / "mlflow" / "model"
# Default model path - can be switched
DEFAULT_MODEL_NAME = "xgboost_binary.onnx"


class _ModelSession:
    """Singleton to hold the ONNX session."""

    instance: ort.InferenceSession | None = None
    current_model_name: str = DEFAULT_MODEL_NAME

    @classmethod
    def get(cls) -> ort.InferenceSession:
        return cls._get_session(cls.current_model_name)

    @classmethod
    def _get_session(cls, model_name: str) -> ort.InferenceSession:
        """Helper to get or create session for a specific model name"""
        # If we are requesting a different model than loaded, or no model is loaded
        if cls.instance is None or cls.current_model_name != model_name:
            model_path = MODEL_DIR / model_name
            
            if not model_path.exists():
                raise FileNotFoundError(
                    f"ONNX model not found at {model_path}. "
                    "Please ensure the model is available."
                )
            
            print(f"Loading ONNX model: {model_name}")
            cls.instance = ort.InferenceSession(str(model_path))
            cls.current_model_name = model_name
            
        return cls.instance
    
    @classmethod
    def set_model(cls, model_name: str):
        """Swaps the active model if it exists."""
        model_path = MODEL_DIR / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_name} not found in {MODEL_DIR}")
        
        # Force reload
        cls._get_session(model_name)

def set_active_model(model_name: str):
    """
    Switch the active model used for predictions.
    
    Parameters
    ----------
    model_name : str
        Filename of the .onnx model to use (e.g. 'xgboost_binary.onnx')
    """
    _ModelSession.set_model(model_name)

def _get_ort_session() -> ort.InferenceSession:
    """
    Lazy load the ONNX session.

    Returns
    -------
    ort.InferenceSession
        The loaded ONNX runtime inference session.
    """
    return _ModelSession.get()


def predict(features: np.ndarray | list[float]) -> str:
    """
    Predict rice class from features.

    Parameters
    ----------
    features : np.ndarray | list[float]
        Array-like containing the 10 features:
        Area, MajorAxisLength, MinorAxisLength, Eccentricity,
        ConvexArea, EquivDiameter, Extent, Perimeter,
        Roundness, AspectRation.

    Returns
    -------
    str
        Predicted class label.

    Raises
    ------
    ValueError
        If the input features do not match the expected count (10).
    """
    session = _get_ort_session()

    # Ensure input is numpy array
    if isinstance(features, list):
        data = np.array(features, dtype=np.float32)
    else:
        data = features.astype(np.float32)

    # Handle single sample (1D -> 2D)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # Validate feature count
    if data.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Expected {len(FEATURE_COLUMNS)} features, got {data.shape[1]}. "
            f"Features: {FEATURE_COLUMNS}"
        )

    # Run inference
    input_name = session.get_inputs()[0].name
    # Output 0 is the label (int64), Output 1 is probabilities
    output_name = session.get_outputs()[0].name

    outputs = session.run([output_name], {input_name: data})

    # outputs[0] contains the predicted labels directly (e.g., [0] or [1])
    predictions = cast(np.ndarray, outputs[0])
    predicted_idx = predictions[0]

    return CLASS_LABELS.get(int(predicted_idx), "Unknown")
