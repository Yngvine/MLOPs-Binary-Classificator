"""
Rice classification model inference.
"""

from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort

from .data import CLASS_LABELS, FEATURE_COLUMNS

# Assuming the model is stored in the mlflow/model directory
MODEL_PATH = Path(__file__).parent.parent / "mlflow" / "model" / "xgboost_binary.onnx"


class _ModelSession:
    """Singleton to hold the ONNX session."""

    instance: ort.InferenceSession | None = None

    @classmethod
    def get(cls) -> ort.InferenceSession:
        if cls.instance is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"ONNX model not found at {MODEL_PATH}. "
                    "Please ensure the model is trained and saved."
                )
            cls.instance = ort.InferenceSession(str(MODEL_PATH))
        return cls.instance


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
