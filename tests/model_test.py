import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from custom_lib.model import predict

class TestModelPrediction(unittest.TestCase):
    @patch('custom_lib.model._get_ort_session')
    def test_predict_gonen(self, mock_get_session):
        """Test prediction for class 0 (Gonen)."""
        # Setup mock session
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        
        # Mock inputs/outputs
        mock_session.get_inputs.return_value = [MagicMock(name='input')]
        mock_session.get_outputs.return_value = [MagicMock(name='output')]
        
        # Mock return value: Class 0 higher probability
        # Shape (1, 2) -> [[0.9, 0.1]]
        mock_session.run.return_value = [np.array([[0.9, 0.1]], dtype=np.float32)]
        
        features = [100.0] * 10
        result = predict(features)
        
        self.assertEqual(result, "Gonen")

    @patch('custom_lib.model._get_ort_session')
    def test_predict_jasmine(self, mock_get_session):
        """Test prediction for class 1 (Jasmine)."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        mock_session.get_inputs.return_value = [MagicMock(name='input')]
        mock_session.get_outputs.return_value = [MagicMock(name='output')]
        
        # Mock return value: Class 1 higher probability
        mock_session.run.return_value = [np.array([[0.1, 0.9]], dtype=np.float32)]
        
        features = [100.0] * 10
        result = predict(features)
        
        self.assertEqual(result, "Jasmine")

    def test_predict_invalid_feature_count(self):
        """Test that ValueError is raised for wrong number of features."""
        features = [100.0] * 5 # Only 5 features
        # We need to mock session even here because predict calls _get_ort_session early
        with patch('custom_lib.model._get_ort_session') as mock_get_session:
            with self.assertRaises(ValueError):
                predict(features)
