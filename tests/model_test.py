import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from custom_lib.model import predict, set_active_model, _ModelSession

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
        
        # Mock return value: Class 0 (Gonen)
        # Shape (1,) -> [0]
        mock_session.run.return_value = [np.array([0], dtype=np.int64)]
        
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
        
        # Mock return value: Class 1 (Jasmine)
        mock_session.run.return_value = [np.array([1], dtype=np.int64)]
        
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

class TestModelSwitching(unittest.TestCase):
    @patch('custom_lib.model.ort.InferenceSession')
    @patch('custom_lib.model.Path.exists')
    def test_set_active_model_success(self, mock_exists, mock_session_cls):
        """Test that set_active_model updates the session."""
        mock_exists.return_value = True
        
        # Reset the singleton before test
        _ModelSession.instance = None
        _ModelSession.current_model_name = "default"

        # Mock the session instance
        mock_session_instance = MagicMock()
        mock_session_cls.return_value = mock_session_instance

        # Switch to new model
        set_active_model("new_model.onnx")

        # Verify Path existence check was called
        self.assertTrue(mock_exists.called)
        
        # Verify InferenceSession was initialized
        mock_session_cls.assert_called_once()
        self.assertEqual(_ModelSession.current_model_name, "new_model.onnx")
        self.assertEqual(_ModelSession.instance, mock_session_instance)

    @patch('custom_lib.model.Path.exists')
    def test_set_active_model_not_found(self, mock_exists):
        """Test that FileNotFoundError is raised for missing model."""
        mock_exists.return_value = False
        
        with self.assertRaises(FileNotFoundError):
            set_active_model("missing_model.onnx")

    @patch('custom_lib.model.ort.InferenceSession')
    @patch('custom_lib.model.Path.exists')
    def test_no_reload_on_same_model(self, mock_exists, mock_session_cls):
        """Test that setting the same model does not reload the session."""
        mock_exists.return_value = True
        
         # Set up initial state
        _ModelSession.current_model_name = "same_model.onnx"
        mock_initial_session = MagicMock()
        _ModelSession.instance = mock_initial_session
        
        # Call set_active_model with same name
        set_active_model("same_model.onnx")
        
        # Should NOT call InferenceSession constructor again
        mock_session_cls.assert_not_called()
        self.assertEqual(_ModelSession.instance, mock_initial_session)

class TestModelIntegration(unittest.TestCase):
    @patch('custom_lib.model.ort.InferenceSession')
    @patch('custom_lib.model.Path.exists')
    def test_predict_full_flow(self, mock_exists, mock_session_cls):
        """Test predict without patching _get_ort_session to cover lazy load logic."""
        mock_exists.return_value = True
        
        # Reset singleton
        _ModelSession.instance = None
        current_model = _ModelSession.current_model_name
        
        # Mock session run
        mock_session_instance = MagicMock()
        mock_session_cls.return_value = mock_session_instance
        # Mock run return: [array([0])]
        mock_session_instance.run.return_value = [np.array([0], dtype=np.int64)]
        mock_session_instance.get_inputs.return_value = [MagicMock(name='input')]
        mock_session_instance.get_outputs.return_value = [MagicMock(name='output')]

        # Test with Numpy array to cover line 109
        features = np.array([100.0] * 10)
        
        result = predict(features)
        
        self.assertEqual(result, "Gonen")
        
        # Verify checking existence of default model
        mock_exists.assert_called()
        # Verify session created
        mock_session_cls.assert_called()

    @patch('custom_lib.model.Path.exists')
    def test_predict_default_model_missing(self, mock_exists):
        """Test predict raises FileNotFoundError if default model missing."""
        mock_exists.return_value = False
        _ModelSession.instance = None # Ensure we try to load
        
        features = np.array([100.0] * 10)
        
        with self.assertRaises(FileNotFoundError):
            predict(features)


