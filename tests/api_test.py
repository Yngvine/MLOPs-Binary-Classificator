from fastapi.testclient import TestClient
from unittest.mock import patch
from api.api_main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

@patch('api.api_main.predict')
def test_classify_gonen(mock_predict):
    """Test API endpoint for Gonen prediction."""
    mock_predict.return_value = "Gonen"
    
    payload = {
        "Area": 1000, "MajorAxisLength": 100.0, "MinorAxisLength": 50.0,
        "Eccentricity": 0.5, "ConvexArea": 1050, "EquivDiameter": 80.0,
        "Extent": 0.6, "Perimeter": 300.0, "Roundness": 0.8, "AspectRation": 2.0
    }
    
    response = client.post("/classify/", json=payload)
    
    assert response.status_code == 200
    assert response.json() == {"predicted_class": "Gonen"}
    
    # Verify correct list conversion
    expected_features = [1000, 100.0, 50.0, 0.5, 1050, 80.0, 0.6, 300.0, 0.8, 2.0]
    mock_predict.assert_called_once_with(expected_features)

@patch('api.api_main.predict')
def test_classify_jasmine(mock_predict):
    """Test API endpoint for Jasmine prediction."""
    mock_predict.return_value = "Jasmine"
    
    payload = {
        "Area": 2000, "MajorAxisLength": 200.0, "MinorAxisLength": 100.0,
        "Eccentricity": 0.6, "ConvexArea": 2050, "EquivDiameter": 160.0,
        "Extent": 0.7, "Perimeter": 600.0, "Roundness": 0.9, "AspectRation": 2.5
    }
    
    response = client.post("/classify/", json=payload)
    
    assert response.status_code == 200
    assert response.json() == {"predicted_class": "Jasmine"}

def test_classify_invalid_json():
    """Test API endpoint with missing fields."""
    payload = {"Area": 1000} # Missing other fields
    response = client.post("/classify/", json=payload)
    assert response.status_code == 422 # Validation error

@patch('api.api_main.predict')
def test_classify_server_error(mock_predict):
    """Test API endpoint handles exceptions gracefully."""
    mock_predict.side_effect = Exception("Model inference failed")
    
    payload = {
        "Area": 1000, "MajorAxisLength": 100.0, "MinorAxisLength": 50.0,
        "Eccentricity": 0.5, "ConvexArea": 1050, "EquivDiameter": 80.0,
        "Extent": 0.6, "Perimeter": 300.0, "Roundness": 0.8, "AspectRation": 2.0
    }
    
    response = client.post("/classify/", json=payload)
    
    assert response.status_code == 500
    assert "Model inference failed" in response.json()['detail']

@patch('api.api_main.set_active_model')
@patch('api.api_main.predict')
def test_classify_switches_model(mock_predict, mock_set_active_model):
    """Test that classify endpoint triggers model switching."""
    mock_predict.return_value = "Gonen"
    
    payload = {
        "Area": 1000, "MajorAxisLength": 100.0, "MinorAxisLength": 50.0,
        "Eccentricity": 0.5, "ConvexArea": 1050, "EquivDiameter": 80.0,
        "Extent": 0.6, "Perimeter": 300.0, "Roundness": 0.8, "AspectRation": 2.0,
        "ModelName": "custom_model.onnx"
    }
    
    response = client.post("/classify/", json=payload)
    
    assert response.status_code == 200
    # Verify set_active_model was called with the model name from payload
    mock_set_active_model.assert_called_once_with("custom_model.onnx")

def test_metrics_endpoint():
    """Test /metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "rice_predictions_total" in response.text


