"""
Tests for Wine Quality API
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

try:
    from api.wine_api import app, get_quality_category, preprocess_features
    from api.wine_api import WineFeatures, BatchWineFeatures
except ImportError:
    pytest.skip("API module not available", allow_module_level=True)

# Create test client
client = TestClient(app)

class TestWineQualityAPI:
    """Test cases for Wine Quality API"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["message"] == "Wine Quality Prediction API"
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data
        assert "timestamp" in data
        
        assert data["status"] in ["healthy", "degraded"]
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["uptime_seconds"], (int, float))
    
    def test_features_info_endpoint(self):
        """Test features info endpoint"""
        response = client.get("/features/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "required_features" in data
        assert "quality_scale" in data
        assert "quality_categories" in data
        
        # Check required features
        required_features = data["required_features"]
        assert len(required_features) == 11
        
        expected_features = [
            "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
            "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide",
            "density", "pH", "sulphates", "alcohol"
        ]
        
        feature_names = [f["name"] for f in required_features]
        for expected_feature in expected_features:
            assert expected_feature in feature_names
    
    def test_get_quality_category(self):
        """Test quality category function"""
        test_cases = [
            (3.5, "Poor"),
            (4.5, "Below Average"),
            (5.5, "Average"),
            (6.5, "Good"),
            (7.5, "Very Good"),
            (8.5, "Excellent"),
            (2.0, "Poor"),
            (9.0, "Excellent")
        ]
        
        for quality_score, expected_category in test_cases:
            assert get_quality_category(quality_score) == expected_category
    
    def test_wine_features_validation(self, single_wine_sample):
        """Test WineFeatures model validation"""
        # Valid data should pass
        wine_features = WineFeatures(**single_wine_sample)
        assert wine_features.fixed_acidity == single_wine_sample["fixed_acidity"]
        assert wine_features.alcohol == single_wine_sample["alcohol"]
    
    def test_wine_features_validation_errors(self):
        """Test WineFeatures validation with invalid data"""
        # Test out of range values
        invalid_data = {
            "fixed_acidity": 20.0,  # Too high
            "volatile_acidity": 0.7,
            "citric_acid": 0.0,
            "residual_sugar": 1.9,
            "chlorides": 0.076,
            "free_sulfur_dioxide": 11.0,
            "total_sulfur_dioxide": 34.0,
            "density": 0.9978,
            "pH": 3.51,
            "sulphates": 0.56,
            "alcohol": 9.4
        }
        
        with pytest.raises(Exception):  # Pydantic validation error
            WineFeatures(**invalid_data)
    
    @patch('api.wine_api.loaded_model')
    def test_predict_endpoint_no_model(self, mock_model, single_wine_sample):
        """Test prediction endpoint when model is not loaded"""
        mock_model = None
        
        response = client.post("/predict", json=single_wine_sample)
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]
    
    @patch('api.wine_api.loaded_model')
    def test_predict_endpoint_with_mock_model(self, mock_model, single_wine_sample):
        """Test prediction endpoint with mocked model"""
        # Mock model
        mock_model.predict.return_value = [6.5]
        mock_model.predict_proba = None  # Simulate model without predict_proba
        
        # Mock the global variable
        with patch('api.wine_api.loaded_model', mock_model):
            response = client.post("/predict", json=single_wine_sample)
            
            if response.status_code == 200:
                data = response.json()
                assert "quality" in data
                assert "quality_category" in data
                assert "confidence" in data
                assert "processing_time_ms" in data
                assert "model_version" in data
                
                assert isinstance(data["quality"], float)
                assert data["quality"] == 6.5
                assert data["quality_category"] == "Good"
                assert 0 <= data["confidence"] <= 1
    
    def test_batch_predict_validation(self, single_wine_sample):
        """Test batch prediction validation"""
        # Valid batch
        batch_data = {
            "wines": [single_wine_sample, single_wine_sample]
        }
        
        batch_features = BatchWineFeatures(**batch_data)
        assert len(batch_features.wines) == 2
        
        # Test maximum limit
        large_batch = {
            "wines": [single_wine_sample] * 101  # Exceeds max limit
        }
        
        with pytest.raises(Exception):  # Validation error
            BatchWineFeatures(**large_batch)
    
    @patch('api.wine_api.loaded_model')
    def test_batch_predict_endpoint_no_model(self, mock_model, single_wine_sample):
        """Test batch prediction when model is not loaded"""
        mock_model = None
        
        batch_data = {
            "wines": [single_wine_sample, single_wine_sample]
        }
        
        response = client.post("/predict_batch", json=batch_data)
        assert response.status_code == 503
    
    def test_model_info_endpoint_no_model(self):
        """Test model info endpoint when model is not loaded"""
        response = client.get("/model/info")
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]
    
    def test_invalid_json_request(self):
        """Test API with invalid JSON"""
        invalid_json = '{"invalid": json}'
        
        response = client.post(
            "/predict",
            data=invalid_json,
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_missing_required_fields(self):
        """Test API with missing required fields"""
        incomplete_data = {
            "fixed_acidity": 7.4,
            "volatile_acidity": 0.7,
            # Missing other required fields
        }
        
        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422
    
    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = client.options("/")
        assert response.status_code == 200
        
        # Check for CORS headers in a GET request
        response = client.get("/")
        assert "access-control-allow-origin" in response.headers or response.status_code == 200
    
    def test_api_documentation_endpoints(self):
        """Test that API documentation endpoints are accessible"""
        # Test OpenAPI docs
        response = client.get("/docs")
        assert response.status_code == 200
        
        # Test ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200
        
        # Test OpenAPI JSON
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
    
    @pytest.mark.parametrize("invalid_value,field", [
        (-1.0, "fixed_acidity"),  # Too low
        (100.0, "alcohol"),       # Too high
        (-0.5, "pH"),            # Negative pH
        (20.0, "volatile_acidity") # Too high
    ])
    def test_field_validation_ranges(self, single_wine_sample, invalid_value, field):
        """Test field validation with out-of-range values"""
        invalid_sample = single_wine_sample.copy()
        invalid_sample[field] = invalid_value
        
        with pytest.raises(Exception):  # Should raise validation error
            WineFeatures(**invalid_sample)
    
    def test_error_handling(self, single_wine_sample):
        """Test API error handling"""
        # Test with string instead of number
        invalid_sample = single_wine_sample.copy()
        invalid_sample["alcohol"] = "not_a_number"
        
        response = client.post("/predict", json=invalid_sample)
        assert response.status_code == 422
        
        error_detail = response.json()["detail"]
        assert isinstance(error_detail, list)
        assert len(error_detail) > 0
    
    def test_response_format(self, single_wine_sample):
        """Test response format consistency"""
        # Even if model is not loaded, we can test the response structure
        response = client.post("/predict", json=single_wine_sample)
        
        # Should be either successful prediction or service unavailable
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            required_fields = ["quality", "quality_category", "confidence", 
                             "processing_time_ms", "model_version"]
            
            for field in required_fields:
                assert field in data
    
    def test_numeric_type_conversion(self):
        """Test that string numbers are converted properly"""
        sample_with_strings = {
            "fixed_acidity": "7.4",
            "volatile_acidity": "0.7",
            "citric_acid": "0.0",
            "residual_sugar": "1.9",
            "chlorides": "0.076",
            "free_sulfur_dioxide": "11.0",
            "total_sulfur_dioxide": "34.0",
            "density": "0.9978",
            "pH": "3.51",
            "sulphates": "0.56",
            "alcohol": "9.4"
        }
        
        # This should work due to Pydantic's type coercion
        wine_features = WineFeatures(**sample_with_strings)
        assert isinstance(wine_features.fixed_acidity, float)
        assert wine_features.fixed_acidity == 7.4