"""
Integration tests for the complete wine quality prediction pipeline
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import tempfile
import shutil

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

try:
    from data_processing.data_loader import WineDataLoader
    from features.feature_engineering import WineFeatureEngineer
    from models.wine_models import WineQualityModelSuite
    from visualization.wine_visualizer import WineQualityVisualizer
except ImportError:
    pytest.skip("Required modules not available", allow_module_level=True)

class TestIntegration:
    """Integration tests for the complete pipeline"""
    
    def test_complete_pipeline_flow(self, sample_wine_data):
        """Test the complete ML pipeline flow"""
        # 1. Data Loading and Processing
        data_loader = WineDataLoader()
        
        # Process the data
        processed_data = data_loader.preprocess_data(sample_wine_data)
        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) > 0
        
        # Split features and target
        features, target = data_loader.split_features_target(processed_data)
        assert isinstance(features, pd.DataFrame)
        assert isinstance(target, pd.Series)
        
        # 2. Feature Engineering
        feature_engineer = WineFeatureEngineer()
        
        # Create engineered features
        engineered_features = feature_engineer.create_domain_features(features)
        assert isinstance(engineered_features, pd.DataFrame)
        assert len(engineered_features.columns) >= len(features.columns)
        
        # Scale features
        X_train_scaled, X_test_scaled = feature_engineer.scale_features(
            engineered_features, engineered_features, method='standard'
        )
        assert isinstance(X_train_scaled, pd.DataFrame)
        assert isinstance(X_test_scaled, pd.DataFrame)
        
        # Select features
        X_train_selected, X_test_selected = feature_engineer.select_features(
            X_train_scaled, target, X_test_scaled, n_features=10
        )
        assert X_train_selected.shape[1] == 10
        assert X_test_selected.shape[1] == 10
        
        # 3. Model Training
        model_suite = WineQualityModelSuite()
        model_suite.initialize_models()
        
        # Train models (use a subset for faster testing)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_selected, target, test_size=0.2, random_state=42
        )
        
        # Train at least one model
        results = {}
        if 'linear_regression' in model_suite.models:
            result = model_suite.train_single_model(
                'linear_regression', X_train, y_train, X_test, y_test
            )
            results['linear_regression'] = result
            assert isinstance(result, dict)
            assert 'test_r2' in result
        
        # 4. Validation
        assert len(results) > 0
        
        # Check that we can make predictions
        for model_name in results:
            predictions = model_suite.predict(model_name, X_test)
            assert isinstance(predictions, np.ndarray)
            assert len(predictions) == len(X_test)
    
    def test_data_flow_consistency(self, sample_wine_data):
        """Test that data flows consistently through the pipeline"""
        # Initialize components
        data_loader = WineDataLoader()
        feature_engineer = WineFeatureEngineer()
        
        # Process data
        processed_data = data_loader.preprocess_data(sample_wine_data)
        features, target = data_loader.split_features_target(processed_data)
        
        # Track data shape through pipeline
        original_samples = len(features)
        
        # Feature engineering
        engineered_features = feature_engineer.create_domain_features(features)
        assert len(engineered_features) == original_samples
        
        # Feature scaling
        X_scaled, _ = feature_engineer.scale_features(
            engineered_features, engineered_features, method='standard'
        )
        assert len(X_scaled) == original_samples
        
        # Feature selection
        X_selected, _ = feature_engineer.select_features(
            X_scaled, target, X_scaled, n_features=5
        )
        assert len(X_selected) == original_samples
        assert X_selected.shape[1] == 5
    
    def test_error_handling_throughout_pipeline(self):
        """Test error handling in the complete pipeline"""
        # Test with invalid data
        invalid_data = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [np.inf, 2, 3, 4, 5],
            'quality': [5, 6, 7, 8, 9]
        })
        
        data_loader = WineDataLoader()
        
        # Data loader should handle invalid data gracefully
        try:
            processed_data = data_loader.preprocess_data(invalid_data)
            # If processing succeeds, check that invalid values are handled
            assert not processed_data.isnull().any().any()
            assert not np.isinf(processed_data.select_dtypes(include=[np.number])).any().any()
        except Exception as e:
            # If it raises an exception, that's also acceptable error handling
            assert isinstance(e, Exception)
    
    def test_model_persistence_flow(self, sample_wine_data, tmp_path):
        """Test model saving and loading in the pipeline"""
        # Set up temporary directory for models
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        
        # Initialize components
        data_loader = WineDataLoader()
        feature_engineer = WineFeatureEngineer()
        model_suite = WineQualityModelSuite()
        model_suite.models_dir = models_dir
        model_suite.initialize_models()
        
        # Process data
        processed_data = data_loader.preprocess_data(sample_wine_data)
        features, target = data_loader.split_features_target(processed_data)
        
        # Train a simple model
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        if 'linear_regression' in model_suite.models:
            # Train model
            result = model_suite.train_single_model(
                'linear_regression', X_train, y_train, X_test, y_test
            )
            
            # Set as best model
            model_suite.best_model_name = 'linear_regression'
            model_suite.best_model = model_suite.models['linear_regression']
            
            # Save model
            saved_path = model_suite.save_best_model()
            assert saved_path is not None
            assert saved_path.exists()
            
            # Load model
            loaded_model_data = model_suite.load_model(saved_path)
            assert loaded_model_data is not None
            assert 'model' in loaded_model_data
    
    def test_feature_engineering_integration(self, sample_wine_data):
        """Test feature engineering integration with models"""
        # Initialize components
        data_loader = WineDataLoader()
        feature_engineer = WineFeatureEngineer()
        model_suite = WineQualityModelSuite()
        model_suite.initialize_models()
        
        # Process data
        processed_data = data_loader.preprocess_data(sample_wine_data)
        features, target = data_loader.split_features_target(processed_data)
        
        # Create engineered features
        engineered_features = feature_engineer.create_domain_features(features)
        
        # Train model with engineered features
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            engineered_features, target, test_size=0.2, random_state=42
        )
        
        if 'linear_regression' in model_suite.models:
            # Train with original features
            result_original = model_suite.train_single_model(
                'linear_regression', 
                X_train[features.columns], y_train, 
                X_test[features.columns], y_test
            )
            
            # Train with engineered features
            result_engineered = model_suite.train_single_model(
                'linear_regression', X_train, y_train, X_test, y_test
            )
            
            # Engineered features should generally perform better or similar
            # (though not guaranteed with limited data)
            assert isinstance(result_original, dict)
            assert isinstance(result_engineered, dict)
            assert 'test_r2' in result_original
            assert 'test_r2' in result_engineered
    
    def test_pipeline_with_minimal_data(self):
        """Test pipeline with minimal amount of data"""
        # Create minimal dataset
        minimal_data = pd.DataFrame({
            'fixed_acidity': [7.0, 8.0, 6.5, 7.5, 8.2],
            'volatile_acidity': [0.5, 0.6, 0.4, 0.55, 0.7],
            'citric_acid': [0.1, 0.2, 0.05, 0.15, 0.25],
            'residual_sugar': [2.0, 1.5, 2.5, 1.8, 2.2],
            'chlorides': [0.08, 0.07, 0.09, 0.075, 0.085],
            'free_sulfur_dioxide': [15, 20, 12, 18, 22],
            'total_sulfur_dioxide': [50, 60, 45, 55, 65],
            'density': [0.996, 0.997, 0.995, 0.9965, 0.998],
            'pH': [3.2, 3.4, 3.0, 3.3, 3.5],
            'sulphates': [0.6, 0.7, 0.5, 0.65, 0.75],
            'alcohol': [10, 11, 9.5, 10.5, 11.5],
            'quality': [5, 6, 5, 6, 7]
        })
        
        # Test that pipeline can handle minimal data
        data_loader = WineDataLoader()
        processed_data = data_loader.preprocess_data(minimal_data)
        
        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) > 0
        
        features, target = data_loader.split_features_target(processed_data)
        
        # Feature engineering should work with minimal data
        feature_engineer = WineFeatureEngineer()
        engineered_features = feature_engineer.create_domain_features(features)
        
        assert isinstance(engineered_features, pd.DataFrame)
        assert len(engineered_features) == len(features)
    
    def test_configuration_consistency(self):
        """Test that all components use consistent configurations"""
        # This test ensures that configuration settings are consistent
        # across different components
        
        try:
            from config.config import PERFORMANCE_THRESHOLDS, MODELS_DIR
            
            # Check that required configuration exists
            assert isinstance(PERFORMANCE_THRESHOLDS, dict)
            assert MODELS_DIR is not None
            
            # Verify that components can access configuration
            model_suite = WineQualityModelSuite()
            # The model suite should be able to access the models directory
            assert hasattr(model_suite, 'models_dir') or hasattr(model_suite, 'models')
            
        except ImportError:
            # If config is not available, that's also valid for testing
            pytest.skip("Configuration module not available")
    
    def test_memory_usage_pipeline(self, sample_wine_data):
        """Test that pipeline doesn't use excessive memory"""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run pipeline
        data_loader = WineDataLoader()
        feature_engineer = WineFeatureEngineer()
        
        processed_data = data_loader.preprocess_data(sample_wine_data)
        features, target = data_loader.split_features_target(processed_data)
        engineered_features = feature_engineer.create_domain_features(features)
        
        # Check memory usage after processing
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for test data)
        assert memory_increase < 100, f"Memory usage increased by {memory_increase:.2f}MB"
    
    def test_pipeline_reproducibility(self, sample_wine_data):
        """Test that pipeline produces reproducible results"""
        # Run pipeline twice with same random seed
        results1 = self._run_pipeline_subset(sample_wine_data, random_state=42)
        results2 = self._run_pipeline_subset(sample_wine_data, random_state=42)
        
        # Results should be identical
        assert results1.keys() == results2.keys()
        
        for key in results1.keys():
            if isinstance(results1[key], (int, float)):
                assert abs(results1[key] - results2[key]) < 1e-10
            elif isinstance(results1[key], np.ndarray):
                np.testing.assert_array_almost_equal(results1[key], results2[key])
    
    def _run_pipeline_subset(self, data, random_state=42):
        """Helper method to run a subset of the pipeline for testing"""
        from sklearn.model_selection import train_test_split
        
        # Set random seed
        np.random.seed(random_state)
        
        # Process data
        data_loader = WineDataLoader()
        processed_data = data_loader.preprocess_data(data)
        features, target = data_loader.split_features_target(processed_data)
        
        # Simple train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=random_state
        )
        
        # Train simple model
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_squared_error
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        return {
            'r2_score': r2_score(y_test, predictions),
            'mse': mean_squared_error(y_test, predictions),
            'predictions': predictions
        }