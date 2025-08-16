"""
Tests for machine learning models module
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch
from sklearn.model_selection import train_test_split

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

try:
    from models.wine_models import WineQualityModelSuite
except ImportError:
    pytest.skip("Models module not available", allow_module_level=True)

class TestWineQualityModelSuite:
    """Test cases for WineQualityModelSuite"""
    
    def test_model_suite_initialization(self):
        """Test if model suite initializes properly"""
        model_suite = WineQualityModelSuite()
        assert model_suite is not None
        assert hasattr(model_suite, 'models')
        assert hasattr(model_suite, 'results')
        assert hasattr(model_suite, 'best_model_name')
        assert hasattr(model_suite, 'best_model')
    
    def test_initialize_models(self):
        """Test model initialization"""
        model_suite = WineQualityModelSuite()
        model_suite.initialize_models()
        
        assert isinstance(model_suite.models, dict)
        assert len(model_suite.models) > 0
        
        # Check that common models are included
        expected_models = ['random_forest', 'xgboost', 'linear_regression', 'svm']
        
        for model_name in expected_models:
            if model_name in model_suite.models:
                assert model_suite.models[model_name] is not None
    
    def test_train_single_model(self, sample_wine_data):
        """Test training a single model"""
        model_suite = WineQualityModelSuite()
        model_suite.initialize_models()
        
        # Prepare data
        features = sample_wine_data.drop('quality', axis=1)
        target = sample_wine_data['quality']
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Train a single model (Random Forest)
        if 'random_forest' in model_suite.models:
            result = model_suite.train_single_model(
                'random_forest', X_train, y_train, X_test, y_test
            )
            
            assert isinstance(result, dict)
            assert 'train_score' in result
            assert 'test_score' in result
            assert 'train_r2' in result
            assert 'test_r2' in result
            assert 'training_time' in result
            
            # Check score ranges
            assert -1 <= result['train_r2'] <= 1
            assert -1 <= result['test_r2'] <= 1
            assert result['training_time'] > 0
    
    def test_model_evaluation_metrics(self, sample_wine_data):
        """Test model evaluation metrics calculation"""
        model_suite = WineQualityModelSuite()
        
        # Create dummy predictions
        y_true = sample_wine_data['quality'][:50]
        y_pred = y_true + np.random.normal(0, 0.5, len(y_true))  # Add some noise
        
        metrics = model_suite.calculate_metrics(y_true, y_pred)
        
        assert isinstance(metrics, dict)
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'accuracy' in metrics
        
        # Check that metrics are reasonable
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert -1 <= metrics['r2'] <= 1
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_hyperparameter_tuning(self, sample_wine_data):
        """Test hyperparameter tuning"""
        model_suite = WineQualityModelSuite()
        model_suite.initialize_models()
        
        features = sample_wine_data.drop('quality', axis=1)
        target = sample_wine_data['quality']
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Test hyperparameter tuning for Random Forest
        if 'random_forest' in model_suite.models:
            original_model = model_suite.models['random_forest']
            
            tuned_model = model_suite.tune_hyperparameters(
                'random_forest', X_train, y_train, cv_folds=3, max_iter=5
            )
            
            assert tuned_model is not None
            # The tuned model should have hyperparameters set
            assert hasattr(tuned_model, 'n_estimators')
    
    def test_cross_validation(self, sample_wine_data):
        """Test cross-validation functionality"""
        model_suite = WineQualityModelSuite()
        model_suite.initialize_models()
        
        features = sample_wine_data.drop('quality', axis=1)
        target = sample_wine_data['quality']
        
        if 'random_forest' in model_suite.models:
            cv_scores = model_suite.cross_validate_model(
                'random_forest', features, target, cv_folds=3
            )
            
            assert isinstance(cv_scores, dict)
            assert 'cv_scores' in cv_scores
            assert 'mean_cv_score' in cv_scores
            assert 'std_cv_score' in cv_scores
            
            assert isinstance(cv_scores['cv_scores'], np.ndarray)
            assert len(cv_scores['cv_scores']) == 3
            assert isinstance(cv_scores['mean_cv_score'], float)
            assert isinstance(cv_scores['std_cv_score'], float)
    
    def test_feature_importance_extraction(self, sample_wine_data):
        """Test feature importance extraction"""
        model_suite = WineQualityModelSuite()
        model_suite.initialize_models()
        
        features = sample_wine_data.drop('quality', axis=1)
        target = sample_wine_data['quality']
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Train Random Forest and get feature importance
        if 'random_forest' in model_suite.models:
            model_suite.train_single_model('random_forest', X_train, y_train, X_test, y_test)
            
            feature_importance = model_suite.get_feature_importance('random_forest')
            
            if feature_importance is not None:
                assert isinstance(feature_importance, dict)
                assert len(feature_importance) == len(features.columns)
                
                # Check that importances sum to approximately 1
                total_importance = sum(feature_importance.values())
                assert abs(total_importance - 1.0) < 0.01
    
    def test_model_comparison(self, sample_wine_data):
        """Test model comparison functionality"""
        model_suite = WineQualityModelSuite()
        model_suite.initialize_models()
        
        features = sample_wine_data.drop('quality', axis=1)
        target = sample_wine_data['quality']
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Train a few models
        models_to_train = ['linear_regression']
        if 'random_forest' in model_suite.models:
            models_to_train.append('random_forest')
        
        for model_name in models_to_train:
            model_suite.train_single_model(model_name, X_train, y_train, X_test, y_test)
        
        # Get comparison
        comparison_df = model_suite.get_model_comparison()
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == len(models_to_train)
        assert 'model_name' in comparison_df.columns
        assert 'test_r2' in comparison_df.columns
        assert 'test_rmse' in comparison_df.columns
    
    def test_best_model_selection(self, sample_wine_data):
        """Test best model selection"""
        model_suite = WineQualityModelSuite()
        model_suite.initialize_models()
        
        features = sample_wine_data.drop('quality', axis=1)
        target = sample_wine_data['quality']
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Train models
        model_suite.train_all_models(X_train, y_train, X_test, y_test)
        
        # Check best model selection
        assert model_suite.best_model_name is not None
        assert model_suite.best_model is not None
        assert model_suite.best_model_name in model_suite.models
    
    def test_ensemble_methods(self, sample_wine_data):
        """Test ensemble model creation"""
        model_suite = WineQualityModelSuite()
        model_suite.initialize_models()
        
        features = sample_wine_data.drop('quality', axis=1)
        target = sample_wine_data['quality']
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Train base models first
        model_suite.train_all_models(X_train, y_train, X_test, y_test)
        
        # Create ensemble
        ensemble_results = model_suite.create_ensemble_models(X_train, y_train, X_test, y_test)
        
        if ensemble_results:
            assert isinstance(ensemble_results, dict)
            
            for ensemble_name, result in ensemble_results.items():
                assert isinstance(result, dict)
                assert 'test_r2' in result
                assert 'test_rmse' in result
    
    def test_model_saving_and_loading(self, sample_wine_data, tmp_path):
        """Test model saving and loading"""
        model_suite = WineQualityModelSuite()
        model_suite.initialize_models()
        
        # Override the models directory for testing
        model_suite.models_dir = tmp_path
        
        features = sample_wine_data.drop('quality', axis=1)
        target = sample_wine_data['quality']
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Train and save a model
        if 'linear_regression' in model_suite.models:
            model_suite.train_single_model('linear_regression', X_train, y_train, X_test, y_test)
            model_suite.best_model_name = 'linear_regression'
            model_suite.best_model = model_suite.models['linear_regression']
            
            # Save the model
            saved_path = model_suite.save_best_model()
            
            assert saved_path is not None
            assert saved_path.exists()
            
            # Try to load the model
            loaded_model_data = model_suite.load_model(saved_path)
            assert loaded_model_data is not None
            assert 'model' in loaded_model_data
    
    def test_prediction_methods(self, sample_wine_data):
        """Test prediction methods"""
        model_suite = WineQualityModelSuite()
        model_suite.initialize_models()
        
        features = sample_wine_data.drop('quality', axis=1)
        target = sample_wine_data['quality']
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Train a model
        if 'linear_regression' in model_suite.models:
            model_suite.train_single_model('linear_regression', X_train, y_train, X_test, y_test)
            
            # Test predictions
            predictions = model_suite.predict('linear_regression', X_test)
            
            assert isinstance(predictions, np.ndarray)
            assert len(predictions) == len(X_test)
            assert all(isinstance(pred, (int, float, np.number)) for pred in predictions)
    
    def test_model_performance_thresholds(self, sample_wine_data):
        """Test model performance against thresholds"""
        model_suite = WineQualityModelSuite()
        model_suite.initialize_models()
        
        features = sample_wine_data.drop('quality', axis=1)
        target = sample_wine_data['quality']
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Train models
        results = model_suite.train_all_models(X_train, y_train, X_test, y_test)
        
        # Check performance thresholds
        performance_check = model_suite.check_performance_thresholds()
        
        assert isinstance(performance_check, dict)
        assert 'meets_threshold' in performance_check
        assert 'details' in performance_check
        
        # The check should return boolean
        assert isinstance(performance_check['meets_threshold'], bool)
    
    @pytest.mark.parametrize("test_size", [0.1, 0.2, 0.3])
    def test_different_data_splits(self, sample_wine_data, test_size):
        """Test model training with different data split sizes"""
        model_suite = WineQualityModelSuite()
        model_suite.initialize_models()
        
        features = sample_wine_data.drop('quality', axis=1)
        target = sample_wine_data['quality']
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=42
        )
        
        # Train linear regression (fastest model)
        if 'linear_regression' in model_suite.models:
            result = model_suite.train_single_model(
                'linear_regression', X_train, y_train, X_test, y_test
            )
            
            assert result is not None
            assert 'test_r2' in result
            
            # Check that we have reasonable split sizes
            expected_test_size = int(len(features) * test_size)
            assert abs(len(X_test) - expected_test_size) <= 1
    
    def test_model_robustness_with_noise(self, sample_wine_data):
        """Test model robustness with noisy data"""
        model_suite = WineQualityModelSuite()
        model_suite.initialize_models()
        
        features = sample_wine_data.drop('quality', axis=1)
        target = sample_wine_data['quality']
        
        # Add noise to features
        noisy_features = features + np.random.normal(0, 0.1, features.shape)
        
        X_train, X_test, y_train, y_test = train_test_split(
            noisy_features, target, test_size=0.2, random_state=42
        )
        
        # Train with noisy data
        if 'linear_regression' in model_suite.models:
            result = model_suite.train_single_model(
                'linear_regression', X_train, y_train, X_test, y_test
            )
            
            assert result is not None
            # Model should still provide reasonable results
            assert result['test_r2'] > -1  # Not completely broken