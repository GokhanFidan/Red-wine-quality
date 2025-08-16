"""
Tests for feature engineering module
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

try:
    from features.feature_engineering import WineFeatureEngineer
except ImportError:
    pytest.skip("Feature engineering module not available", allow_module_level=True)

class TestWineFeatureEngineer:
    """Test cases for WineFeatureEngineer"""
    
    def test_engineer_initialization(self):
        """Test if feature engineer initializes properly"""
        engineer = WineFeatureEngineer()
        assert engineer is not None
        assert hasattr(engineer, 'engineered_features')
        assert isinstance(engineer.engineered_features, list)
    
    def test_domain_features_creation(self, sample_features):
        """Test domain-specific feature creation"""
        engineer = WineFeatureEngineer()
        
        engineered_df = engineer.create_domain_features(sample_features)
        
        assert isinstance(engineered_df, pd.DataFrame)
        assert len(engineered_df) == len(sample_features)
        
        # Should have more features than original
        assert len(engineered_df.columns) >= len(sample_features.columns)
        
        # Original features should still be present
        for col in sample_features.columns:
            assert col in engineered_df.columns
    
    def test_acidity_ratios(self, sample_features):
        """Test acidity ratio calculations"""
        engineer = WineFeatureEngineer()
        
        # Test individual ratio calculation
        ratios = engineer.calculate_acidity_ratios(sample_features)
        
        assert isinstance(ratios, pd.DataFrame)
        assert 'fixed_volatile_ratio' in ratios.columns
        assert 'total_acidity' in ratios.columns
        assert 'citric_fixed_ratio' in ratios.columns
        
        # Check that ratios are calculated correctly
        expected_ratio = sample_features['fixed_acidity'] / (sample_features['volatile_acidity'] + 1e-6)
        pd.testing.assert_series_equal(
            ratios['fixed_volatile_ratio'], 
            expected_ratio, 
            check_names=False
        )
    
    def test_sulfur_features(self, sample_features):
        """Test sulfur dioxide feature engineering"""
        engineer = WineFeatureEngineer()
        
        sulfur_features = engineer.calculate_sulfur_features(sample_features)
        
        assert isinstance(sulfur_features, pd.DataFrame)
        assert 'free_total_sulfur_ratio' in sulfur_features.columns
        assert 'bound_sulfur_dioxide' in sulfur_features.columns
        assert 'sulfur_efficiency' in sulfur_features.columns
        
        # Check bound sulfur calculation
        expected_bound = sample_features['total_sulfur_dioxide'] - sample_features['free_sulfur_dioxide']
        pd.testing.assert_series_equal(
            sulfur_features['bound_sulfur_dioxide'],
            expected_bound,
            check_names=False
        )
    
    def test_alcohol_sugar_interactions(self, sample_features):
        """Test alcohol-sugar interaction features"""
        engineer = WineFeatureEngineer()
        
        interactions = engineer.calculate_alcohol_sugar_interactions(sample_features)
        
        assert isinstance(interactions, pd.DataFrame)
        assert 'alcohol_sugar_ratio' in interactions.columns
        assert 'alcohol_sugar_product' in interactions.columns
        assert 'sweetness_index' in interactions.columns
        
        # Check alcohol-sugar ratio calculation
        expected_ratio = sample_features['alcohol'] / (sample_features['residual_sugar'] + 1e-6)
        pd.testing.assert_series_equal(
            interactions['alcohol_sugar_ratio'],
            expected_ratio,
            check_names=False
        )
    
    def test_chemical_balance_features(self, sample_features):
        """Test chemical balance feature calculations"""
        engineer = WineFeatureEngineer()
        
        balance_features = engineer.calculate_chemical_balance(sample_features)
        
        assert isinstance(balance_features, pd.DataFrame)
        assert 'acid_alcohol_balance' in balance_features.columns
        assert 'preservation_index' in balance_features.columns
        assert 'chemical_complexity' in balance_features.columns
    
    def test_quality_indicators(self, sample_features):
        """Test quality indicator calculations"""
        engineer = WineFeatureEngineer()
        
        quality_indicators = engineer.calculate_quality_indicators(sample_features)
        
        assert isinstance(quality_indicators, pd.DataFrame)
        assert 'taste_balance' in quality_indicators.columns
        assert 'structure_index' in quality_indicators.columns
        assert 'harmony_score' in quality_indicators.columns
    
    def test_feature_scaling(self, sample_features):
        """Test feature scaling functionality"""
        engineer = WineFeatureEngineer()
        
        # Test standard scaling
        scaled_train, scaled_test = engineer.scale_features(
            sample_features, sample_features, method='standard'
        )
        
        assert isinstance(scaled_train, pd.DataFrame)
        assert isinstance(scaled_test, pd.DataFrame)
        assert scaled_train.shape == sample_features.shape
        assert scaled_test.shape == sample_features.shape
        
        # Check that scaled data has mean ≈ 0 and std ≈ 1 (for train set)
        assert abs(scaled_train.mean().mean()) < 0.1
        assert abs(scaled_train.std().mean() - 1.0) < 0.1
    
    def test_robust_scaling(self, sample_features):
        """Test robust scaling"""
        engineer = WineFeatureEngineer()
        
        scaled_train, scaled_test = engineer.scale_features(
            sample_features, sample_features, method='robust'
        )
        
        assert isinstance(scaled_train, pd.DataFrame)
        assert isinstance(scaled_test, pd.DataFrame)
        assert scaled_train.shape == sample_features.shape
    
    def test_minmax_scaling(self, sample_features):
        """Test min-max scaling"""
        engineer = WineFeatureEngineer()
        
        scaled_train, scaled_test = engineer.scale_features(
            sample_features, sample_features, method='minmax'
        )
        
        assert isinstance(scaled_train, pd.DataFrame)
        assert isinstance(scaled_test, pd.DataFrame)
        
        # Check that values are between 0 and 1
        assert (scaled_train >= 0).all().all()
        assert (scaled_train <= 1).all().all()
    
    def test_feature_selection(self, sample_features, sample_wine_data):
        """Test feature selection functionality"""
        engineer = WineFeatureEngineer()
        
        # Create engineered features first
        engineered_features = engineer.create_domain_features(sample_features)
        target = sample_wine_data['quality'][:len(sample_features)]
        
        selected_train, selected_test = engineer.select_features(
            engineered_features, target, engineered_features, n_features=5
        )
        
        assert isinstance(selected_train, pd.DataFrame)
        assert isinstance(selected_test, pd.DataFrame)
        assert selected_train.shape[1] == 5
        assert selected_test.shape[1] == 5
        assert selected_train.shape[0] == engineered_features.shape[0]
    
    def test_feature_importance_analysis(self, sample_wine_data):
        """Test feature importance analysis"""
        engineer = WineFeatureEngineer()
        
        features = sample_wine_data.drop('quality', axis=1)
        target = sample_wine_data['quality']
        
        importance_analysis = engineer.get_feature_importance_analysis(features, target)
        
        assert isinstance(importance_analysis, dict)
        assert 'correlations' in importance_analysis
        assert 'mutual_info' in importance_analysis
        assert 'variance' in importance_analysis
        
        # Check correlations
        correlations = importance_analysis['correlations']
        assert isinstance(correlations, dict)
        assert len(correlations) == len(features.columns)
        
        # All correlation values should be between -1 and 1
        for corr_value in correlations.values():
            assert -1 <= corr_value <= 1
    
    def test_polynomial_features(self, sample_features):
        """Test polynomial feature generation"""
        engineer = WineFeatureEngineer()
        
        # Select a subset of features for polynomial expansion
        subset_features = sample_features[['alcohol', 'volatile_acidity', 'pH']]
        
        poly_features = engineer.create_polynomial_features(subset_features, degree=2)
        
        assert isinstance(poly_features, pd.DataFrame)
        assert poly_features.shape[1] > subset_features.shape[1]
        assert poly_features.shape[0] == subset_features.shape[0]
    
    def test_interaction_features(self, sample_features):
        """Test interaction feature creation"""
        engineer = WineFeatureEngineer()
        
        interaction_features = engineer.create_interaction_features(sample_features)
        
        assert isinstance(interaction_features, pd.DataFrame)
        assert interaction_features.shape[0] == sample_features.shape[0]
        assert interaction_features.shape[1] > 0
    
    def test_no_nan_in_engineered_features(self, sample_features):
        """Test that engineered features don't contain NaN values"""
        engineer = WineFeatureEngineer()
        
        engineered_df = engineer.create_domain_features(sample_features)
        
        # Check for NaN values
        nan_columns = engineered_df.columns[engineered_df.isnull().any()].tolist()
        assert len(nan_columns) == 0, f"NaN values found in columns: {nan_columns}"
    
    def test_feature_names_consistency(self, sample_features):
        """Test that feature names are consistent and meaningful"""
        engineer = WineFeatureEngineer()
        
        engineered_df = engineer.create_domain_features(sample_features)
        
        # Check that engineered feature names are tracked
        assert len(engineer.engineered_features) > 0
        
        # Check that all engineered features are in the dataframe
        for feature_name in engineer.engineered_features:
            assert feature_name in engineered_df.columns
    
    def test_feature_reproducibility(self, sample_features):
        """Test that feature engineering is reproducible"""
        engineer1 = WineFeatureEngineer()
        engineer2 = WineFeatureEngineer()
        
        engineered_df1 = engineer1.create_domain_features(sample_features)
        engineered_df2 = engineer2.create_domain_features(sample_features)
        
        # Results should be identical
        pd.testing.assert_frame_equal(engineered_df1, engineered_df2)
    
    @pytest.mark.parametrize("n_features", [5, 10, 15])
    def test_different_feature_selection_sizes(self, sample_features, sample_wine_data, n_features):
        """Test feature selection with different numbers of features"""
        engineer = WineFeatureEngineer()
        
        engineered_features = engineer.create_domain_features(sample_features)
        target = sample_wine_data['quality'][:len(sample_features)]
        
        # Only test if we have enough features
        if engineered_features.shape[1] >= n_features:
            selected_train, selected_test = engineer.select_features(
                engineered_features, target, engineered_features, n_features=n_features
            )
            
            assert selected_train.shape[1] == n_features
            assert selected_test.shape[1] == n_features