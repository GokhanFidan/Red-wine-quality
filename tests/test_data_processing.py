"""
Tests for data processing module
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
    from data_processing.data_loader import WineDataLoader
except ImportError:
    pytest.skip("Data processing module not available", allow_module_level=True)

class TestWineDataLoader:
    """Test cases for WineDataLoader"""
    
    def test_loader_initialization(self):
        """Test if data loader initializes properly"""
        loader = WineDataLoader()
        assert loader is not None
        assert hasattr(loader, 'data_path')
        assert hasattr(loader, 'processed_data_path')
    
    def test_data_validation(self, sample_wine_data):
        """Test data validation functionality"""
        loader = WineDataLoader()
        
        # Test with valid data
        is_valid, report = loader.validate_data(sample_wine_data)
        assert isinstance(is_valid, bool)
        assert isinstance(report, dict)
        assert 'missing_values' in report
        assert 'duplicates' in report
        assert 'outliers' in report
    
    def test_data_validation_with_missing_values(self):
        """Test data validation with missing values"""
        loader = WineDataLoader()
        
        # Create data with missing values
        data = pd.DataFrame({
            'fixed_acidity': [7.4, np.nan, 7.8],
            'volatile_acidity': [0.7, 0.88, np.nan],
            'quality': [5, 6, 5]
        })
        
        is_valid, report = loader.validate_data(data)
        assert not is_valid or len(report['missing_values']) > 0
    
    def test_preprocessing_pipeline(self, sample_wine_data):
        """Test the preprocessing pipeline"""
        loader = WineDataLoader()
        
        # Test preprocessing
        processed_data = loader.preprocess_data(sample_wine_data)
        
        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) <= len(sample_wine_data)  # May have fewer rows after cleaning
        
        # Check that there are no missing values in processed data
        assert not processed_data.isnull().any().any()
    
    def test_data_splitting(self, sample_wine_data):
        """Test data splitting functionality"""
        loader = WineDataLoader()
        
        features, target = loader.split_features_target(sample_wine_data)
        
        assert isinstance(features, pd.DataFrame)
        assert isinstance(target, pd.Series)
        assert len(features) == len(target)
        assert 'quality' not in features.columns
        assert target.name == 'quality'
    
    def test_feature_columns_consistency(self, sample_wine_data):
        """Test that feature columns are consistent"""
        loader = WineDataLoader()
        
        features, _ = loader.split_features_target(sample_wine_data)
        
        expected_features = [
            'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
            'density', 'pH', 'sulphates', 'alcohol'
        ]
        
        for feature in expected_features:
            assert feature in features.columns
    
    def test_data_types(self, sample_wine_data):
        """Test that data types are correct"""
        loader = WineDataLoader()
        processed_data = loader.preprocess_data(sample_wine_data)
        
        # All columns should be numeric
        for column in processed_data.columns:
            assert pd.api.types.is_numeric_dtype(processed_data[column])
    
    def test_outlier_detection(self, sample_wine_data):
        """Test outlier detection"""
        loader = WineDataLoader()
        
        # Add some obvious outliers
        outlier_data = sample_wine_data.copy()
        outlier_data.loc[0, 'alcohol'] = 50.0  # Impossible alcohol content
        outlier_data.loc[1, 'pH'] = 10.0  # Impossible pH for wine
        
        outliers = loader.detect_outliers(outlier_data)
        
        assert isinstance(outliers, dict)
        assert len(outliers) > 0  # Should detect the outliers we added
    
    def test_quality_distribution(self, sample_wine_data):
        """Test quality distribution analysis"""
        loader = WineDataLoader()
        
        distribution = loader.get_quality_distribution(sample_wine_data['quality'])
        
        assert isinstance(distribution, dict)
        assert all(isinstance(k, (int, np.integer)) for k in distribution.keys())
        assert all(isinstance(v, (int, np.integer)) for v in distribution.values())
        assert sum(distribution.values()) == len(sample_wine_data)
    
    def test_save_and_load_processed_data(self, sample_wine_data, tmp_path):
        """Test saving and loading processed data"""
        loader = WineDataLoader()
        loader.processed_data_path = tmp_path / "test_processed.csv"
        
        # Process and save data
        processed_data = loader.preprocess_data(sample_wine_data)
        loader.save_processed_data(processed_data)
        
        # Check file exists
        assert loader.processed_data_path.exists()
        
        # Load and verify
        loaded_data = loader.load_processed_data()
        pd.testing.assert_frame_equal(processed_data, loaded_data)
    
    def test_data_summary_statistics(self, sample_wine_data):
        """Test data summary statistics"""
        loader = WineDataLoader()
        
        summary = loader.get_data_summary(sample_wine_data)
        
        assert isinstance(summary, dict)
        assert 'shape' in summary
        assert 'missing_values' in summary
        assert 'numeric_columns' in summary
        assert 'quality_range' in summary
        
        assert summary['shape'] == sample_wine_data.shape
        assert isinstance(summary['missing_values'], int)
        assert isinstance(summary['numeric_columns'], int)
    
    @pytest.mark.parametrize("test_size", [0.1, 0.2, 0.3])
    def test_train_test_split_sizes(self, sample_wine_data, test_size):
        """Test different train-test split sizes"""
        loader = WineDataLoader()
        
        features, target = loader.split_features_target(sample_wine_data)
        train_features, test_features, train_target, test_target = loader.create_train_test_split(
            features, target, test_size=test_size
        )
        
        expected_test_size = int(len(features) * test_size)
        assert abs(len(test_features) - expected_test_size) <= 1  # Allow for rounding
        assert len(train_features) + len(test_features) == len(features)
        assert len(train_target) == len(train_features)
        assert len(test_target) == len(test_features)