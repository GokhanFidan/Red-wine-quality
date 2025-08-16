"""
Wine Quality Data Loading and Validation
Professional data loading with comprehensive validation
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional
import warnings

from config.config import (
    WINE_DATA_FILE, FEATURE_COLUMNS, TARGET_COLUMN, 
    DATA_VALIDATION_RULES, PROCESSED_DATA_DIR
)

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class WineDataLoader:
    """
    Professional data loader for wine quality dataset with validation and preprocessing
    """
    
    def __init__(self):
        self.data = None
        self.features = None
        self.target = None
        self.validation_report = {}
        
    def load_raw_data(self, filepath: Optional[Path] = None) -> pd.DataFrame:
        """
        Load raw wine quality data with error handling
        
        Args:
            filepath: Optional path to data file
            
        Returns:
            DataFrame with raw wine data
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            pd.errors.EmptyDataError: If file is empty
        """
        try:
            filepath = filepath or WINE_DATA_FILE
            logger.info(f"Loading wine data from {filepath}")
            
            if not filepath.exists():
                raise FileNotFoundError(f"Data file not found: {filepath}")
            
            # Load data with proper error handling
            data = pd.read_csv(filepath)
            
            if data.empty:
                raise pd.errors.EmptyDataError("Data file is empty")
                
            logger.info(f"Successfully loaded {len(data)} records with {len(data.columns)} columns")
            
            # Basic data info
            logger.info(f"Data shape: {data.shape}")
            logger.info(f"Columns: {list(data.columns)}")
            
            self.data = data
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def validate_data(self, data: pd.DataFrame) -> Dict:
        """
        Comprehensive data validation
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_report = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'summary': {}
        }
        
        try:
            # Check for required columns
            missing_columns = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] 
                             if col not in data.columns]
            
            if missing_columns:
                validation_report['is_valid'] = False
                validation_report['issues'].append(f"Missing columns: {missing_columns}")
            
            # Check data types
            numeric_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
            non_numeric = []
            
            for col in numeric_columns:
                if col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        non_numeric.append(col)
                        
            if non_numeric:
                validation_report['warnings'].append(f"Non-numeric columns: {non_numeric}")
            
            # Check for missing values
            missing_values = data.isnull().sum()
            if missing_values.sum() > 0:
                validation_report['warnings'].append(f"Missing values found: {missing_values[missing_values > 0].to_dict()}")
            
            # Check data ranges
            range_violations = []
            for col, rules in DATA_VALIDATION_RULES.items():
                if col in data.columns:
                    min_val, max_val = rules['min'], rules['max']
                    violations = ((data[col] < min_val) | (data[col] > max_val)).sum()
                    if violations > 0:
                        range_violations.append(f"{col}: {violations} values outside [{min_val}, {max_val}]")
            
            if range_violations:
                validation_report['warnings'].append(f"Range violations: {range_violations}")
            
            # Check for duplicates
            duplicates = data.duplicated().sum()
            if duplicates > 0:
                validation_report['warnings'].append(f"Duplicate rows: {duplicates}")
            
            # Summary statistics
            validation_report['summary'] = {
                'total_rows': len(data),
                'total_columns': len(data.columns),
                'missing_values_total': missing_values.sum(),
                'duplicate_rows': duplicates,
                'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024**2
            }
            
            self.validation_report = validation_report
            logger.info("Data validation completed")
            
            return validation_report
            
        except Exception as e:
            logger.error(f"Error in data validation: {str(e)}")
            validation_report['is_valid'] = False
            validation_report['issues'].append(f"Validation error: {str(e)}")
            return validation_report
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data
        
        Args:
            data: Raw data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        try:
            logger.info("Starting data cleaning process")
            cleaned_data = data.copy()
            
            # Remove duplicates
            initial_rows = len(cleaned_data)
            cleaned_data = cleaned_data.drop_duplicates()
            removed_duplicates = initial_rows - len(cleaned_data)
            
            if removed_duplicates > 0:
                logger.info(f"Removed {removed_duplicates} duplicate rows")
            
            # Handle missing values
            missing_before = cleaned_data.isnull().sum().sum()
            if missing_before > 0:
                # For numerical columns, use median imputation
                numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if cleaned_data[col].isnull().any():
                        median_val = cleaned_data[col].median()
                        cleaned_data[col].fillna(median_val, inplace=True)
                        logger.info(f"Filled {col} missing values with median: {median_val:.2f}")
            
            # Handle outliers (using IQR method)
            outliers_removed = 0
            for col in FEATURE_COLUMNS:
                if col in cleaned_data.columns:
                    Q1 = cleaned_data[col].quantile(0.25)
                    Q3 = cleaned_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers_mask = (cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound)
                    outliers_count = outliers_mask.sum()
                    
                    if outliers_count > 0:
                        # Cap outliers instead of removing
                        cleaned_data.loc[cleaned_data[col] < lower_bound, col] = lower_bound
                        cleaned_data.loc[cleaned_data[col] > upper_bound, col] = upper_bound
                        outliers_removed += outliers_count
            
            if outliers_removed > 0:
                logger.info(f"Capped {outliers_removed} outlier values")
            
            # Ensure target variable is integer
            if TARGET_COLUMN in cleaned_data.columns:
                cleaned_data[TARGET_COLUMN] = cleaned_data[TARGET_COLUMN].astype(int)
            
            logger.info(f"Data cleaning completed. Final shape: {cleaned_data.shape}")
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error in data cleaning: {str(e)}")
            raise
    
    def split_features_target(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split data into features and target
        
        Args:
            data: Cleaned DataFrame
            
        Returns:
            Tuple of (features, target)
        """
        try:
            features = data[FEATURE_COLUMNS].copy()
            target = data[TARGET_COLUMN].copy()
            
            logger.info(f"Features shape: {features.shape}")
            logger.info(f"Target shape: {target.shape}")
            logger.info(f"Target distribution:\n{target.value_counts().sort_index()}")
            
            self.features = features
            self.target = target
            
            return features, target
            
        except Exception as e:
            logger.error(f"Error splitting features and target: {str(e)}")
            raise
    
    def save_processed_data(self, data: pd.DataFrame, filename: str = 'wine_processed.csv'):
        """
        Save processed data to file
        
        Args:
            data: Processed DataFrame
            filename: Output filename
        """
        try:
            output_path = PROCESSED_DATA_DIR / filename
            data.to_csv(output_path, index=False)
            logger.info(f"Processed data saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete data loading and preparation pipeline
        
        Returns:
            Tuple of (features, target)
        """
        try:
            # Load raw data
            raw_data = self.load_raw_data()
            
            # Validate data
            validation_result = self.validate_data(raw_data)
            
            if not validation_result['is_valid']:
                raise ValueError(f"Data validation failed: {validation_result['issues']}")
            
            # Clean data
            cleaned_data = self.clean_data(raw_data)
            
            # Save processed data
            self.save_processed_data(cleaned_data)
            
            # Split features and target
            features, target = self.split_features_target(cleaned_data)
            
            logger.info("Data preparation pipeline completed successfully")
            
            return features, target
            
        except Exception as e:
            logger.error(f"Error in data preparation pipeline: {str(e)}")
            raise

def get_wine_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function to get preprocessed wine data
    
    Returns:
        Tuple of (features, target)
    """
    loader = WineDataLoader()
    return loader.load_and_prepare_data()