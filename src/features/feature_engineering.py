"""
Advanced Feature Engineering for Wine Quality Prediction
Domain-specific features based on wine chemistry knowledge
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from typing import Tuple, List, Dict, Optional

logger = logging.getLogger(__name__)

class WineFeatureEngineer:
    """
    Advanced feature engineering for wine quality prediction
    """
    
    def __init__(self):
        self.scaler = None
        self.feature_selector = None
        self.feature_names = None
        self.engineered_features = []
        
    def create_domain_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific features based on wine chemistry knowledge
        
        Args:
            data: Input DataFrame with wine features
            
        Returns:
            DataFrame with additional engineered features
        """
        try:
            logger.info("Creating domain-specific wine features")
            df = data.copy()
            
            # Acidity-related features
            df['total_acidity'] = df['fixed acidity'] + df['volatile acidity']
            df['acidity_ratio'] = df['fixed acidity'] / (df['volatile acidity'] + 1e-8)
            df['citric_acid_ratio'] = df['citric acid'] / df['total_acidity']
            
            # Sulfur dioxide features
            df['bound_sulfur_dioxide'] = df['total sulfur dioxide'] - df['free sulfur dioxide']
            df['sulfur_ratio'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 1e-8)
            df['sulfur_density'] = df['total sulfur dioxide'] / df['density']
            
            # Alcohol and density relationship
            df['alcohol_density_interaction'] = df['alcohol'] * df['density']
            df['alcohol_sugar_ratio'] = df['alcohol'] / (df['residual sugar'] + 1e-8)
            
            # Chemical balance indicators
            df['pH_acidity_balance'] = df['pH'] * df['total_acidity']
            df['salt_acid_ratio'] = df['chlorides'] / df['total_acidity']
            df['sulphates_alcohol_ratio'] = df['sulphates'] / df['alcohol']
            
            # Quality indicators based on chemistry
            df['preservation_index'] = (df['total sulfur dioxide'] + df['citric acid']) / 2
            df['taste_balance'] = df['residual sugar'] - df['volatile acidity']
            df['chemical_complexity'] = (df['citric acid'] + df['sulphates'] + df['alcohol']) / 3
            
            # Interaction features
            df['alcohol_sulphates'] = df['alcohol'] * df['sulphates']
            df['density_alcohol'] = df['density'] / df['alcohol']
            df['volatile_citric'] = df['volatile acidity'] * df['citric acid']
            
            # Logarithmic transformations for skewed features
            df['log_volatile_acidity'] = np.log1p(df['volatile acidity'])
            df['log_residual_sugar'] = np.log1p(df['residual sugar'])
            df['log_chlorides'] = np.log1p(df['chlorides'])
            
            # Polynomial features for important variables
            df['alcohol_squared'] = df['alcohol'] ** 2
            df['sulphates_squared'] = df['sulphates'] ** 2
            df['volatile_acidity_squared'] = df['volatile acidity'] ** 2
            
            # Binned features
            df['alcohol_category'] = pd.cut(df['alcohol'], bins=3, labels=['Low', 'Medium', 'High'])
            df['alcohol_category'] = df['alcohol_category'].cat.codes
            
            df['acidity_category'] = pd.cut(df['total_acidity'], bins=3, labels=['Low', 'Medium', 'High'])
            df['acidity_category'] = df['acidity_category'].cat.codes
            
            # Store engineered feature names
            original_features = data.columns.tolist()
            self.engineered_features = [col for col in df.columns if col not in original_features]
            
            logger.info(f"Created {len(self.engineered_features)} engineered features")
            logger.info(f"Engineered features: {self.engineered_features}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      method: str = 'standard') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale features using specified method
        
        Args:
            X_train: Training features
            X_test: Test features
            method: Scaling method ('standard', 'robust', 'minmax')
            
        Returns:
            Tuple of scaled (X_train, X_test)
        """
        try:
            logger.info(f"Scaling features using {method} method")
            
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'robust':
                self.scaler = RobustScaler()
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            # Fit on training data and transform both
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            
            logger.info("Feature scaling completed")
            
            return X_train_scaled, X_test_scaled
            
        except Exception as e:
            logger.error(f"Error in feature scaling: {str(e)}")
            raise
    
    def select_features(self, X_train: pd.DataFrame, y_train: pd.Series, 
                       X_test: pd.DataFrame, n_features: int = 20, 
                       method: str = 'f_regression') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Select best features using statistical methods
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            n_features: Number of features to select
            method: Selection method ('f_regression', 'mutual_info')
            
        Returns:
            Tuple of selected (X_train, X_test)
        """
        try:
            logger.info(f"Selecting {n_features} best features using {method}")
            
            if method == 'f_regression':
                selector = SelectKBest(score_func=f_regression, k=n_features)
            elif method == 'mutual_info':
                selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
            else:
                raise ValueError(f"Unknown selection method: {method}")
            
            # Fit on training data
            X_train_selected = pd.DataFrame(
                selector.fit_transform(X_train, y_train),
                index=X_train.index
            )
            
            X_test_selected = pd.DataFrame(
                selector.transform(X_test),
                index=X_test.index
            )
            
            # Get selected feature names
            selected_features = X_train.columns[selector.get_support()].tolist()
            X_train_selected.columns = selected_features
            X_test_selected.columns = selected_features
            
            # Get feature scores
            scores = selector.scores_[selector.get_support()]
            feature_scores = dict(zip(selected_features, scores))
            
            logger.info("Selected features with scores:")
            for feature, score in sorted(feature_scores.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {feature}: {score:.4f}")
            
            self.feature_selector = selector
            self.feature_names = selected_features
            
            return X_train_selected, X_test_selected
            
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            raise
    
    def get_feature_importance_analysis(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Analyze feature importance using multiple methods
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Dictionary with importance scores from different methods
        """
        try:
            logger.info("Performing feature importance analysis")
            
            # F-regression scores
            f_selector = SelectKBest(score_func=f_regression, k='all')
            f_selector.fit(X, y)
            f_scores = dict(zip(X.columns, f_selector.scores_))
            
            # Mutual information scores
            mi_selector = SelectKBest(score_func=mutual_info_regression, k='all')
            mi_selector.fit(X, y)
            mi_scores = dict(zip(X.columns, mi_selector.scores_))
            
            # Correlation with target
            correlations = X.corrwith(y).abs().to_dict()
            
            importance_analysis = {
                'f_regression_scores': f_scores,
                'mutual_info_scores': mi_scores,
                'correlations': correlations
            }
            
            logger.info("Feature importance analysis completed")
            
            return importance_analysis
            
        except Exception as e:
            logger.error(f"Error in feature importance analysis: {str(e)}")
            raise
    
    def create_interaction_features(self, data: pd.DataFrame, 
                                  top_features: List[str]) -> pd.DataFrame:
        """
        Create interaction features between top performing features
        
        Args:
            data: Input DataFrame
            top_features: List of top performing feature names
            
        Returns:
            DataFrame with interaction features
        """
        try:
            logger.info(f"Creating interaction features for top {len(top_features)} features")
            df = data.copy()
            
            interactions_created = 0
            for i, feat1 in enumerate(top_features):
                for feat2 in top_features[i+1:]:
                    if feat1 in df.columns and feat2 in df.columns:
                        # Multiplicative interaction
                        interaction_name = f"{feat1}_x_{feat2}"
                        df[interaction_name] = df[feat1] * df[feat2]
                        interactions_created += 1
            
            logger.info(f"Created {interactions_created} interaction features")
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating interaction features: {str(e)}")
            raise
    
    def transform_new_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessors
        
        Args:
            data: New data to transform
            
        Returns:
            Transformed DataFrame
        """
        try:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call scale_features first.")
            
            # Apply feature engineering
            data_engineered = self.create_domain_features(data)
            
            # Apply scaling
            data_scaled = pd.DataFrame(
                self.scaler.transform(data_engineered),
                columns=data_engineered.columns,
                index=data_engineered.index
            )
            
            # Apply feature selection if fitted
            if self.feature_selector is not None and self.feature_names is not None:
                # Ensure we have the same features that were selected during training
                missing_features = [f for f in self.feature_names if f not in data_scaled.columns]
                if missing_features:
                    logger.warning(f"Missing features in new data: {missing_features}")
                
                available_features = [f for f in self.feature_names if f in data_scaled.columns]
                data_scaled = data_scaled[available_features]
            
            return data_scaled
            
        except Exception as e:
            logger.error(f"Error transforming new data: {str(e)}")
            raise