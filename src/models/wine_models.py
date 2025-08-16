"""
Advanced Machine Learning Models for Wine Quality Prediction
Enterprise-grade model training, evaluation, and comparison
"""

import pandas as pd
import numpy as np
import logging
import joblib
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, 
    cross_val_score, StratifiedKFold, KFold
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

# SHAP for model interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

from config.config import (
    MODEL_CONFIGS, RANDOM_STATE, CV_FOLDS, MODELS_DIR,
    PERFORMANCE_THRESHOLDS, QUALITY_CATEGORIES
)

logger = logging.getLogger(__name__)

class WineQualityModelSuite:
    """
    Comprehensive machine learning model suite for wine quality prediction
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.model_results = {}
        self.feature_importance = {}
        self.shap_explainer = None
        
    def initialize_models(self) -> Dict:
        """
        Initialize all machine learning models
        
        Returns:
            Dictionary of initialized models
        """
        try:
            logger.info("Initializing machine learning models")
            
            models = {
                # Tree-based models
                'random_forest': RandomForestRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                ),
                'decision_tree': DecisionTreeRegressor(
                    random_state=RANDOM_STATE
                ),
                
                # Linear models
                'linear_regression': LinearRegression(),
                'ridge': Ridge(random_state=RANDOM_STATE),
                'lasso': Lasso(random_state=RANDOM_STATE),
                'elastic_net': ElasticNet(random_state=RANDOM_STATE),
                
                # Support Vector Machine
                'svm': SVR(),
                
                # Neural Network
                'neural_network': MLPRegressor(
                    random_state=RANDOM_STATE,
                    max_iter=1000
                ),
                
                # K-Nearest Neighbors
                'knn': KNeighborsRegressor()
            }
            
            # Add XGBoost if available
            if XGBOOST_AVAILABLE:
                models['xgboost'] = xgb.XGBRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            
            self.models = models
            logger.info(f"Initialized {len(models)} models")
            
            return models
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
    
    def train_single_model(self, model_name: str, model: Any, 
                          X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series,
                          hyperparameter_tuning: bool = True) -> Dict:
        """
        Train and evaluate a single model
        
        Args:
            model_name: Name of the model
            model: Model instance
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with model results
        """
        try:
            logger.info(f"Training {model_name} model")
            start_time = time.time()
            
            # Hyperparameter tuning if requested and config available
            if hyperparameter_tuning and model_name in MODEL_CONFIGS:
                logger.info(f"Performing hyperparameter tuning for {model_name}")
                
                param_grid = MODEL_CONFIGS[model_name]
                cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                
                # Use RandomizedSearchCV for faster tuning
                grid_search = RandomizedSearchCV(
                    model, param_grid, cv=cv, scoring='neg_mean_squared_error',
                    n_iter=50, random_state=RANDOM_STATE, n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                
                logger.info(f"Best parameters for {model_name}: {best_params}")
            else:
                # Train with default parameters
                best_model = model
                best_model.fit(X_train, y_train)
                best_params = best_model.get_params()
            
            training_time = time.time() - start_time
            
            # Make predictions
            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)
            
            # Calculate regression metrics
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                best_model, X_train, y_train, cv=CV_FOLDS, 
                scoring='neg_mean_squared_error'
            )
            cv_rmse = np.sqrt(-cv_scores)
            
            # Convert to classification for additional metrics
            y_train_class = np.round(y_train).astype(int)
            y_test_class = np.round(y_test).astype(int)
            y_pred_train_class = np.round(y_pred_train).astype(int)
            y_pred_test_class = np.round(y_pred_test).astype(int)
            
            # Ensure predictions are within valid range
            y_pred_train_class = np.clip(y_pred_train_class, 3, 9)
            y_pred_test_class = np.clip(y_pred_test_class, 3, 9)
            
            # Classification metrics
            train_accuracy = accuracy_score(y_train_class, y_pred_train_class)
            test_accuracy = accuracy_score(y_test_class, y_pred_test_class)
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = dict(zip(X_train.columns, best_model.feature_importances_))
            elif hasattr(best_model, 'coef_'):
                feature_importance = dict(zip(X_train.columns, np.abs(best_model.coef_)))
            
            # Compile results
            results = {
                'model': best_model,
                'model_name': model_name,
                'best_params': best_params,
                'training_time': training_time,
                
                # Regression metrics
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_rmse': np.sqrt(train_mse),
                'test_rmse': np.sqrt(test_mse),
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                
                # Cross-validation
                'cv_rmse_mean': cv_rmse.mean(),
                'cv_rmse_std': cv_rmse.std(),
                
                # Classification metrics
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                
                # Predictions
                'y_pred_train': y_pred_train,
                'y_pred_test': y_pred_test,
                
                # Feature importance
                'feature_importance': feature_importance
            }
            
            logger.info(f"{model_name} training completed - Test R²: {test_r2:.4f}, Test RMSE: {np.sqrt(test_mse):.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            raise
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series,
                        hyperparameter_tuning: bool = True) -> Dict:
        """
        Train all models and compare performance
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with all model results
        """
        try:
            logger.info("Training all models")
            
            if not self.models:
                self.initialize_models()
            
            all_results = {}
            
            for model_name, model in self.models.items():
                try:
                    results = self.train_single_model(
                        model_name, model, X_train, y_train, 
                        X_test, y_test, hyperparameter_tuning
                    )
                    all_results[model_name] = results
                    
                except Exception as e:
                    logger.error(f"Failed to train {model_name}: {str(e)}")
                    continue
            
            self.model_results = all_results
            
            # Find best model based on test R²
            best_model_name = max(all_results.keys(), 
                                key=lambda x: all_results[x]['test_r2'])
            
            self.best_model = all_results[best_model_name]['model']
            self.best_model_name = best_model_name
            
            logger.info(f"Best model: {best_model_name} with R² = {all_results[best_model_name]['test_r2']:.4f}")
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error training all models: {str(e)}")
            raise
    
    def create_ensemble_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                              X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Create ensemble models using top performers
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with ensemble model results
        """
        try:
            logger.info("Creating ensemble models")
            
            if not self.model_results:
                raise ValueError("No trained models available. Train individual models first.")
            
            # Get top 3 models based on test R²
            top_models = sorted(self.model_results.items(), 
                              key=lambda x: x[1]['test_r2'], reverse=True)[:3]
            
            top_model_names = [name for name, _ in top_models]
            logger.info(f"Top models for ensemble: {top_model_names}")
            
            # Prepare estimators for ensemble
            estimators = [(name, results['model']) for name, results in top_models]
            
            ensemble_results = {}
            
            # Voting Regressor
            try:
                voting_regressor = VotingRegressor(estimators=estimators)
                voting_results = self.train_single_model(
                    'voting_ensemble', voting_regressor,
                    X_train, y_train, X_test, y_test, 
                    hyperparameter_tuning=False
                )
                ensemble_results['voting_ensemble'] = voting_results
                
            except Exception as e:
                logger.error(f"Error creating voting ensemble: {str(e)}")
            
            # Stacking Regressor
            try:
                stacking_regressor = StackingRegressor(
                    estimators=estimators,
                    final_estimator=LinearRegression(),
                    cv=3
                )
                stacking_results = self.train_single_model(
                    'stacking_ensemble', stacking_regressor,
                    X_train, y_train, X_test, y_test,
                    hyperparameter_tuning=False
                )
                ensemble_results['stacking_ensemble'] = stacking_results
                
            except Exception as e:
                logger.error(f"Error creating stacking ensemble: {str(e)}")
            
            # Update model results with ensemble models
            self.model_results.update(ensemble_results)
            
            # Check if ensemble model is now the best
            all_models_with_ensemble = {**self.model_results}
            new_best_model_name = max(all_models_with_ensemble.keys(),
                                    key=lambda x: all_models_with_ensemble[x]['test_r2'])
            
            if new_best_model_name in ensemble_results:
                self.best_model = ensemble_results[new_best_model_name]['model']
                self.best_model_name = new_best_model_name
                logger.info(f"New best model: {new_best_model_name} (ensemble)")
            
            return ensemble_results
            
        except Exception as e:
            logger.error(f"Error creating ensemble models: {str(e)}")
            raise
    
    def get_model_comparison(self) -> pd.DataFrame:
        """
        Get comprehensive model comparison table
        
        Returns:
            DataFrame with model comparison metrics
        """
        try:
            if not self.model_results:
                raise ValueError("No model results available")
            
            comparison_data = []
            
            for model_name, results in self.model_results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Test_R2': results['test_r2'],
                    'Test_RMSE': results['test_rmse'],
                    'Test_MAE': results['test_mae'],
                    'Test_Accuracy': results['test_accuracy'],
                    'CV_RMSE_Mean': results['cv_rmse_mean'],
                    'CV_RMSE_Std': results['cv_rmse_std'],
                    'Training_Time': results['training_time']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('Test_R2', ascending=False)
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error creating model comparison: {str(e)}")
            raise
    
    def save_best_model(self, filepath: Optional[Path] = None):
        """
        Save the best model to disk
        
        Args:
            filepath: Optional file path to save model
        """
        try:
            if self.best_model is None:
                raise ValueError("No best model available")
            
            filepath = filepath or MODELS_DIR / f"best_wine_model_{self.best_model_name}.pkl"
            
            joblib.dump(self.best_model, filepath)
            logger.info(f"Best model ({self.best_model_name}) saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving best model: {str(e)}")
            raise
    
    def load_model(self, filepath: Path):
        """
        Load a saved model
        
        Args:
            filepath: Path to saved model
        """
        try:
            self.best_model = joblib.load(filepath)
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict_wine_quality(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict wine quality using the best model
        
        Args:
            features: DataFrame with wine features
            
        Returns:
            Array of quality predictions
        """
        try:
            if self.best_model is None:
                raise ValueError("No trained model available")
            
            predictions = self.best_model.predict(features)
            
            # Ensure predictions are within valid range
            predictions = np.clip(predictions, 3, 9)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise