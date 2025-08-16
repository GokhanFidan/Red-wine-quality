"""
Wine Quality Prediction - Configuration Management
Enterprise-grade configuration for ML pipeline
"""

import os
from pathlib import Path
from typing import Dict, List

# Project Structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
EXTERNAL_DATA_DIR = DATA_DIR / 'external'

OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
MODELS_DIR = OUTPUTS_DIR / 'models'
REPORTS_DIR = OUTPUTS_DIR / 'reports'
PLOTS_DIR = OUTPUTS_DIR / 'plots'

# Data Configuration
WINE_DATA_FILE = RAW_DATA_DIR / 'winequality-red.csv'
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / 'wine_processed.csv'

# Model Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature Engineering
FEATURE_COLUMNS = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

TARGET_COLUMN = 'quality'

# Wine Quality Categories
QUALITY_CATEGORIES = {
    'Poor': [3, 4],
    'Average': [5, 6],
    'Good': [7, 8],
    'Excellent': [9, 10]
}

# Model Hyperparameters
MODEL_CONFIGS = {
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'xgboost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    },
    'svm': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01],
        'kernel': ['rbf', 'poly', 'sigmoid']
    },
    'neural_network': {
        'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }
}

# Visualization Configuration
PLOT_STYLE = 'seaborn-v0_8'
COLOR_PALETTE = {
    'primary': '#8B0000',      # Dark Red (Wine Theme)
    'secondary': '#DC143C',    # Crimson
    'accent': '#FFD700',       # Gold
    'neutral': '#2F4F4F',      # Dark Slate Gray
    'success': '#228B22',      # Forest Green
    'warning': '#FF8C00',      # Dark Orange
    'error': '#B22222'         # Fire Brick
}

FIGURE_SIZE = {
    'small': (8, 6),
    'medium': (12, 8),
    'large': (15, 10),
    'dashboard': (20, 12)
}

# Business Metrics
WINE_BUSINESS_METRICS = {
    'cost_per_poor_wine': 5.0,     # USD cost of producing poor quality wine
    'revenue_per_good_wine': 25.0,  # USD revenue from good quality wine
    'cost_per_test': 2.0,          # USD cost per quality test
    'processing_cost': 1.5,        # USD processing cost per bottle
}

# API Configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'debug': False,
    'model_version': '1.0.0',
    'max_predictions_per_request': 1000
}

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': PROJECT_ROOT / 'logs' / 'wine_quality.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

# Model Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'min_accuracy': 0.80,
    'min_precision': 0.75,
    'min_recall': 0.75,
    'min_f1_score': 0.75,
    'max_training_time': 300  # seconds
}

# Data Validation Rules
DATA_VALIDATION_RULES = {
    'fixed acidity': {'min': 4.0, 'max': 16.0},
    'volatile acidity': {'min': 0.0, 'max': 2.0},
    'citric acid': {'min': 0.0, 'max': 1.0},
    'residual sugar': {'min': 0.0, 'max': 16.0},
    'chlorides': {'min': 0.0, 'max': 1.0},
    'free sulfur dioxide': {'min': 0.0, 'max': 80.0},
    'total sulfur dioxide': {'min': 0.0, 'max': 300.0},
    'density': {'min': 0.99, 'max': 1.01},
    'pH': {'min': 2.5, 'max': 4.5},
    'sulphates': {'min': 0.0, 'max': 2.0},
    'alcohol': {'min': 8.0, 'max': 16.0},
    'quality': {'min': 0, 'max': 10}
}

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
                  OUTPUTS_DIR, MODELS_DIR, REPORTS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)