"""
Pytest configuration and fixtures for wine quality prediction tests
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

@pytest.fixture
def sample_wine_data():
    """Create sample wine data for testing"""
    np.random.seed(42)
    
    data = {
        'fixed_acidity': np.random.uniform(4.6, 15.9, 100),
        'volatile_acidity': np.random.uniform(0.12, 1.58, 100),
        'citric_acid': np.random.uniform(0.0, 1.0, 100),
        'residual_sugar': np.random.uniform(0.9, 15.5, 100),
        'chlorides': np.random.uniform(0.012, 0.611, 100),
        'free_sulfur_dioxide': np.random.uniform(1, 72, 100),
        'total_sulfur_dioxide': np.random.uniform(6, 289, 100),
        'density': np.random.uniform(0.99007, 1.00369, 100),
        'pH': np.random.uniform(2.74, 4.01, 100),
        'sulphates': np.random.uniform(0.33, 2.0, 100),
        'alcohol': np.random.uniform(8.4, 14.9, 100),
        'quality': np.random.randint(3, 9, 100)
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_features():
    """Create sample features without target"""
    np.random.seed(42)
    
    data = {
        'fixed_acidity': np.random.uniform(4.6, 15.9, 10),
        'volatile_acidity': np.random.uniform(0.12, 1.58, 10),
        'citric_acid': np.random.uniform(0.0, 1.0, 10),
        'residual_sugar': np.random.uniform(0.9, 15.5, 10),
        'chlorides': np.random.uniform(0.012, 0.611, 10),
        'free_sulfur_dioxide': np.random.uniform(1, 72, 10),
        'total_sulfur_dioxide': np.random.uniform(6, 289, 10),
        'density': np.random.uniform(0.99007, 1.00369, 10),
        'pH': np.random.uniform(2.74, 4.01, 10),
        'sulphates': np.random.uniform(0.33, 2.0, 10),
        'alcohol': np.random.uniform(8.4, 14.9, 10)
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def single_wine_sample():
    """Single wine sample for API testing"""
    return {
        "fixed_acidity": 7.4,
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