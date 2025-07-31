"""
Unit tests for the training pipeline
"""
import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sklearn.linear_model import LinearRegression
from utils import load_california_housing_data, scale_features
from train import train_model


class TestDatasetLoading:
    """Test dataset loading functionality"""
    
    def test_load_california_housing_data(self):
        """Test if dataset loads correctly"""
        X_train, X_test, y_train, y_test = load_california_housing_data()
        
        # Check if data is loaded
        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None
        
        # Check shapes
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        
        # Check if features match
        assert X_train.shape[1] == X_test.shape[1]
        assert X_train.shape[1] == 8  # California housing has 8 features
        
    def test_data_split_ratio(self):
        """Test if data split maintains correct ratio"""
        X_train, X_test, y_train, y_test = load_california_housing_data(test_size=0.2)
        
        total_samples = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total_samples
        
        # Allow small tolerance for rounding
        assert abs(test_ratio - 0.2) < 0.01


class TestModelCreation:
    """Test model creation and training"""
    
    def test_model_instance_creation(self):
        """Test if LinearRegression instance is created correctly"""
        model = LinearRegression()
        assert isinstance(model, LinearRegression)
        
    def test_model_training(self):
        """Test if model can be trained"""
        # Load data
        X_train, X_test, y_train, y_test = load_california_housing_data()
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        
        # Create and train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Check if model has been trained (coefficients exist)
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        assert model.coef_ is not None
        assert model.intercept_ is not None
        
        # Check coefficient shape
        assert len(model.coef_) == 8  # 8 features
        
    def test_model_coefficients_exist(self):
        """Test if trained model has coefficients"""
        model, scaler, r2_score = train_model()
        
        # Check if coefficients exist after training
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        assert model.coef_ is not None
        assert model.intercept_ is not None


class TestModelPerformance:
    """Test model performance metrics"""
    
    def test_r2_score_threshold(self):
        """Test if R² score exceeds minimum threshold"""
        model, scaler, test_r2 = train_model()
        
        # R² score should be reasonable for California housing dataset
        # Typically should be above 0.5 for this dataset
        MIN_R2_THRESHOLD = 0.5
        assert test_r2 > MIN_R2_THRESHOLD, f"R² score {test_r2:.4f} is below minimum threshold {MIN_R2_THRESHOLD}"
        
    def test_predictions_shape(self):
        """Test if predictions have correct shape"""
        X_train, X_test, y_train, y_test = load_california_housing_data()
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        predictions = model.predict(X_test_scaled)
        
        # Check prediction shape
        assert len(predictions) == len(y_test)
        assert predictions.shape == y_test.shape


class TestFeatureScaling:
    """Test feature scaling functionality"""
    
    def test_feature_scaling(self):
        """Test if feature scaling works correctly"""
        X_train, X_test, y_train, y_test = load_california_housing_data()
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        
        # Check if scaling is applied (mean should be close to 0, std close to 1)
        assert abs(np.mean(X_train_scaled)) < 0.1
        assert abs(np.std(X_train_scaled) - 1.0) < 0.1
        
        # Check shapes are preserved
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape