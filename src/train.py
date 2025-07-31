"""
Training script for Linear Regression model on California Housing dataset
"""
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from utils import (
    load_california_housing_data, 
    scale_features, 
    save_model,
    calculate_r2_score,
    calculate_mse
)


def train_model():
    """Train Linear Regression model"""
    print("Loading California Housing dataset...")
    
    # Load and split data
    X_train, X_test, y_train, y_test = load_california_housing_data()
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    print(f"Training set size: {X_train_scaled.shape}")
    print(f"Test set size: {X_test_scaled.shape}")
    
    # Create and train model
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    # Print results
    print("\n=== Training Results ===")
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Test R² Score: {test_r2:.4f}")
    print(f"Training MSE (Loss): {train_mse:.4f}")
    print(f"Test MSE (Loss): {test_mse:.4f}")
    
    # Save model and scaler
    save_model(model, 'linear_regression_model.joblib')
    save_model(scaler, 'scaler.joblib')
    
    return model, scaler, test_r2


if __name__ == "__main__":
    train_model()