"""
Prediction script for model verification
"""
import numpy as np
from utils import load_model, load_california_housing_data, scale_features


def run_predictions():
    """Load trained model and run predictions on test set"""
    print("Loading trained model for predictions...")
    
    try:
        # Load model and scaler
        model = load_model('linear_regression_model.joblib')
        scaler = load_model('scaler.joblib')
        print("Model and scaler loaded successfully!")
        
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model is trained first.")
        return
    
    # Load test data
    print("Loading California Housing dataset...")
    X_train, X_test, y_train, y_test = load_california_housing_data()
    
    # Scale features
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)
    
    # Make predictions
    print("Making predictions on test set...")
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\n=== Model Performance ===")
    print(f"Test R² Score: {r2:.4f}")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    
    # Show sample predictions
    print(f"\n=== Sample Predictions ===")
    print("Actual vs Predicted (first 10 samples):")
    print("Actual\t\tPredicted\tDifference")
    print("-" * 45)
    
    for i in range(min(10, len(y_test))):
        actual = y_test[i]
        predicted = y_pred[i]
        diff = abs(actual - predicted)
        print(f"{actual:.4f}\t\t{predicted:.4f}\t\t{diff:.4f}")
    
    # Statistics
    print(f"\n=== Prediction Statistics ===")
    print(f"Mean Actual Value: {np.mean(y_test):.4f}")
    print(f"Mean Predicted Value: {np.mean(y_pred):.4f}")
    print(f"Std Actual Value: {np.std(y_test):.4f}")
    print(f"Std Predicted Value: {np.std(y_pred):.4f}")
    print(f"Min Actual Value: {np.min(y_test):.4f}")
    print(f"Max Actual Value: {np.max(y_test):.4f}")
    print(f"Min Predicted Value: {np.min(y_pred):.4f}")
    print(f"Max Predicted Value: {np.max(y_pred):.4f}")
    
    print("\nPrediction completed successfully!")
    return True


if __name__ == "__main__":
    run_predictions()