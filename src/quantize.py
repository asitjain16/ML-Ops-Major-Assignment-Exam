"""
Manual quantization script for Linear Regression model
"""
import numpy as np
import joblib
from utils import load_model, save_model, load_california_housing_data, scale_features


def quantize_parameters(params, bits=8):
    """
    Manually quantize parameters to unsigned 8-bit integers
    
    Args:
        params: numpy array of parameters
        bits: number of bits for quantization (default: 8)
    
    Returns:
        tuple: (quantized_params, scale, zero_point)
    """
    # Calculate min and max values
    param_min = np.min(params)
    param_max = np.max(params)
    
    # Calculate scale and zero point for quantization
    qmin = 0
    qmax = (2 ** bits) - 1
    
    scale = (param_max - param_min) / (qmax - qmin)
    zero_point = qmin - param_min / scale
    zero_point = np.clip(np.round(zero_point), qmin, qmax)
    
    # Quantize parameters
    quantized = np.clip(np.round(params / scale + zero_point), qmin, qmax)
    quantized = quantized.astype(np.uint8)
    
    return quantized, scale, zero_point


def dequantize_parameters(quantized_params, scale, zero_point):
    """
    Dequantize parameters back to float32
    
    Args:
        quantized_params: quantized parameters
        scale: quantization scale
        zero_point: quantization zero point
    
    Returns:
        numpy array: dequantized parameters
    """
    return scale * (quantized_params.astype(np.float32) - zero_point)


def quantize_model():
    """Main quantization function"""
    print("Loading trained model...")
    
    try:
        # Load trained model
        model = load_model('linear_regression_model.joblib')
        scaler = load_model('scaler.joblib')
        
        print("Model loaded successfully!")
        print(f"Model coefficients shape: {model.coef_.shape}")
        print(f"Model intercept: {model.intercept_}")
        
    except FileNotFoundError:
        print("Model not found. Please run training first.")
        return
    
    # Extract coefficients and intercept
    coef = model.coef_
    intercept = np.array([model.intercept_])
    
    # Combine all parameters
    all_params = np.concatenate([coef, intercept])
    
    print(f"\nOriginal parameters shape: {all_params.shape}")
    print(f"Original parameters range: [{np.min(all_params):.4f}, {np.max(all_params):.4f}]")
    
    # Save raw parameters
    raw_params = {
        'coef': coef,
        'intercept': intercept,
        'all_params': all_params
    }
    save_model(raw_params, 'unquant_params.joblib')
    
    # Quantize parameters
    print("\nQuantizing parameters to 8-bit unsigned integers...")
    quantized_params, scale, zero_point = quantize_parameters(all_params, bits=8)
    
    print(f"Quantized parameters shape: {quantized_params.shape}")
    print(f"Quantized parameters range: [{np.min(quantized_params)}, {np.max(quantized_params)}]")
    print(f"Quantization scale: {scale:.6f}")
    print(f"Quantization zero point: {zero_point:.2f}")
    
    # Save quantized parameters
    quant_params = {
        'quantized_params': quantized_params,
        'scale': scale,
        'zero_point': zero_point,
        'coef_shape': coef.shape,
        'intercept_shape': intercept.shape
    }
    save_model(quant_params, 'quant_params.joblib')
    
    # Dequantize for inference test
    print("\nTesting dequantization...")
    dequantized_params = dequantize_parameters(quantized_params, scale, zero_point)
    
    # Split back to coefficients and intercept
    dequant_coef = dequantized_params[:-1]
    dequant_intercept = dequantized_params[-1]
    
    print(f"Dequantized coefficients shape: {dequant_coef.shape}")
    print(f"Dequantized intercept: {dequant_intercept:.4f}")
    
    # Calculate quantization error
    coef_error = np.mean(np.abs(coef - dequant_coef))
    intercept_error = abs(model.intercept_ - dequant_intercept)
    
    print(f"\nQuantization Error:")
    print(f"Coefficients MAE: {coef_error:.6f}")
    print(f"Intercept Error: {intercept_error:.6f}")
    
    # Test inference with dequantized weights
    print("\nTesting inference with dequantized weights...")
    
    # Load test data
    X_train, X_test, y_train, y_test = load_california_housing_data()
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)
    
    # Original model predictions
    original_pred = model.predict(X_test_scaled[:5])  # Test on first 5 samples
    
    # Dequantized model predictions
    dequant_pred = X_test_scaled[:5] @ dequant_coef + dequant_intercept
    
    print(f"\nInference Comparison (first 5 samples):")
    print(f"Original predictions: {original_pred}")
    print(f"Dequantized predictions: {dequant_pred}")
    print(f"Prediction difference (MAE): {np.mean(np.abs(original_pred - dequant_pred)):.6f}")
    
    # Calculate size reduction
    original_size = all_params.nbytes
    quantized_size = quantized_params.nbytes + 8 + 8  # +8 for scale, +8 for zero_point
    compression_ratio = original_size / quantized_size
    
    print(f"\nModel Size Comparison:")
    print(f"Original model size: {original_size} bytes")
    print(f"Quantized model size: {quantized_size} bytes")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    return quant_params


if __name__ == "__main__":
    quantize_model()