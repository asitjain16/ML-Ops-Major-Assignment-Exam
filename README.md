# MLOps Pipeline for Linear Regression - Major Assignment

A comprehensive MLOps pipeline implementing training, testing, quantization, Dockerization, and CI/CD for Linear Regression on the California Housing dataset.

## Assignment Overview

This project demonstrates a complete MLOps pipeline with the following components:
- **Model**: Scikit-learn LinearRegression
- **Dataset**: California Housing dataset from sklearn.datasets  
- **Pipeline**: Training → Testing → Quantization → Dockerization → CI/CD
- **Assignment Focus**: Manual quantization implementation and MLOps best practices

## Project Structure

```
├── src/
│   ├── train.py          # Model training script
│   ├── quantize.py       # Manual quantization implementation
│   ├── predict.py        # Prediction and model verification
│   └── utils.py          # Utility functions
├── tests/
│   └── test_train.py     # Unit tests for training pipeline
├── .github/workflows/
│   └── ci.yml           # CI/CD pipeline configuration
├── Dockerfile           # Container configuration
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## Features

### 1. Model Training (`src/train.py`)
- Loads California Housing dataset
- Trains LinearRegression model with feature scaling
- Prints R² score and MSE loss
- Saves model and scaler using joblib

### 2. Testing Pipeline (`tests/test_train.py`)
- Unit tests for dataset loading
- Validates LinearRegression model creation
- Checks if model coefficients exist after training
- Ensures R² score exceeds minimum threshold (0.5)

### 3. Manual Quantization (`src/quantize.py`)
- Extracts model coefficients and intercept
- Saves raw parameters (`unquant_params.joblib`)
- Manually quantizes to unsigned 8-bit integers
- Saves quantized parameters (`quant_params.joblib`)
- Performs inference with dequantized weights
- Calculates compression ratio and quantization error

### 4. Dockerization
- Multi-stage Docker build
- Installs dependencies and copies source code
- Includes `predict.py` for model verification
- Optimized for production deployment

### 5. CI/CD Pipeline (`.github/workflows/ci.yml`)
Three sequential jobs:
1. **test-suite**: Runs pytest (must pass before others execute)
2. **train-and-quantize**: Trains model and runs quantization (depends on test-suite)
3. **build-and-test-container**: Builds Docker image and tests container execution (depends on train-and-quantize)

## Assignment Execution Steps

### Local Development (Windows)

1. **Install dependencies:**
```cmd
pip install -r requirements.txt
```

2. **Run complete pipeline:**
```cmd
# Step 1: Train the model
cd src
python train.py

# Step 2: Run quantization
python quantize.py

# Step 3: Test predictions
python predict.py

# Step 4: Run unit tests
cd ..
python -m pytest tests/ -v
```

### Expected Output:
- **Training**: R² Score: 0.5758, MSE: 0.5559
- **Quantization**: 2.88x compression, MAE: 0.007215
- **Tests**: 8 tests passed
- **Predictions**: Successful model verification

### Docker Deployment (Assignment Verification)

1. **Build image:**
```cmd
docker build -t mlops-pipeline .
```

2. **Train model in container:**
```cmd
docker run --rm mlops-pipeline python src/train.py
```

3. **Run predictions (requires trained model):**
```cmd
docker run --rm mlops-pipeline python src/predict.py
```

**Note**: Container runs training first since models are generated at runtime.

## Actual Performance Results (Assignment Execution)

### Model Training Results
- **Training R² Score**: 0.6126
- **Test R² Score**: 0.5758  
- **Training MSE**: 0.5179
- **Test MSE**: 0.5559
- **Test RMSE**: 0.7456

### Quantization Results
| Metric | Original Model | Quantized Model | Actual Results |
|--------|---------------|-----------------|----------------|
| **Model Size** | 72 bytes | 25 bytes | **2.88x compression** |
| **Parameter Precision** | float32 (32-bit) | uint8 (8-bit) | **4x reduction** |
| **R² Score** | 0.5758 | 0.5758 | < 0.001 difference |
| **Quantization Scale** | N/A | 0.011643 | Linear quantization |
| **Zero Point** | N/A | 77.00 | Quantization offset |
| **Coefficients MAE** | N/A | 0.003351 | Minimal error |
| **Intercept Error** | N/A | 0.000445 | Negligible impact |
| **Prediction MAE** | N/A | 0.007215 | High accuracy maintained |

### Key Benefits of Manual Quantization (Assignment Implementation):
- **Storage Efficiency**: 65% reduction in model size (72 → 25 bytes)
- **Memory Optimization**: 4x less precision per parameter (float32 → uint8)
- **Accuracy Preservation**: No loss in R² score (0.5758 maintained)
- **Manual Implementation**: Custom quantization algorithm with scale/zero-point
- **Deployment Ready**: Optimized for edge devices and bandwidth-limited environments

## CI/CD Pipeline Status

The pipeline automatically:
- Runs comprehensive unit tests
- Trains and validates model performance
- Applies quantization with error analysis
- Builds and tests Docker container
- Uploads model artifacts for deployment

## Assignment Technical Implementation

### Model Architecture
- **Algorithm**: Linear Regression (scikit-learn)
- **Features**: 8 input features (California Housing dataset)
- **Preprocessing**: StandardScaler normalization
- **Target**: Continuous house price prediction (median value)

### Manual Quantization Implementation (Core Assignment)
- **Method**: Custom linear quantization algorithm
- **Target**: 8-bit unsigned integers (0-255 range)
- **Quantization Formula**: `quantized = clip(round(params/scale + zero_point), 0, 255)`
- **Dequantization Formula**: `params = scale * (quantized - zero_point)`
- **Scale Calculation**: `(param_max - param_min) / (qmax - qmin)`
- **Zero Point**: `qmin - param_min / scale`

### Assignment Deliverables
1. **Training Pipeline**: Automated model training with performance metrics
2. **Manual Quantization**: Custom 8-bit quantization implementation
3. **Testing Suite**: Comprehensive unit tests (8 test cases)
4. **Dockerization**: Containerized deployment
5. **CI/CD Pipeline**: Automated testing and deployment workflow
6. **Performance Analysis**: Quantization impact assessment

### Quality Assurance (Assignment Requirements)
- R² Score threshold validation (> 0.5): **0.5758 achieved**
- Quantization error analysis: **MAE < 0.01**
- Model size reduction: **2.88x compression**
- Automated testing: **8/8 tests passed**
- Container deployment: **Successfully built and tested**

This assignment demonstrates advanced MLOps practices with manual quantization implementation for model optimization and deployment.

## Assignment Execution Summary

### Completed Tasks
1. **Model Training**: Linear Regression on California Housing dataset
   - Training R²: 0.6126, Test R²: 0.5758
   - MSE Loss: 0.5559, RMSE: 0.7456

2. **Manual Quantization Implementation**: 
   - Custom 8-bit quantization algorithm
   - 2.88x model size reduction (72 → 25 bytes)
   - Quantization error MAE: 0.007215 (negligible)

3. **Testing Pipeline**:
   - 8 comprehensive unit tests
   - All tests passed successfully
   - Performance threshold validation (R² > 0.5)

4. **Dockerization**:
   - Successfully built Docker image
   - Container runs training and prediction
   - Production-ready deployment

5. **CI/CD Pipeline**:
   - Automated testing workflow
   - Model training and quantization
   - Container build and testing

Name: Asit Jain
Roll Number: G24AI1069

### Files Cleaned Up
- Removed duplicate model files from root directory
- Cleaned Python cache directories (__pycache__)
- Removed unnecessary pytest cache files
- Streamlined project structure for assignment submission
