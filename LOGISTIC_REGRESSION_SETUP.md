# Logistic Regression Node - Setup Guide

## Overview

The Logistic Regression node uses a Python backend API to perform binary classification. The model is trained using scikit-learn's LogisticRegression.

## Setup Instructions

### 1. Install Python Dependencies

Navigate to the `backend` folder and install required packages:

```bash
cd backend
pip install -r requirements.txt
```

Required packages:
- flask==3.0.0
- flask-cors==4.0.0
- numpy==1.24.3
- scikit-learn==1.3.2

### 2. Start the Python API Server

**Windows:**
```bash
cd backend
python app.py
```
Or double-click `backend/start.bat`

**Linux/Mac:**
```bash
cd backend
python3 app.py
```
Or run `backend/start.sh`

The API will start on `http://localhost:5000`

### 3. Start the Frontend

In a separate terminal:

```bash
npm run dev
```

The frontend will run on `http://localhost:5173`

## Usage

### Workflow

1. **Connect Data Source:**
   - Start → CSV Reader (upload data)
   - Or: Start → CSV Reader → Encoder → Normalizer

2. **Add Logistic Regression Node:**
   - Drag "Logistic Regression" from sidebar
   - Connect it to your data source

3. **Configure:**
   - Select independent (X) columns (checkboxes)
   - Select dependent (Y) column (must be binary: 0/1 or two unique values)
   - Click "Train Model"

4. **View Results:**
   - Model metrics (Accuracy, Precision, Recall, F1-Score)
   - Confusion matrix
   - Predictions and probabilities

## API Endpoints

### POST /api/logistic-regression

**Request:**
```json
{
  "X": [[1, 2], [2, 3], [3, 4]],
  "y": [0, 1, 0],
  "feature_names": ["feature1", "feature2"],
  "target_name": "target"
}
```

**Response:**
```json
{
  "success": true,
  "coefficients": [0.5, -0.3],
  "intercept": 0.2,
  "accuracy": 0.95,
  "precision": 0.92,
  "recall": 0.90,
  "f1_score": 0.91,
  "predictions": [0, 1, 0],
  "probabilities": [0.3, 0.8, 0.2],
  "confusion_matrix": {
    "true_negatives": 10,
    "false_positives": 2,
    "false_negatives": 1,
    "true_positives": 12
  }
}
```

### GET /api/health

Health check endpoint to verify API is running.

## Requirements

- Python 3.7+
- Node.js and npm
- Binary classification data (dependent variable must have exactly 2 unique values)

## Troubleshooting

1. **"API server not running" warning:**
   - Make sure Python backend is running on port 5000
   - Check if port 5000 is available
   - Verify dependencies are installed

2. **"Binary classification required" error:**
   - Ensure your dependent variable has exactly 2 unique values
   - The API will automatically convert to 0/1 if needed

3. **CORS errors:**
   - Make sure flask-cors is installed
   - Check that the API is running on the correct port
