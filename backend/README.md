# Logistic Regression API Backend

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Running the Server

```bash
python app.py
```

The API will run on `http://localhost:5000`

## API Endpoints

### POST /api/logistic-regression
Train a logistic regression model

**Request Body:**
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
Health check endpoint
