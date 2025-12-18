import React, { memo, useMemo, useState } from 'react';
import { Handle, Position, useStore } from 'reactflow';
import './ModelEvaluatorNode.css';

// Sigmoid function for logistic regression
function sigmoid(z) {
  // Clip z to prevent overflow
  z = Math.max(-500, Math.min(500, z));
  return 1 / (1 + Math.exp(-z));
}

// Distance calculation functions for KNN
function euclideanDistance(x1, x2) {
  return Math.sqrt(x1.reduce((sum, val, i) => sum + Math.pow(val - x2[i], 2), 0));
}

function manhattanDistance(x1, x2) {
  return x1.reduce((sum, val, i) => sum + Math.abs(val - x2[i]), 0);
}

function minkowskiDistance(x1, x2, p = 3) {
  return Math.pow(x1.reduce((sum, val, i) => sum + Math.pow(Math.abs(val - x2[i]), p), 0), 1 / p);
}

function chebyshevDistance(x1, x2) {
  return Math.max(...x1.map((val, i) => Math.abs(val - x2[i])));
}

function cosineSimilarityDistance(x1, x2) {
  const dotProduct = x1.reduce((sum, val, i) => sum + val * x2[i], 0);
  const norm1 = Math.sqrt(x1.reduce((sum, val) => sum + val * val, 0));
  const norm2 = Math.sqrt(x2.reduce((sum, val) => sum + val * val, 0));

  if (norm1 === 0 || norm2 === 0) return 1.0;

  const cosineSim = dotProduct / (norm1 * norm2);
  return 1 - cosineSim;
}


const ModelEvaluatorNode = ({ id, data, isConnectable }) => {
  const [inputValues, setInputValues] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState('');

  // Find upstream model node
  const upstreamModel = useStore(
    (store) => {
      const incoming = Array.from(store.edges.values()).filter((e) => e.target === id);
      for (const e of incoming) {
        const src = store.nodeInternals.get(e.source);
        if (src?.type === 'linearRegression' ||
          src?.type === 'multiLinearRegression' ||
          src?.type === 'polynomialRegression' ||
          src?.type === 'logisticRegression' ||
          src?.type === 'knnRegression' ||
          src?.type === 'knnClassification') {
          const modelData = src.data?.model;
          // Return the model data even if it's null/undefined, so we can detect when it changes
          return {
            type: src.type,
            model: modelData,
            nodeId: src.id,
            // Include a hash of model data to force re-render when model changes
            modelHash: modelData ? JSON.stringify(modelData) : null
          };
        }
      }
      return null;
    },
    // Custom equality function to prevent re-renders when the object content hasn't changed
    (prev, next) => {
      if (prev === next) return true;
      if (!prev || !next) return false;
      return prev.type === next.type &&
        prev.nodeId === next.nodeId &&
        prev.modelHash === next.modelHash;
    }
  );

  // Determine model type and extract feature names
  // This will re-compute whenever upstreamModel changes (including modelHash)
  const modelInfo = useMemo(() => {
    if (!upstreamModel) return null;

    const { type, model } = upstreamModel;

    // If model is not available yet, return null
    if (!model) return null;

    if (type === 'linearRegression') {
      // Simple linear regression: one X variable
      // Validate that required model properties exist
      if (model.slope === undefined || model.intercept === undefined || !model.xCol) {
        return null;
      }
      return {
        type: 'linear',
        featureNames: [model.xCol || 'X'],
        model: {
          slope: model.slope,
          intercept: model.intercept
        }
      };
    } else if (type === 'multiLinearRegression') {
      // Multi linear regression: multiple X variables
      // Validate that required model properties exist
      if (!model.coefficients || !Array.isArray(model.coefficients) ||
        model.coefficients.length === 0 || model.intercept === undefined ||
        !model.xCols || model.xCols.length === 0) {
        return null;
      }
      return {
        type: 'multiLinear',
        featureNames: model.xCols || [],
        model: {
          coefficients: model.coefficients || [],
          intercept: model.intercept
        }
      };
    } else if (type === 'logisticRegression') {
      // Logistic regression: multiple X variables
      // Validate that required model properties exist
      if (!model.coefficients || !Array.isArray(model.coefficients) ||
        model.coefficients.length === 0 || model.intercept === undefined ||
        !model.xCols || model.xCols.length === 0) {
        return null;
      }
      return {
        type: 'logistic',
        featureNames: model.xCols || [],
        model: {
          coefficients: model.coefficients || [],
          intercept: model.intercept
        }
      };
    } else if (type === 'knnRegression') {
      // KNN regression: requires training data for predictions
      // Validate that required model properties exist
      if (!model.k || !model.distance_metric ||
        !model.xCols || model.xCols.length === 0 ||
        !model.X_train || !model.y_train) {
        return null;
      }
      return {
        type: 'knn',
        featureNames: model.xCols || [],
        model: {
          k: model.k,
          distance_metric: model.distance_metric,
          yCol: model.yCol,
          X_train: model.X_train,
          y_train: model.y_train
        }
      };
    } else if (type === 'polynomialRegression') {
      // Polynomial regression: multiple X variables with polynomial features
      // Validate that required model properties exist
      if (!model.coefficients || !Array.isArray(model.coefficients) ||
        model.coefficients.length === 0 || model.intercept === undefined ||
        !model.xCols || model.xCols.length === 0 ||
        model.degree === undefined) {
        return null;
      }
      return {
        type: 'polynomial',
        featureNames: model.xCols || [],
        model: {
          coefficients: model.coefficients || [],
          intercept: model.intercept,
          degree: model.degree,
          n_features_original: model.n_features_original,
          n_features_poly: model.n_features_poly
        }
      };
    } else if (type === 'knnClassification') {
      // KNN classification: requires training data for predictions
      if (!model.k || !model.distance_metric ||
        !model.xCols || model.xCols.length === 0 ||
        !model.X_train || !model.y_train) {
        return null;
      }
      return {
        type: 'knnClassification',
        featureNames: model.xCols || [],
        model: {
          k: model.k,
          distance_metric: model.distance_metric,
          yCol: model.yCol,
          X_train: model.X_train,
          y_train: model.y_train
        }
      };
    }

    return null;
  }, [upstreamModel]);

  // Initialize input values when model changes or is retrained
  // This effect runs whenever modelInfo changes (which happens when modelHash changes)
  React.useEffect(() => {
    if (modelInfo && modelInfo.featureNames) {
      setInputValues(prev => {
        const initialValues = {};
        const currentFeatureNames = modelInfo.featureNames;

        // Check if feature names have changed (model retrained with different features)
        const featureNamesChanged = currentFeatureNames.some(f => !(f in prev)) ||
          Object.keys(prev).some(f => !currentFeatureNames.includes(f));

        if (featureNamesChanged) {
          // If features changed, clear all inputs
          currentFeatureNames.forEach(feature => {
            initialValues[feature] = '';
          });
        } else {
          // If features are the same, preserve existing values
          currentFeatureNames.forEach(feature => {
            initialValues[feature] = prev[feature] || '';
          });
        }
        return initialValues;
      });
      // Clear prediction when model changes
      setPrediction(null);
      setError('');
    } else if (upstreamModel && !upstreamModel.model) {
      // Model node exists but model is not trained yet
      setInputValues({});
      setPrediction(null);
      setError('');
    }
  }, [modelInfo, upstreamModel]);

  const numericRegex = /^-?\d*\.?\d*$/;

  const handleInputChange = (feature, value) => {
    // Allow empty, "-", ".", "-.", or any partial numeric input
    if (value === "" || numericRegex.test(value)) {
      setInputValues(prev => ({
        ...prev,
        [feature]: value
      }));
    }
  };

  const calculatePrediction = () => {
    if (!modelInfo) {
      setError('No model connected. Please connect a trained model node.');
      return;
    }

    try {
      const { type, featureNames, model } = modelInfo;

      // Validate all inputs are provided
      const values = featureNames.map(feature => {
        const value = inputValues[feature];
        if (value === '' || value === null || value === undefined) {
          throw new Error(`Please enter a value for ${feature}`);
        }
        const numValue = parseFloat(value);
        if (isNaN(numValue)) {
          throw new Error(`${feature} must be a valid number`);
        }
        return numValue;
      });

      let result;

      if (type === 'linear') {
        // Simple linear regression: y = slope * x + intercept
        const x = values[0];
        result = {
          type: 'linear',
          prediction: model.slope * x + model.intercept,
          equation: `y = ${model.slope.toFixed(4)} × ${x} + ${model.intercept.toFixed(4)} = ${(model.slope * x + model.intercept).toFixed(4)}`
        };
      } else if (type === 'multiLinear') {
        // Multi linear regression: y = intercept + sum(coefficients[i] * x[i])
        let sum = model.intercept;
        const terms = [`${model.intercept.toFixed(4)}`];
        featureNames.forEach((feature, i) => {
          const term = model.coefficients[i] * values[i];
          sum += term;
          terms.push(`${model.coefficients[i].toFixed(4)} × ${values[i]}`);
        });
        result = {
          type: 'multiLinear',
          prediction: sum,
          equation: `y = ${terms.join(' + ')} = ${sum.toFixed(4)}`
        };
      } else if (type === 'logistic') {
        // Logistic regression: probability = sigmoid(intercept + sum(coefficients[i] * x[i]))
        let linearSum = model.intercept;
        featureNames.forEach((feature, i) => {
          linearSum += model.coefficients[i] * values[i];
        });
        const probability = sigmoid(linearSum);
        const prediction = probability >= 0.5 ? 1 : 0;
        result = {
          type: 'logistic',
          prediction: prediction,
          probability: probability,
          equation: `P(y=1) = sigmoid(${linearSum.toFixed(4)}) = ${probability.toFixed(4)}`,
          interpretation: `Predicted class: ${prediction} (${(probability * 100).toFixed(2)}% probability)`
        };
      } else if (type === 'polynomial') {
        // Polynomial regression: generate polynomial features and compute prediction
        // Generate polynomial features from input values
        const generatePolynomialFeatures = (X, degree) => {
          const features = [1]; // bias term
          const n = X.length;

          // Add original features (degree 1)
          for (let i = 0; i < n; i++) {
            features.push(X[i]);
          }

          // Generate higher degree features
          if (degree > 1) {
            for (let d = 2; d <= degree; d++) {
              // Generate all combinations with replacement
              const generateCombinations = (arr, size, start = 0, current = []) => {
                if (current.length === size) {
                  // Calculate product of features
                  let product = 1;
                  for (const idx of current) {
                    product *= arr[idx];
                  }
                  features.push(product);
                  return;
                }
                for (let i = start; i < arr.length; i++) {
                  generateCombinations(arr, size, i, [...current, i]);
                }
              };
              generateCombinations(X, d);
            }
          }

          return features;
        };

        const polyFeatures = generatePolynomialFeatures(values, model.degree);

        // Compute prediction: sum of (coefficient * feature)
        let sum = 0;
        for (let i = 0; i < model.coefficients.length && i < polyFeatures.length; i++) {
          sum += model.coefficients[i] * polyFeatures[i];
        }

        result = {
          type: 'polynomial',
          prediction: sum,
          degree: model.degree,
          n_features_original: model.n_features_original,
          n_features_poly: model.n_features_poly
        };
      } else if (type === 'knn') {
        // KNN regression: find K nearest neighbors and average their values
        const inputPoint = values; // User's input values

        // Select distance function based on metric
        const distanceFunctions = {
          'euclidean': euclideanDistance,
          'manhattan': manhattanDistance,
          'minkowski': minkowskiDistance,
          'chebyshev': chebyshevDistance,
          'cosine': cosineSimilarityDistance
        };

        const distanceFunc = distanceFunctions[model.distance_metric] || euclideanDistance;

        // Calculate distances to all training points
        const distances = model.X_train.map((trainPoint, idx) => ({
          distance: distanceFunc(inputPoint, trainPoint),
          yValue: model.y_train[idx]
        }));

        // Sort by distance and get K nearest neighbors
        distances.sort((a, b) => a.distance - b.distance);
        const kNearest = distances.slice(0, model.k);

        // Average the y values of K nearest neighbors
        const prediction = kNearest.reduce((sum, neighbor) => sum + neighbor.yValue, 0) / model.k;

        result = {
          type: 'knn',
          prediction: prediction,
          yCol: model.yCol,
          k: model.k,
          distance_metric: model.distance_metric,
          nearestDistances: kNearest.map(n => n.distance.toFixed(4))
        };
      } else if (type === 'knnClassification') {
        // KNN classification: find K nearest neighbors and use majority vote
        const inputPoint = values; // User's input values

        // Select distance function based on metric
        const distanceFunctions = {
          'euclidean': euclideanDistance,
          'manhattan': manhattanDistance,
          'minkowski': minkowskiDistance,
          'chebyshev': chebyshevDistance,
          'cosine': cosineSimilarityDistance
        };

        const distanceFunc = distanceFunctions[model.distance_metric] || euclideanDistance;

        // Calculate distances to all training points
        const distances = model.X_train.map((trainPoint, idx) => ({
          distance: distanceFunc(inputPoint, trainPoint),
          classLabel: model.y_train[idx]
        }));

        // Sort by distance and get K nearest neighbors
        distances.sort((a, b) => a.distance - b.distance);
        const kNearest = distances.slice(0, model.k);

        // Majority vote - count occurrences of each class
        const classCounts = {};
        kNearest.forEach(neighbor => {
          const label = neighbor.classLabel;
          classCounts[label] = (classCounts[label] || 0) + 1;
        });

        // Find the class with the most votes
        let predictedClass = null;
        let maxVotes = 0;
        for (const [classLabel, count] of Object.entries(classCounts)) {
          if (count > maxVotes) {
            maxVotes = count;
            predictedClass = parseFloat(classLabel);
          }
        }

        result = {
          type: 'knnClassification',
          prediction: predictedClass,
          yCol: model.yCol,
          k: model.k,
          distance_metric: model.distance_metric,
          votes: classCounts,
          confidence: (maxVotes / model.k * 100).toFixed(1)
        };
      }

      setPrediction(result);
      setError('');
    } catch (err) {
      setError(err.message);
      setPrediction(null);
    }
  };


  return (
    <div className="model-evaluator-node">
      <Handle type="target" position={Position.Top} isConnectable={isConnectable} />

      <div className="evaluator-header">
        <span className="evaluator-title">{data.label || 'Model Evaluator'}</span>
      </div>

      {!upstreamModel && (
        <div className="evaluator-content">
          <div className="evaluator-placeholder">
            Connect a trained model node (Linear Regression, Multi Linear Regression, Polynomial Regression, Logistic Regression, KNN Regression, or KNN Classification)
          </div>
        </div>
      )}

      {upstreamModel && !modelInfo && (
        <div className="evaluator-content">
          <div className="evaluator-error">
            Model not trained yet. Please train the connected model first.
          </div>
        </div>
      )}

      {modelInfo && (
        <div className="evaluator-content">
          <div className="input-section">
            {modelInfo.featureNames.map((feature, index) => (
              <div key={feature} className="input-field">
                <label className="input-label">{feature}:</label>
                <div className="nodrag">
                  <input
                    type="text"
                    step="any"
                    value={inputValues[feature] || ''}
                    onChange={(e) => handleInputChange(feature, e.target.value)}
                    placeholder={`Enter ${feature} value`}
                    onKeyPress={(e) => {
                      if (e.key === 'Enter') {
                        calculatePrediction();
                      }
                    }}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="evaluator-actions">
            <button className="btn compute-btn" onClick={calculatePrediction}>
              Compute
            </button>
          </div>

          {error && (
            <div className="evaluator-error">{error}</div>
          )}

          {prediction && (
            <div className="prediction-result">
              {prediction.type === 'polynomial' ? (
                <div>
                  <div style={{ marginBottom: '12px', padding: '8px', background: '#f0f4f8', borderRadius: '4px' }}>
                    <div style={{ fontSize: '0.85em', color: '#64748b', marginBottom: '4px' }}>Prediction:</div>
                    <div className="prediction-value">
                      <strong>{prediction.prediction.toFixed(4)}</strong>
                    </div>
                  </div>
                  <div style={{ fontSize: '0.8em', color: '#64748b', padding: '8px', background: '#f8fafc', borderRadius: '4px' }}>
                    <div><strong>Polynomial Degree:</strong> {prediction.degree}</div>
                    <div><strong>Original Features:</strong> {prediction.n_features_original}</div>
                    <div><strong>Polynomial Features:</strong> {prediction.n_features_poly}</div>
                  </div>
                </div>
              ) : prediction.type === 'knn' ? (
                <div>
                  <div style={{ marginBottom: '12px', padding: '8px', background: '#f0f4f8', borderRadius: '4px' }}>
                    <div style={{ fontSize: '0.85em', color: '#64748b', marginBottom: '4px' }}>Predicted {prediction.yCol}:</div>
                    <div className="prediction-value">
                      <strong>{prediction.prediction.toFixed(4)}</strong>
                    </div>
                  </div>
                  <div style={{ fontSize: '0.8em', color: '#64748b', padding: '8px', background: '#f8fafc', borderRadius: '4px' }}>
                    <div><strong>K Neighbors:</strong> {prediction.k}</div>
                    <div><strong>Distance Metric:</strong> {prediction.distance_metric}</div>
                    <div style={{ marginTop: '4px', fontSize: '0.75em', fontStyle: 'italic' }}>
                      Avg of {prediction.k} nearest training points
                    </div>
                  </div>
                </div>
              ) : prediction.type === 'knnClassification' ? (
                <div>
                  <div style={{ marginBottom: '12px', padding: '8px', background: '#f0f4f8', borderRadius: '4px' }}>
                    <div style={{ fontSize: '0.85em', color: '#64748b', marginBottom: '4px' }}>Predicted {prediction.yCol}:</div>
                    <div className="prediction-value">
                      <strong>Class {prediction.prediction}</strong>
                    </div>
                    <div style={{ fontSize: '0.8em', color: '#16a34a', marginTop: '4px' }}>
                      Confidence: {prediction.confidence}%
                    </div>
                  </div>
                  <div style={{ fontSize: '0.8em', color: '#64748b', padding: '8px', background: '#f8fafc', borderRadius: '4px' }}>
                    <div><strong>K Neighbors:</strong> {prediction.k}</div>
                    <div><strong>Distance Metric:</strong> {prediction.distance_metric}</div>
                    <div style={{ marginTop: '6px' }}><strong>Votes:</strong></div>
                    {Object.entries(prediction.votes).map(([classLabel, count]) => (
                      <div key={classLabel} style={{ marginLeft: '8px', fontSize: '0.75em' }}>
                        Class {classLabel}: {count} vote{count !== 1 ? 's' : ''}
                      </div>
                    ))}
                  </div>
                </div>
              ) : prediction.type === 'logistic' ? (
                <div className="prediction-value">
                  <strong>{prediction.prediction}</strong> ({(prediction.probability * 100).toFixed(1)}%)
                </div>
              ) : (
                <div className="prediction-value">
                  <strong>{prediction.prediction.toFixed(4)}</strong>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      <Handle type="source" position={Position.Bottom} isConnectable={isConnectable} />
    </div>
  );
};

export default memo(ModelEvaluatorNode);