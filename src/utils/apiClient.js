// API client for backend communication
const API_BASE_URL = 'http://localhost:5000/api';

export async function trainLogisticRegression(X, y, trainPercent, featureNames = [], targetName = 'target') {
  try {
    const response = await fetch(`${API_BASE_URL}/logistic-regression`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        X,
        y,
        train_percent: trainPercent,
        feature_names: featureNames,
        target_name: targetName
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    throw new Error(`API call failed: ${error.message}`);
  }
}

export async function trainLinearRegression(X, y, trainPercent, featureName = 'X', targetName = 'y') {
  try {
    const response = await fetch(`${API_BASE_URL}/linear-regression`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        X,
        y,
        train_percent: trainPercent,
        feature_name: featureName,
        target_name: targetName
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    throw new Error(`API call failed: ${error.message}`);
  }
}

export async function trainMultiLinearRegression(X, y, trainPercent, featureNames = [], targetName = 'y') {
  try {
    const response = await fetch(`${API_BASE_URL}/multi-linear-regression`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        X,
        y,
        train_percent: trainPercent,
        feature_names: featureNames,
        target_name: targetName
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    throw new Error(`API call failed: ${error.message}`);
  }
}

export async function trainKNNRegression(X, y, trainPercent, k, distanceMetric, featureNames = [], targetName = 'y', minkowskiP = 3) {
  try {
    const response = await fetch(`${API_BASE_URL}/knn-regression`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        X,
        y,
        train_percent: trainPercent,
        k,
        distance_metric: distanceMetric,
        minkowski_p: minkowskiP,
        feature_names: featureNames,
        target_name: targetName
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    throw new Error(`API call failed: ${error.message}`);
  }
}

export async function trainKNNClassification(X, y, trainPercent, k, distanceMetric, featureNames = [], targetName = 'y', minkowskiP = 3) {
  try {
    const response = await fetch(`${API_BASE_URL}/knn-classification`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        X,
        y,
        train_percent: trainPercent,
        k,
        distance_metric: distanceMetric,
        minkowski_p: minkowskiP,
        feature_names: featureNames,
        target_name: targetName
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    throw new Error(`API call failed: ${error.message}`);
  }
}

export async function trainPolynomialRegression(X, y, trainPercent, degree, includeBias, interactionOnly, featureNames = [], targetName = 'y') {
  try {
    const response = await fetch(`${API_BASE_URL}/polynomial-regression`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        X,
        y,
        train_percent: trainPercent,
        degree,
        include_bias: includeBias,
        interaction_only: interactionOnly,
        feature_names: featureNames,
        target_name: targetName
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    throw new Error(`API call failed: ${error.message}`);
  }
}

export async function applyPCA(data, headers, config, fullRows = null, allHeaders = null, selectedIndices = null) {
  try {
    const requestBody = {
      data,
      headers,
      n_components: config.n_components,
      variance_threshold: config.variance_threshold,
      standardize: config.standardize !== undefined ? config.standardize : true,
      return_loadings: config.return_loadings || false,
      return_explained_variance: config.return_explained_variance !== undefined ? config.return_explained_variance : true
    };

    // Add optional full row data for propagating unselected columns
    if (fullRows !== null && allHeaders !== null && selectedIndices !== null) {
      requestBody.full_rows = fullRows;
      requestBody.all_headers = allHeaders;
      requestBody.selected_indices = selectedIndices;
    }

    const response = await fetch(`${API_BASE_URL}/pca`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    return result;
  } catch (error) {
    throw new Error(`API call failed: ${error.message}`);
  }
}

export async function checkApiHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      return false;
    }
    const data = await response.json();
    return data.status === 'ok';
  } catch (error) {
    return false;
  }
}
