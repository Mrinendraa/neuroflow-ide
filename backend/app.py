from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import math

app = Flask(__name__)
CORS(app)

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------

def train_test_split(X, y, test_size=0.2, random_state=None):
    """Robust train-test split implementation"""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = max(1, int(n_samples * test_size))  # Ensure at least 1 test sample
    n_train = n_samples - n_test
    
    if n_train < 1:
        raise ValueError("Not enough samples for training")
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

def add_intercept(X):
    """Add intercept term to features"""
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    return np.column_stack([np.ones(X.shape[0]), X])

def safe_float_conversion(arr):
    """Convert numpy array to list with NaN/Inf checking"""
    if arr is None:
        return []
    arr = np.array(arr)
    # Replace NaN and Inf with 0 or appropriate values
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=-1e10)
    # Convert to Python float types
    return [float(x) for x in arr]

def robust_feature_scaling(X):
    """Robust feature scaling that handles constant features"""
    if X.size == 0:
        return X, np.array([]), np.array([])
    
    X = np.array(X, dtype=float)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    # Handle constant features
    std = np.where(std < 1e-10, 1.0, std)
    
    X_scaled = (X - mean) / std
    return X_scaled, mean, std

def robust_target_scaling(y):
    """Scale target variable for better convergence"""
    y = np.array(y, dtype=float)
    y_mean = np.mean(y)
    y_std = np.std(y)
    
    if y_std < 1e-10:
        return y, 0.0, 1.0  # Constant target
    
    y_scaled = (y - y_mean) / y_std
    return y_scaled, y_mean, y_std

def generate_polynomial_features(X, degree=2, include_bias=True, interaction_only=False):
    """
    Generate polynomial features from input data
    
    Parameters:
    - X: Input features (numpy array), shape (n_samples, n_features)
    - degree: Polynomial degree (1-5)
    - include_bias: Whether to include bias term (constant 1)
    - interaction_only: If True, only generate interaction terms (no powers)
    
    Returns:
    - X_poly: Transformed feature matrix
    - feature_names: List of feature names for reference
    """
    X = np.array(X, dtype=float)
    
    # Handle 1D case
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples, n_features = X.shape
    
    # Start with bias if requested
    features = []
    feature_names = []
    
    if include_bias:
        features.append(np.ones(n_samples))
        feature_names.append('1')
    
    # Add original features (degree 1)
    for i in range(n_features):
        features.append(X[:, i])
        feature_names.append(f'x{i}')
    
    # Generate higher degree features
    if degree > 1:
        # For each degree from 2 to specified degree
        for d in range(2, degree + 1):
            # Generate all combinations of features with total degree = d
            from itertools import combinations_with_replacement
            
            for combo in combinations_with_replacement(range(n_features), d):
                # Check if this is an interaction term or a power term
                is_interaction = len(set(combo)) > 1
                is_power = len(set(combo)) == 1
                
                # Skip power terms if interaction_only is True
                if interaction_only and is_power:
                    continue
                
                # Compute the feature
                feature = np.ones(n_samples)
                name_parts = []
                for idx in combo:
                    feature *= X[:, idx]
                    name_parts.append(f'x{idx}')
                
                features.append(feature)
                
                # Create readable name
                if is_power:
                    feature_names.append(f'x{combo[0]}^{d}')
                else:
                    feature_names.append('*'.join(name_parts))
    
    # Stack all features
    X_poly = np.column_stack(features)
    
    return X_poly, feature_names

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def regression_metrics(y_true, y_pred):
    """Calculate regression metrics without sklearn"""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    # Replace any NaN values
    y_true = np.nan_to_num(y_true, nan=0.0)
    y_pred = np.nan_to_num(y_pred, nan=0.0)
    
    # Handle edge cases
    if len(y_true) == 0:
        return {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "r2_score": 0.0, "mape": 0.0}
    
    # MSE
    mse = float(np.mean((y_true - y_pred) ** 2))
    
    # RMSE
    rmse = float(np.sqrt(max(0, mse)))  # Ensure non-negative
    
    # MAE
    mae = float(np.mean(np.abs(y_true - y_pred)))
    
    # R² Score
    y_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    
    if abs(ss_tot) < 1e-10:
        r2 = 1.0 if abs(ss_res) < 1e-10 else 0.0
    else:
        r2 = max(-1.0, min(1.0, 1 - (ss_res / ss_tot)))  # Clamp between -1 and 1
    
    # MAPE
    denom = np.where(np.abs(y_true) < 1e-10, 1e-10, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2,
        "mape": mape
    }
def classification_metrics(y_true, y_pred):
    """Calculate classification metrics without sklearn"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Replace any NaN values
    y_true = np.nan_to_num(y_true, nan=0.0)
    y_pred = np.nan_to_num(y_pred, nan=0.0)
    
    # Ensure binary labels
    y_true = (y_true > 0.5).astype(int)
    y_pred = (y_pred > 0.5).astype(int)
    
    # Confusion matrix
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    }
# ------------------------------------------------------------
# ULTIMATE Linear Regression Implementation
# ------------------------------------------------------------

class LinearRegression:
    """ Linear Regression - Handles ALL edge cases with multiple optimization strategies"""
    
    def __init__(self, method='auto', learning_rate=0.01, n_iterations=10000, tol=1e-8):
        self.method = method  # 'normal', 'gradient', 'svd', 'auto'
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.weights = None
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
        self.is_fitted = False
    
    def _normal_equation(self, X, y):
        """Solve using normal equation with multiple fallbacks"""
        try:
            # Method 1: Standard normal equation
            return np.linalg.inv(X.T @ X) @ X.T @ y
        except np.linalg.LinAlgError:
            try:
                # Method 2: Moore-Penrose pseudoinverse
                return np.linalg.pinv(X.T @ X) @ X.T @ y
            except:
                try:
                    # Method 3: Direct pseudoinverse
                    return np.linalg.pinv(X) @ y
                except:
                    # Method 4: Ridge regression fallback
                    ridge_lambda = 1e-6
                    n_features = X.shape[1]
                    return np.linalg.inv(X.T @ X + ridge_lambda * np.eye(n_features)) @ X.T @ y
    
    def _gradient_descent(self, X, y):
        """Robust gradient descent with adaptive learning rate"""
        n_samples, n_features = X.shape
        
        # Initialize weights with small random values
        self.weights = np.random.normal(0, 0.01, n_features)
        
        best_weights = self.weights.copy()
        best_loss = float('inf')
        patience = 100
        patience_counter = 0
        
        for iteration in range(self.n_iterations):
            # Forward pass
            y_pred = X @ self.weights
            error = y_pred - y
            
            # Compute loss (MSE)
            loss = np.mean(error ** 2)
            
            # Check for improvement
            if loss < best_loss - self.tol:
                best_loss = loss
                best_weights = self.weights.copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                break
            
            # Compute gradients
            gradients = (2 / n_samples) * (X.T @ error)
            
            # Adaptive learning rate
            grad_norm = np.linalg.norm(gradients)
            if grad_norm > 1.0:
                gradients = gradients / grad_norm  # Normalize large gradients
            
            current_lr = self.learning_rate / (1 + 0.001 * iteration)  # Decay learning rate
            
            # Update weights
            self.weights -= current_lr * gradients
            
            # Check convergence
            if grad_norm < self.tol:
                break
        
        return best_weights
    
    def _svd_solution(self, X, y):
        """SVD-based solution for ill-conditioned problems"""
        try:
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            # Regularize small singular values
            s_inv = np.divide(1, s, out=np.zeros_like(s), where=np.abs(s) > 1e-10)
            return Vt.T @ np.diag(s_inv) @ U.T @ y
        except:
            # Fallback to pseudoinverse
            return np.linalg.pinv(X) @ y
    
    def fit(self, X, y):
        """Robust fitting that handles all edge cases"""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Handle 1D case
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Handle insufficient samples
        if n_samples < n_features:
            # Use gradient descent for underdetermined system
            self.method = 'gradient'
        
        # Store original data statistics
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.X_std = np.where(self.X_std < 1e-10, 1.0, self.X_std)
        
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        if self.y_std < 1e-10:
            self.y_std = 1.0
        
        # Scale features
        X_scaled = (X - self.X_mean) / self.X_std
        # Scale target
        y_scaled = (y - self.y_mean) / self.y_std
        
        # Add intercept
        X_with_intercept = add_intercept(X_scaled)
        
        # Choose method automatically if 'auto'
        actual_method = self.method
        if actual_method == 'auto':
            if n_samples > 1000 or n_samples < n_features:
                actual_method = 'gradient'
            else:
                actual_method = 'normal'
        
        # Fit using chosen method
        try:
            if actual_method == 'normal':
                self.weights = self._normal_equation(X_with_intercept, y_scaled)
            elif actual_method == 'gradient':
                self.weights = self._gradient_descent(X_with_intercept, y_scaled)
            elif actual_method == 'svd':
                self.weights = self._svd_solution(X_with_intercept, y_scaled)
            else:
                self.weights = self._normal_equation(X_with_intercept, y_scaled)
        except Exception as e:
            # Ultimate fallback - mean prediction
            print(f"All methods failed, using fallback: {e}")
            self.weights = np.zeros(X_with_intercept.shape[1])
            self.weights[0] = np.mean(y_scaled)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make robust predictions"""
        if not self.is_fitted or self.weights is None:
            raise ValueError("Model not fitted yet")
        
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Scale features
        X_scaled = (X - self.X_mean) / self.X_std
        # Add intercept
        X_with_intercept = add_intercept(X_scaled)
        
        # Predict scaled values
        y_pred_scaled = X_with_intercept @ self.weights
        
        # Scale back to original
        y_pred = y_pred_scaled * self.y_std + self.y_mean
        
        # Ensure no crazy values
        y_pred = np.clip(y_pred, -1e10, 1e10)  # Prevent extreme values
        y_pred = np.nan_to_num(y_pred, nan=self.y_mean)  # Replace NaN with mean
        
        return y_pred
    
    @property
    def coef_(self):
        """Get coefficients in original scale"""
        if self.weights is None or not self.is_fitted:
            return np.array([])
        
        # weights[1:] are coefficients for scaled features
        # Convert back to original scale
        coef = self.weights[1:] * self.y_std / self.X_std
        return coef
    
    @property
    def intercept_(self):
        """Get intercept in original scale"""
        if self.weights is None or not self.is_fitted:
            return 0.0
        
        # intercept = weights[0] * y_std + y_mean - sum(coef * X_mean / X_std)
        intercept_scaled = self.weights[0]
        intercept_original = intercept_scaled * self.y_std + self.y_mean
        
        if len(self.weights) > 1:
            # Adjust for feature scaling
            coef_scaled = self.weights[1:]
            adjustment = np.sum(coef_scaled * self.y_std * self.X_mean / self.X_std)
            intercept_original -= adjustment
        
        return intercept_original

# ------------------------------------------------------------
# Logistic Regression Implementation (Also Improved)
# ------------------------------------------------------------

def sigmoid(x):
    """Numerically stable sigmoid"""
    x = np.clip(x, -500, 500)  # Prevent overflow
    return np.where(x >= 0, 
                   1 / (1 + np.exp(-x)), 
                   np.exp(x) / (1 + np.exp(x)))

class LogisticRegression:
    """Logistic Regression implementation from scratch"""
    
    def __init__(self, learning_rate=0.1, n_iterations=10000, C=1.0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.C = C  # Regularization parameter
        self.weights = None
        self.X_mean = None
        self.X_std = None
    
    def fit(self, X, y):
        """Train using gradient descent with feature scaling"""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Add intercept and handle 1D case
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_original = X.copy()
        X = add_intercept(X)
        
        n_samples, n_features = X.shape
        
        # Feature scaling (crucial for logistic regression stability)
        if n_features > 1:
            self.X_mean = np.mean(X_original, axis=0)
            self.X_std = np.std(X_original, axis=0)
            self.X_std = np.where(self.X_std < 1e-10, 1.0, self.X_std)
            X[:, 1:] = (X[:, 1:] - self.X_mean) / self.X_std
        
        # Initialize weights
        self.weights = np.zeros(n_features)
        
        # Gradient descent
        for iteration in range(self.n_iterations):
            linear_model = X @ self.weights
            y_pred = sigmoid(linear_model)
            
            # Compute gradients with L2 regularization
            # Remove regularization from intercept
            regularization = np.zeros_like(self.weights)
            if len(self.weights) > 1:
                regularization[1:] = (1 / self.C) * self.weights[1:]
            
            dw = (1 / n_samples) * (X.T @ (y_pred - y) + regularization)
            
            # Update weights
            self.weights -= self.learning_rate * dw
            
            # Early stopping if gradients are very small
            if np.linalg.norm(dw) < 1e-8:
                break
    
    def predict_proba(self, X):
        """Predict probabilities"""
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_original = X.copy()
        X = add_intercept(X)
        
        # Apply same scaling if needed
        if self.X_mean is not None and self.X_std is not None:
            X[:, 1:] = (X[:, 1:] - self.X_mean) / self.X_std
        
        linear_model = X @ self.weights
        probabilities = sigmoid(linear_model)
        # Ensure no NaN values
        probabilities = np.nan_to_num(probabilities, nan=0.5)
        return np.clip(probabilities, 0.0, 1.0)
    
    def predict(self, X, threshold=0.5):
        """Make binary predictions"""
        probabilities = self.predict_proba(X)
        predictions = (probabilities >= threshold).astype(int)
        return predictions
    
    @property
    def coef_(self):
        if self.weights is None:
            return np.array([])
        
        coef = self.weights[1:].copy()
        # Adjust coefficients back to original scale
        if self.X_std is not None:
            coef = coef / self.X_std
        return coef
    
    @property
    def intercept_(self):
        if self.weights is None:
            return 0.0
        
        intercept = self.weights[0]
        # Adjust intercept back to original scale
        if self.X_mean is not None and self.X_std is not None:
            intercept = intercept - np.sum(self.weights[1:] * self.X_mean / self.X_std)
        return intercept

# ------------------------------------------------------------
# Polynomial Regression Implementation
# ------------------------------------------------------------

class PolynomialRegression:
    """
    Polynomial Regression using feature transformation + Linear Regression
    Transforms input features to polynomial space and fits a linear model
    """
    
    def __init__(self, degree=2, include_bias=True, interaction_only=False):
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.linear_model = LinearRegression()
        self.feature_names = None
        self.n_features_original = None
        self.n_features_poly = None
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Fit polynomial regression model
        
        Parameters:
        - X: Input features (n_samples, n_features)
        - y: Target values (n_samples,)
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Handle 1D case
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_features_original = X.shape[1]
        
        # Generate polynomial features
        X_poly, self.feature_names = generate_polynomial_features(
            X, 
            degree=self.degree, 
            include_bias=self.include_bias,
            interaction_only=self.interaction_only
        )
        
        self.n_features_poly = X_poly.shape[1]
        
        # Fit linear regression on polynomial features
        # Note: LinearRegression will add its own intercept, so we need to handle this
        # If we already included bias in polynomial features, don't let LinearRegression add another
        self.linear_model.fit(X_poly, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Make predictions using polynomial regression
        
        Parameters:
        - X: Input features (n_samples, n_features)
        
        Returns:
        - y_pred: Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X = np.array(X, dtype=float)
        
        # Handle 1D case
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Generate polynomial features (same transformation as training)
        X_poly, _ = generate_polynomial_features(
            X, 
            degree=self.degree, 
            include_bias=self.include_bias,
            interaction_only=self.interaction_only
        )
        
        # Predict using linear model
        y_pred = self.linear_model.predict(X_poly)
        
        return y_pred
    
    @property
    def coef_(self):
        """Get coefficients for polynomial features"""
        if not self.is_fitted or self.linear_model.coef_ is None:
            return np.array([])
        return self.linear_model.coef_
    
    @property
    def intercept_(self):
        """Get intercept term"""
        if not self.is_fitted:
            return 0.0
        return self.linear_model.intercept_

# ------------------------------------------------------------
# PCA Implementation
# ------------------------------------------------------------

class PCA:
    """
    Principal Component Analysis implementation from scratch
    Supports both explicit component count and variance-based selection
    """
    
    def __init__(self, n_components=None, variance_threshold=None, standardize=True):
        """
        Initialize PCA
        
        Parameters:
        - n_components: Explicit number of components (1 to N)
        - variance_threshold: Variance retention threshold (0.0 to 1.0)
        - standardize: Whether to standardize data before PCA
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.standardize = standardize
        
        # Fitted attributes
        self.mean_ = None
        self.std_ = None
        self.components_ = None  # Principal components (eigenvectors)
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.n_components_ = None  # Actual number of components used
        self.is_fitted = False
    
    def fit(self, X):
        """
        Fit PCA on data X
        
        Parameters:
        - X: Input data (n_samples, n_features)
        
        Returns:
        - self
        """
        X = np.array(X, dtype=float)
        
        # Handle 1D case
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Standardize data if requested
        if self.standardize:
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
            # Handle constant features
            self.std_ = np.where(self.std_ < 1e-10, 1.0, self.std_)
            X_centered = (X - self.mean_) / self.std_
        else:
            self.mean_ = np.mean(X, axis=0)
            X_centered = X - self.mean_
            self.std_ = np.ones(n_features)
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Ensure eigenvalues are non-negative (numerical stability)
        eigenvalues = np.maximum(eigenvalues, 0)
        
        # Calculate explained variance
        total_variance = np.sum(eigenvalues)
        if total_variance < 1e-10:
            # All features are constant
            self.explained_variance_ratio_ = np.zeros(n_features)
        else:
            self.explained_variance_ratio_ = eigenvalues / total_variance
        
        self.explained_variance_ = eigenvalues
        
        # Determine number of components
        if self.variance_threshold is not None:
            # Use variance threshold
            cumulative_variance = np.cumsum(self.explained_variance_ratio_)
            self.n_components_ = np.argmax(cumulative_variance >= self.variance_threshold) + 1
            # Ensure at least 1 component
            self.n_components_ = max(1, self.n_components_)
        elif self.n_components is not None:
            # Use explicit component count
            self.n_components_ = min(self.n_components, n_features)
        else:
            # Default: use all components
            self.n_components_ = n_features
        
        # Store principal components (eigenvectors)
        self.components_ = eigenvectors[:, :self.n_components_]
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """
        Transform data using fitted PCA
        
        Parameters:
        - X: Input data (n_samples, n_features)
        
        Returns:
        - X_transformed: Transformed data (n_samples, n_components)
        """
        if not self.is_fitted:
            raise ValueError("PCA not fitted yet. Call fit() first.")
        
        X = np.array(X, dtype=float)
        
        # Handle 1D case
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Standardize using fitted parameters
        if self.standardize:
            X_centered = (X - self.mean_) / self.std_
        else:
            X_centered = X - self.mean_
        
        # Project onto principal components
        X_transformed = X_centered @ self.components_
        
        return X_transformed
    
    def fit_transform(self, X):
        """
        Fit PCA and transform data in one step
        
        Parameters:
        - X: Input data (n_samples, n_features)
        
        Returns:
        - X_transformed: Transformed data (n_samples, n_components)
        """
        self.fit(X)
        return self.transform(X)
    
    def get_loadings(self):
        """
        Get component loadings (correlation between original features and PCs)
        
        Returns:
        - loadings: Component loadings matrix (n_features, n_components)
        """
        if not self.is_fitted:
            raise ValueError("PCA not fitted yet. Call fit() first.")
        
        # Loadings are the eigenvectors scaled by sqrt of eigenvalues
        loadings = self.components_ * np.sqrt(self.explained_variance_[:self.n_components_])
        return loadings

# ------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------

@app.route('/api/logistic-regression', methods=['POST'])
def api_logistic_regression():
    try:
        data = request.json
        X = np.array(data["X"], dtype=float)
        y = np.array(data["y"])

        train_percent = data.get("train_percent", 80)
        test_size = 1 - train_percent/100

        # Convert labels to 0/1 if not already
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            return jsonify({"error": "Logistic regression requires binary labels"}), 400

        # Map smallest label → 0, largest → 1
        if not np.array_equal(unique_labels, [0, 1]):
            low = unique_labels.min()
            y = (y != low).astype(int)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Model (custom implementation)
        model = LogisticRegression(C=data.get("C", 1.0))
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)

        return jsonify({
            "success": True,
            "coefficients": safe_float_conversion(model.coef_),
            "intercept": float(model.intercept_),
            "train_metrics": classification_metrics(y_train, y_train_pred),
            "test_metrics": classification_metrics(y_test, y_test_pred),
            "train_predictions": safe_float_conversion(y_train_pred),
            "test_predictions": safe_float_conversion(y_test_pred),
            "test_probabilities": safe_float_conversion(y_test_proba),
            "train_size": len(X_train),
            "test_size": len(X_test)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/linear-regression", methods=["POST"])
def api_linear_regression():
    try:
        data = request.json
        X = np.array(data["X"], dtype=float)
        y = np.array(data["y"], dtype=float)

        train_percent = data.get("train_percent", 80)
        test_size = max(0.1, min(0.5, 1 - train_percent/100))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        return jsonify({
            "success": True,
            "slope": float(model.coef_[0]) if len(model.coef_) > 0 else 0.0,
            "intercept": float(model.intercept_),
            "train_metrics": regression_metrics(y_train, y_train_pred),
            "test_metrics": regression_metrics(y_test, y_test_pred),
            "train_predictions": safe_float_conversion(y_train_pred),
            "test_predictions": safe_float_conversion(y_test_pred),
            "train_size": len(X_train),
            "test_size": len(X_test)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/multi-linear-regression", methods=["POST"])
def api_multi_linear_regression():
    try:
        data = request.json
        X = np.array(data["X"], dtype=float)
        y = np.array(data["y"], dtype=float)

        train_percent = data.get("train_percent", 80)
        test_size = max(0.1, min(0.5, 1 - train_percent/100))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        return jsonify({
            "success": True,
            "coefficients": safe_float_conversion(model.coef_),
            "intercept": float(model.intercept_),
            "train_metrics": regression_metrics(y_train, y_train_pred),
            "test_metrics": regression_metrics(y_test, y_test_pred),
            "train_predictions": safe_float_conversion(y_train_pred),
            "test_predictions": safe_float_conversion(y_test_pred),
            "train_size": len(X_train),
            "test_size": len(X_test)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/polynomial-regression", methods=["POST"])
def api_polynomial_regression():
    try:
        data = request.json
        X = np.array(data["X"], dtype=float)
        y = np.array(data["y"], dtype=float)

        train_percent = data.get("train_percent", 80)
        degree = data.get("degree", 2)
        include_bias = data.get("include_bias", True)
        interaction_only = data.get("interaction_only", False)
        
        # Validate degree
        if degree < 1 or degree > 5:
            return jsonify({"error": "Degree must be between 1 and 5"}), 400

        test_size = max(0.1, min(0.5, 1 - train_percent/100))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = PolynomialRegression(
            degree=degree,
            include_bias=include_bias,
            interaction_only=interaction_only
        )
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        return jsonify({
            "success": True,
            "coefficients": safe_float_conversion(model.coef_),
            "intercept": float(model.intercept_),
            "degree": degree,
            "include_bias": include_bias,
            "interaction_only": interaction_only,
            "n_features_original": model.n_features_original,
            "n_features_poly": model.n_features_poly,
            "feature_names": model.feature_names,
            "train_metrics": regression_metrics(y_train, y_train_pred),
            "test_metrics": regression_metrics(y_test, y_test_pred),
            "train_predictions": safe_float_conversion(y_train_pred),
            "test_predictions": safe_float_conversion(y_test_pred),
            "train_size": len(X_train),
            "test_size": len(X_test)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------
# KNN Regression Implementation
# ------------------------------------------------------------

def euclidean_distance(x1, x2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    """Calculate Manhattan distance between two points"""
    return np.sum(np.abs(x1 - x2))

def minkowski_distance(x1, x2, p=3):
    """Calculate Minkowski distance between two points"""
    return np.power(np.sum(np.abs(x1 - x2) ** p), 1/p)

def chebyshev_distance(x1, x2):
    """Calculate Chebyshev distance between two points"""
    return np.max(np.abs(x1 - x2))

def cosine_similarity_distance(x1, x2):
    """Calculate Cosine Similarity distance between two points"""
    dot_product = np.dot(x1, x2)
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)
    
    if norm_x1 == 0 or norm_x2 == 0:
        return 1.0  # Maximum distance if either vector is zero
    
    cosine_sim = dot_product / (norm_x1 * norm_x2)
    # Convert similarity to distance (1 - similarity)
    return 1 - cosine_sim

class KNNRegressor:
    """K-Nearest Neighbors Regressor with multiple distance metrics"""
    
    def __init__(self, k=5, distance_metric='euclidean', minkowski_p=3):
        self.k = k
        self.distance_metric = distance_metric
        self.minkowski_p = minkowski_p
        self.X_train = None
        self.y_train = None
        self.X_mean = None
        self.X_std = None
        
        # Map metric names to functions
        self.distance_functions = {
            'euclidean': euclidean_distance,
            'manhattan': manhattan_distance,
            'minkowski': lambda x1, x2: minkowski_distance(x1, x2, p=self.minkowski_p),
            'chebyshev': chebyshev_distance,
            'cosine': cosine_similarity_distance
        }
    
    def fit(self, X, y):
        """Store training data and compute scaling parameters"""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Handle 1D case
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Feature scaling for better distance calculations
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.X_std = np.where(self.X_std < 1e-10, 1.0, self.X_std)
        
        # Scale features
        self.X_train = (X - self.X_mean) / self.X_std
        self.y_train = y
        
        return self
    
    def predict(self, X):
        """Predict using KNN"""
        X = np.array(X, dtype=float)
        
        # Handle 1D case
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Scale features using training statistics
        X_scaled = (X - self.X_mean) / self.X_std
        
        predictions = []
        distance_func = self.distance_functions.get(self.distance_metric, euclidean_distance)
        
        for x in X_scaled:
            # Calculate distances to all training points
            distances = []
            for x_train in self.X_train:
                dist = distance_func(x, x_train)
                distances.append(dist)
            
            distances = np.array(distances)
            
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            
            # Get corresponding y values
            k_nearest_labels = self.y_train[k_indices]
            
            # Predict as mean of k nearest neighbors
            prediction = np.mean(k_nearest_labels)
            predictions.append(prediction)
        
        return np.array(predictions)

@app.route("/api/knn-regression", methods=["POST"])
def api_knn_regression():
    try:
        data = request.json
        X = np.array(data["X"], dtype=float)
        y = np.array(data["y"], dtype=float)

        train_percent = data.get("train_percent", 80)
        k = data.get("k", 5)
        distance_metric = data.get("distance_metric", "euclidean")
        
        # Validate k
        if k < 1:
            return jsonify({"error": "k must be at least 1"}), 400
        
        # Validate distance metric
        valid_metrics = ['euclidean', 'manhattan', 'minkowski', 'chebyshev', 'cosine']
        if distance_metric not in valid_metrics:
            return jsonify({"error": f"Invalid distance metric. Must be one of: {valid_metrics}"}), 400

        test_size = max(0.1, min(0.5, 1 - train_percent/100))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Ensure k is not larger than training set
        k_actual = min(k, len(X_train))
        if k_actual < k:
            print(f"Warning: k={k} is larger than training set size. Using k={k_actual}")

        model = KNNRegressor(k=k_actual, distance_metric=distance_metric, minkowski_p=minkowski_p)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        return jsonify({
            "success": True,
            "k": k_actual,
            "distance_metric": distance_metric,
            "train_metrics": regression_metrics(y_train, y_train_pred),
            "test_metrics": regression_metrics(y_test, y_test_pred),
            "train_predictions": safe_float_conversion(y_train_pred),
            "test_predictions": safe_float_conversion(y_test_pred),
            "train_size": len(X_train),
            "test_size": len(X_test)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------
# KNN Classification Implementation
# ------------------------------------------------------------

class KNNClassifier:
    """K-Nearest Neighbors Classifier with multiple distance metrics"""
    
    def __init__(self, k=5, distance_metric='euclidean', minkowski_p=3):
        self.k = k
        self.distance_metric = distance_metric
        self.minkowski_p = minkowski_p
        self.X_train = None
        self.y_train = None
        self.X_mean = None
        self.X_std = None
        self.classes = None
        
        # Map metric names to functions
        self.distance_functions = {
            'euclidean': euclidean_distance,
            'manhattan': manhattan_distance,
            'minkowski': lambda x1, x2: minkowski_distance(x1, x2, p=self.minkowski_p),
            'chebyshev': chebyshev_distance,
            'cosine': cosine_similarity_distance
        }
    
    def fit(self, X, y):
        """Store training data and compute scaling parameters"""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Handle 1D case
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Store unique classes
        self.classes = np.unique(y)
        
        # Feature scaling for better distance calculations
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.X_std = np.where(self.X_std < 1e-10, 1.0, self.X_std)
        
        # Scale features
        self.X_train = (X - self.X_mean) / self.X_std
        self.y_train = y
        
        return self
    
    def predict(self, X):
        """Predict using KNN classification (majority vote)"""
        X = np.array(X, dtype=float)
        
        # Handle 1D case
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Scale features using training statistics
        X_scaled = (X - self.X_mean) / self.X_std
        
        predictions = []
        distance_func = self.distance_functions.get(self.distance_metric, euclidean_distance)
        
        for x in X_scaled:
            # Calculate distances to all training points
            distances = []
            for x_train in self.X_train:
                dist = distance_func(x, x_train)
                distances.append(dist)
            
            distances = np.array(distances)
            
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            
            # Get corresponding y values (classes)
            k_nearest_labels = self.y_train[k_indices]
            
            # Majority vote - find most common class
            unique, counts = np.unique(k_nearest_labels, return_counts=True)
            prediction = unique[np.argmax(counts)]
            predictions.append(prediction)
        
        return np.array(predictions)

@app.route("/api/knn-classification", methods=["POST"])
def api_knn_classification():
    try:
        data = request.json
        X = np.array(data["X"], dtype=float)
        y = np.array(data["y"], dtype=float)

        train_percent = data.get("train_percent", 80)
        k = data.get("k", 5)
        distance_metric = data.get("distance_metric", "euclidean")
        minkowski_p = data.get("minkowski_p", 3)
        
        # Validate k
        if k < 1:
            return jsonify({"error": "k must be at least 1"}), 400
        
        # Validate distance metric
        valid_metrics = ['euclidean', 'manhattan', 'minkowski', 'chebyshev', 'cosine']
        if distance_metric not in valid_metrics:
            return jsonify({"error": f"Invalid distance metric. Must be one of: {valid_metrics}"}), 400

        test_size = max(0.1, min(0.5, 1 - train_percent/100))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Ensure k is not larger than training set
        k_actual = min(k, len(X_train))
        if k_actual < k:
            print(f"Warning: k={k} is larger than training set size. Using k={k_actual}")

        model = KNNClassifier(k=k_actual, distance_metric=distance_metric, minkowski_p=minkowski_p)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        return jsonify({
            "success": True,
            "k": k_actual,
            "distance_metric": distance_metric,
            "classes": [int(c) for c in model.classes],
            "train_metrics": classification_metrics(y_train, y_train_pred),
            "test_metrics": classification_metrics(y_test, y_test_pred),
            "train_predictions": safe_float_conversion(y_train_pred),
            "test_predictions": safe_float_conversion(y_test_pred),
            "train_size": len(X_train),
            "test_size": len(X_test)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/pca", methods=["POST"])
def api_pca():
    """
    PCA endpoint for dimensionality reduction
    
    Request body:
    - data: 2D array of numeric features (selected columns only)
    - headers: Column names for selected features
    - full_rows: Optional full row data including unselected columns
    - all_headers: Optional all column headers
    - selected_indices: Optional indices of selected columns in full data
    - n_components: Optional explicit component count
    - variance_threshold: Optional variance retention (0.0-1.0)
    - standardize: Whether to standardize data (default: true)
    - return_loadings: Whether to return component loadings (default: false)
    - return_explained_variance: Whether to return variance details (default: true)
    """
    try:
        request_data = request.json
        
        # Extract data and configuration
        data = np.array(request_data["data"], dtype=float)
        headers = request_data.get("headers", [])
        full_rows = request_data.get("full_rows", None)
        all_headers = request_data.get("all_headers", None)
        selected_indices = request_data.get("selected_indices", None)
        n_components = request_data.get("n_components", None)
        variance_threshold = request_data.get("variance_threshold", None)
        standardize = request_data.get("standardize", True)
        return_loadings = request_data.get("return_loadings", False)
        return_explained_variance = request_data.get("return_explained_variance", True)
        
        # Validate inputs
        if data.size == 0:
            return jsonify({"error": "No data provided"}), 400
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_samples, n_features = data.shape
        
        # Validate component count
        if n_components is not None:
            if n_components < 1 or n_components > n_features:
                return jsonify({
                    "error": f"n_components must be between 1 and {n_features}"
                }), 400
        
        # Validate variance threshold
        if variance_threshold is not None:
            if variance_threshold < 0.0 or variance_threshold > 1.0:
                return jsonify({
                    "error": "variance_threshold must be between 0.0 and 1.0"
                }), 400
        
        # Create and fit PCA
        pca = PCA(
            n_components=n_components,
            variance_threshold=variance_threshold,
            standardize=standardize
        )
        
        # Transform data
        transformed_data = pca.fit_transform(data)
        
        # Generate component headers (PC1, PC2, PC3, ...)
        component_headers = [f"PC{i+1}" for i in range(pca.n_components_)]
        
        # Combine transformed data with unselected columns if provided
        if full_rows is not None and all_headers is not None and selected_indices is not None:
            full_rows_array = np.array(full_rows)
            selected_indices_set = set(selected_indices)
            
            # Get unselected column indices
            unselected_indices = [i for i in range(len(all_headers)) if i not in selected_indices_set]
            
            if len(unselected_indices) > 0:
                # Extract unselected columns
                unselected_data = full_rows_array[:, unselected_indices]
                unselected_headers = [all_headers[i] for i in unselected_indices]
                
                # Combine: unselected columns first, then PCA components
                combined_data = np.column_stack([unselected_data, transformed_data])
                combined_headers = unselected_headers + component_headers
            else:
                # No unselected columns, just use transformed data
                combined_data = transformed_data
                combined_headers = component_headers
        else:
            # No full row data provided, just use transformed data
            combined_data = transformed_data
            combined_headers = component_headers
        
        # Prepare response
        response = {
            "success": True,
            "transformed_data": combined_data.tolist(),
            "component_headers": combined_headers,
            "n_components_used": int(pca.n_components_),
            "n_features_original": int(n_features),
            "standardized": standardize
        }
        
        # Add explained variance if requested
        if return_explained_variance:
            explained_var = pca.explained_variance_ratio_[:pca.n_components_]
            cumulative_var = np.cumsum(explained_var)
            
            response["explained_variance"] = safe_float_conversion(explained_var)
            response["cumulative_variance"] = safe_float_conversion(cumulative_var)
            response["total_variance_explained"] = float(cumulative_var[-1]) if len(cumulative_var) > 0 else 0.0
        
        # Add component loadings if requested
        if return_loadings:
            loadings = pca.get_loadings()
            response["loadings"] = loadings.tolist()
            response["original_features"] = headers if headers else [f"Feature_{i+1}" for i in range(n_features)]
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "ml-api", "implementation": "ULTIMATE"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)