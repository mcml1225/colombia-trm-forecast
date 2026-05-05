"""
Ensemble model combining traditional and AI predictions
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class EnsembleForecaster:
    """Ensemble model that combines multiple forecasting methods"""
    
    def __init__(self):
        self.meta_learner = None
        self.base_predictions = {}
        self.weights = None
        self.model_path = Path("models/ensemble")
        self.model_path.mkdir(parents=True, exist_ok=True)
    
    def combine_predictions(self, predictions_dict: dict, method: str = 'weighted_average'):
        """
        Combine predictions from multiple models
        
        Args:
            predictions_dict: Dictionary with model names as keys and predictions as values
            method: 'weighted_average', 'median', 'stacking'
        """
        pred_df = pd.DataFrame(predictions_dict)
        
        if method == 'weighted_average':
            if self.weights is None:
                # Equal weights if not specified
                self.weights = np.ones(len(pred_df.columns)) / len(pred_df.columns)
            combined = np.average(pred_df.values, weights=self.weights, axis=1)
            
        elif method == 'median':
            combined = pred_df.median(axis=1).values
            
        elif method == 'stacking':
            # Use meta-learner for stacking
            if self.meta_learner is None:
                self.meta_learner = GradientBoostingRegressor(n_estimators=100, random_state=42)
            combined = self.meta_learner.predict(pred_df.values)
            
        else:
            combined = pred_df.mean(axis=1).values
        
        return combined
    
    def optimize_weights(self, predictions_dict: dict, y_true: np.ndarray):
        """Optimize weights for weighted average combination"""
        from scipy.optimize import minimize
        
        pred_matrix = np.array(list(predictions_dict.values())).T
        
        def objective(weights):
            combined = np.dot(pred_matrix, weights)
            return mean_absolute_error(y_true, combined)
        
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(len(predictions_dict))]
        initial_weights = np.ones(len(predictions_dict)) / len(predictions_dict)
        
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        self.weights = result.x
        return self.weights
    
    def train_meta_learner(self, predictions_dict: dict, y_true: np.ndarray):
        """Train meta-learner for stacking"""
        pred_matrix = np.array(list(predictions_dict.values())).T
        
        # Use Random Forest as meta-learner
        self.meta_learner = RandomForestRegressor(n_estimators=100, 
                                                  max_depth=5, 
                                                  random_state=42)
        self.meta_learner.fit(pred_matrix, y_true)
        
        # Evaluate feature importance
        feature_importance = dict(zip(predictions_dict.keys(), 
                                     self.meta_learner.feature_importances_))
        logger.info(f"Meta-learner feature importance: {feature_importance}")
        
        return self.meta_learner
    
    def save_ensemble(self):
        """Save ensemble model components"""
        if self.meta_learner is not None:
            joblib.dump(self.meta_learner, self.model_path / "meta_learner.pkl")
        if self.weights is not None:
            joblib.dump(self.weights, self.model_path / "weights.pkl")
        logger.info("Ensemble model saved")
    
    def load_ensemble(self):
        """Load ensemble model components"""
        meta_path = self.model_path / "meta_learner.pkl"
        weights_path = self.model_path / "weights.pkl"
        
        if meta_path.exists():
            self.meta_learner = joblib.load(meta_path)
        if weights_path.exists():
            self.weights = joblib.load(weights_path)
        logger.info("Ensemble model loaded")
    
    def calculate_ensemble_metrics(self, predictions_dict: dict, y_true: np.ndarray) -> dict:
        """Calculate metrics for ensemble combinations"""
        ensemble_methods = ['mean', 'median', 'weighted_average', 'stacking']
        metrics = {}
        
        for method in ensemble_methods:
            combined = self.combine_predictions(predictions_dict, method)
            metrics[method] = {
                'MAE': mean_absolute_error(y_true, combined),
                'RMSE': np.sqrt(mean_squared_error(y_true, combined)),
                'MAPE': mean_absolute_percentage_error(y_true, combined)
            }
        
        return metrics

if __name__ == "__main__":
    # Test ensemble
    ensemble = EnsembleForecaster()
    
    # Simulate predictions from different models
    np.random.seed(42)
    y_true = np.random.randn(100)
    predictions = {
        'arima': y_true + np.random.randn(100) * 0.1,
        'lstm': y_true + np.random.randn(100) * 0.15,
        'prophet': y_true + np.random.randn(100) * 0.12
    }
    
    # Optimize weights
    weights = ensemble.optimize_weights(predictions, y_true)
    print(f"Optimized weights: {dict(zip(predictions.keys(), weights))}")
    
    # Train meta-learner
    ensemble.train_meta_learner(predictions, y_true)
    
    # Evaluate ensemble
    metrics = ensemble.calculate_ensemble_metrics(predictions, y_true)
    print(f"Ensemble metrics: {metrics}")