"""
Traditional time series models for TRM forecasting
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TraditionalModels:
    """Traditional time series forecasting models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.model_path = Path("models/traditional")
        self.model_path.mkdir(parents=True, exist_ok=True)
    
    def fit_arima(self, series: np.ndarray, order: tuple = (5,1,0)):
        """Fit ARIMA model"""
        try:
            model = ARIMA(series, order=order)
            fitted = model.fit()
            self.models['arima'] = fitted
            return fitted
        except Exception as e:
            logger.error(f"ARIMA fitting failed: {e}")
            return None
    
    def fit_sarima(self, series: np.ndarray, order: tuple = (1,1,1), 
                   seasonal_order: tuple = (1,1,1,7)):
        """Fit SARIMA model with weekly seasonality"""
        try:
            model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
            fitted = model.fit(disp=False)
            self.models['sarima'] = fitted
            return fitted
        except Exception as e:
            logger.error(f"SARIMA fitting failed: {e}")
            return None
    
    def fit_ets(self, series: pd.Series, seasonal_periods: int = 7):
        """Fit Exponential Smoothing model"""
        try:
            model = ExponentialSmoothing(series, seasonal_periods=seasonal_periods, 
                                       trend='add', seasonal='add')
            fitted = model.fit()
            self.models['ets'] = fitted
            return fitted
        except Exception as e:
            logger.error(f"ETS fitting failed: {e}")
            return None
    
    def fit_prophet(self, df: pd.DataFrame):
        """Fit Prophet model with additional regressors"""
        try:
            # Prepare data for Prophet
            prophet_df = pd.DataFrame()
            prophet_df['ds'] = df['date']
            prophet_df['y'] = df['trm']
            
            # Add additional regressors
            prophet_df['open_rate'] = df['open_rate']
            prophet_df['close_rate'] = df['close_rate']
            prophet_df['num_trades'] = df['num_trades']
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            
            # Add regressors
            model.add_regressor('open_rate')
            model.add_regressor('close_rate')
            model.add_regressor('num_trades')
            
            fitted = model.fit(prophet_df)
            self.models['prophet'] = fitted
            return fitted
            
        except Exception as e:
            logger.error(f"Prophet fitting failed: {e}")
            return None
    
    def forecast(self, model_name: str, steps: int = 30) -> np.ndarray:
        """Generate forecasts from a specific model"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        
        try:
            if model_name == 'prophet':
                future = model.make_future_dataframe(periods=steps)
                forecast = model.predict(future)
                return forecast['yhat'].values[-steps:]
            else:
                forecast = model.forecast(steps=steps)
                return forecast
        except Exception as e:
            logger.error(f"Forecasting failed for {model_name}: {e}")
            return None
    
    def evaluate_models(self, test_series: np.ndarray) -> dict:
        """Evaluate all trained models"""
        metrics = {}
        
        for model_name, model in self.models.items():
            try:
                predictions = self.forecast(model_name, steps=len(test_series))
                if predictions is not None:
                    metrics[model_name] = {
                        'MAE': mean_absolute_error(test_series, predictions),
                        'RMSE': np.sqrt(mean_squared_error(test_series, predictions)),
                        'MAPE': mean_absolute_percentage_error(test_series, predictions)
                    }
            except Exception as e:
                logger.error(f"Evaluation failed for {model_name}: {e}")
        
        return metrics
    
    def save_models(self):
        """Save all trained models"""
        for name, model in self.models.items():
            joblib.dump(model, self.model_path / f"{name}.pkl")
        logger.info("All traditional models saved")
    
    def load_models(self):
        """Load saved models"""
        for model_file in self.model_path.glob("*.pkl"):
            name = model_file.stem
            self.models[name] = joblib.load(model_file)
        logger.info(f"Loaded {len(self.models)} traditional models")

if __name__ == "__main__":
    # Test traditional models
    df = pd.read_csv("data/processed/trm_historical.csv")
    series = df['trm'].values
    
    traditional = TraditionalModels()
    traditional.fit_arima(series)
    traditional.fit_sarima(series)
    
    # Evaluate
    test_size = 30
    train_series = series[:-test_size]
    test_series = series[-test_size:]
    
    traditional.fit_arima(train_series)
    metrics = traditional.evaluate_models(test_series)
    print(f"Model metrics: {metrics}")