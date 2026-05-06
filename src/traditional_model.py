"""
Traditional time series models for TRM forecasting
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib
from pathlib import Path
import logging
import os
import warnings
warnings.filterwarnings('ignore')

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
    
    def forecast(self, model_name: str, steps: int = 30) -> np.ndarray:
        """Generate forecasts from a specific model"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        
        try:
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


def create_sample_data_if_needed():
    """Create sample data if the CSV file doesn't exist"""
    data_path = Path("data/processed/trm_historical.csv")
    
    if not data_path.exists():
        print("Data file not found. Creating sample data...")
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        
        from datetime import datetime, timedelta
        
        dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
        base_trm = 3600
        trend = np.linspace(0, 100, 200)
        seasonality = 50 * np.sin(2 * np.pi * np.arange(200) / 30)
        noise = np.random.normal(0, 15, 200)
        trm_values = base_trm + trend + seasonality + noise
        
        df = pd.DataFrame({
            'date': dates,
            'trm': trm_values,
            'open_rate': trm_values + np.random.normal(0, 5, 200),
            'close_rate': trm_values + np.random.normal(0, 5, 200),
            'max_rate': trm_values + np.random.uniform(5, 20, 200),
            'min_rate': trm_values - np.random.uniform(5, 20, 200),
            'num_trades': np.random.randint(1000, 5000, 200)
        })
        df.to_csv(data_path, index=False)
        print(f"Sample data created with {len(df)} rows at {data_path}")
    else:
        print(f"Data file found at {data_path}")


if __name__ == "__main__":
    # Crear datos si no existen
    create_sample_data_if_needed()
    
    # Cargar datos
    df = pd.read_csv("data/processed/trm_historical.csv")
    print(f"Data loaded: {len(df)} rows")
    
    # Verificar columnas
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Usar la columna TRM
    series = df['trm'].values
    print(f"Series length: {len(series)}")
    print(f"TRM range: {series.min():.2f} - {series.max():.2f}")
    
    # Dividir datos
    train_size = len(series) - 30
    train_series = series[:train_size]
    test_series = series[train_size:]
    
    print(f"Train size: {len(train_series)}, Test size: {len(test_series)}")
    
    # Probar modelos
    traditional = TraditionalModels()
    
    print("\nTraining ARIMA model...")
    arima_model = traditional.fit_arima(train_series, order=(3,1,0))
    if arima_model:
        print("ARIMA model trained successfully")
        forecast = traditional.forecast('arima', steps=len(test_series))
        if forecast is not None:
            print(f"ARIMA forecast generated: {len(forecast)} steps")
            mae = np.mean(np.abs(test_series - forecast))
            print(f"ARIMA Test MAE: {mae:.2f}")
    else:
        print("ARIMA model training failed")
    
    print("\nDone!")
