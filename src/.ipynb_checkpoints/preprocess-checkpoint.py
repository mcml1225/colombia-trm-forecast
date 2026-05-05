"""
Data preprocessing module for TRM forecasting
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class TRMPreprocessor:
    """Preprocesses TRM data for time series forecasting"""
    
    def __init__(self, scaler_type='minmax'):
        self.scaler_type = scaler_type
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            self.scaler = StandardScaler()
        
        self.feature_columns = ['trm', 'open_rate', 'weighted_avg', 
                                'close_rate', 'max_rate', 'min_rate', 'num_trades']
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = df.copy()
        
        # Lag features
        for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
            df[f'trm_lag_{lag}'] = df['trm'].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 21, 30]:
            df[f'trm_rolling_mean_{window}'] = df['trm'].rolling(window=window).mean()
            df[f'trm_rolling_std_{window}'] = df['trm'].rolling(window=window).std()
        
        # Date features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Time features cyclic encoding
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Price differences
        df['trm_diff_1'] = df['trm'].diff(1)
        df['trm_diff_7'] = df['trm'].diff(7)
        df['trm_pct_change'] = df['trm'].pct_change()
        
        # Volatility
        df['trm_volatility'] = df['trm'].rolling(window=7).std()
        
        # Target for prediction (next day's TRM)
        df['target_trm'] = df['trm'].shift(-1)
        
        return df
    
    def prepare_sequence_data(self, df: pd.DataFrame, sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequence data for LSTM model"""
        # Select features for LSTM
        lstm_features = ['trm', 'open_rate', 'weighted_avg', 'close_rate', 
                         'max_rate', 'min_rate', 'trm_rolling_mean_7', 
                         'trm_rolling_std_7', 'trm_diff_1', 'trm_pct_change']
        
        # Scale data
        scaled_data = self.scaler.fit_transform(df[lstm_features].fillna(method='bfill'))
        
        X, y = [], []
        for i in range(sequence_length, len(scaled_data) - 1):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, 0])  # Predict TRM
        
        return np.array(X), np.array(y)
    
    def prepare_traditional_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for traditional time series models"""
        # Use only closing TRM values
        series = df['trm'].values.reshape(-1, 1)
        return self.scaler.fit_transform(series), self.scaler
    
    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                   val_size: float = 0.1) -> Dict[str, np.ndarray]:
        """Split data into train, validation, and test sets"""
        total_samples = len(X)
        test_samples = int(total_samples * test_size)
        val_samples = int(total_samples * val_size)
        train_samples = total_samples - test_samples - val_samples
        
        X_train = X[:train_samples]
        y_train = y[:train_samples]
        
        X_val = X[train_samples:train_samples+val_samples]
        y_val = y[train_samples:train_samples+val_samples]
        
        X_test = X[train_samples+val_samples:]
        y_test = y[train_samples+val_samples:]
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Inverse transform scaled predictions"""
        return self.scaler.inverse_transform(predictions.reshape(-1, 1))

if __name__ == "__main__":
    # Test preprocessing
    import pandas as pd
    df = pd.read_csv("data/processed/trm_historical.csv")
    preprocessor = TRMPreprocessor()
    df_features = preprocessor.create_features(df)
    print(f"Features created: {df_features.columns.tolist()}")