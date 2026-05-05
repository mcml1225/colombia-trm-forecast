import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path.cwd() / 'src'))

from preprocess import TRMPreprocessor
from traditional_model import TraditionalModels
from utils import create_sample_data

print("=" * 60)
print("TESTING TRADITIONAL MODELS")
print("=" * 60)

# Prepare data
print("\n1. Preparing data...")
df = create_sample_data(300)
preprocessor = TRMPreprocessor()
df_features = preprocessor.create_features(df)
series = df['trm'].values
print(f"   Series length: {len(series)}")

# Split into train/test
train_size = len(series) - 30
train_series = series[:train_size]
test_series = series[train_size:]
print(f"   Train size: {len(train_series)}, Test size: {len(test_series)}")

# Initialize models
print("\n2. Initializing traditional models...")
traditional = TraditionalModels()
print("   Models initialized")

# Test ARIMA
print("\n3. Training ARIMA model...")
try:
    arima_model = traditional.fit_arima(train_series, order=(3,1,0))
    if arima_model:
        print("   ARIMA model trained successfully")
        arima_forecast = traditional.forecast('arima', steps=len(test_series))
        if arima_forecast is not None:
            print(f"   ARIMA forecast generated: {len(arima_forecast)} steps")
            print(f"   Last 3 predictions: {arima_forecast[-3:]}")
except Exception as e:
    print(f"   ARIMA test failed: {e}")

# Test SARIMA
print("\n4. Training SARIMA model...")
try:
    sarima_model = traditional.fit_sarima(train_series, order=(1,1,1), seasonal_order=(1,1,1,7))
    if sarima_model:
        print("   SARIMA model trained successfully")
except Exception as e:
    print(f"   SARIMA test skipped (may need more data): {e}")

# Evaluate models
print("\n5. Evaluating models on test data...")
if len(traditional.models) > 0:
    metrics = traditional.evaluate_models(test_series)
    for model_name, model_metrics in metrics.items():
        print(f"\n   {model_name.upper()}:")
        print(f"     MAE: {model_metrics['MAE']:.2f}")
        print(f"     RMSE: {model_metrics['RMSE']:.2f}")
        print(f"     MAPE: {model_metrics['MAPE']:.2f}%")

print("\n" + "=" * 60)
print("TRADITIONAL MODELS TESTS COMPLETED")
print("=" * 60)
