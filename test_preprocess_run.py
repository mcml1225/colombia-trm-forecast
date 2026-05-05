import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path.cwd() / 'src'))

from preprocess import TRMPreprocessor
from utils import create_sample_data

print("=" * 60)
print("TESTING PREPROCESS MODULE")
print("=" * 60)

# Load or create data
print("\n1. Loading data...")
data_path = Path("data/processed/trm_historical.csv")
if data_path.exists():
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    print(f"   Loaded existing data with {len(df)} rows")
else:
    df = create_sample_data(200)
    print(f"   Created new sample data with {len(df)} rows")

# Initialize preprocessor
print("\n2. Initializing preprocessor...")
preprocessor = TRMPreprocessor(scaler_type='minmax')
print("   Preprocessor initialized successfully")

# Create features
print("\n3. Creating time series features...")
df_features = preprocessor.create_features(df)
print(f"   Original columns: {len(df.columns)}")
print(f"   Features created: {len(df_features.columns)}")
print(f"   New features: {[col for col in df_features.columns if col not in df.columns][:5]}...")

# Check for missing values
print("\n4. Checking for missing values...")
missing_cols = df_features.columns[df_features.isnull().any()].tolist()
if missing_cols:
    print(f"   Columns with missing values: {missing_cols}")
    # Fill missing values
    df_features = df_features.fillna(method='bfill').fillna(method='ffill')
    print("   Missing values filled")
else:
    print("   No missing values found")

# Prepare sequence data
print("\n5. Preparing sequence data for LSTM...")
X, y = preprocessor.prepare_sequence_data(df_features, sequence_length=30)
print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")
print(f"   Feature dimension: {X.shape[2]} features")

# Split data
print("\n6. Splitting data...")
split_data = preprocessor.split_data(X, y, test_size=0.2, val_size=0.1)
for key, value in split_data.items():
    if isinstance(value, np.ndarray):
        print(f"   {key}: {value.shape}")

print("\n" + "=" * 60)
print("PREPROCESS MODULE TESTS COMPLETED SUCCESSFULLY")
print("=" * 60)
