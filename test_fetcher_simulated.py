import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path.cwd() / 'src'))

from utils import create_sample_data

print("=" * 60)
print("TESTING DATA FETCHER (SIMULATED)")
print("=" * 60)

# Create sample data instead of fetching from web
print("\n1. Creating simulated TRM data...")
df = create_sample_data(200)
print(f"   Data created successfully:")
print(f"   - Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
print(f"   - Total records: {len(df)}")
print(f"   - Latest TRM: {df['trm'].iloc[-1]:.2f} COP/USD")
print(f"\n   First 5 rows:")
print(df.head())
print(f"\n   Last 5 rows:")
print(df.tail())

# Save sample data for testing
print("\n2. Saving sample data...")
df.to_csv("data/processed/trm_historical.csv", index=False)
print("   Sample data saved to data/processed/trm_historical.csv")

print("\n" + "=" * 60)
print("DATA FETCHER SIMULATION COMPLETED")
print("=" * 60)
