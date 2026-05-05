import pandas as pd
from pathlib import Path

print("=" * 60)
print("INSPECTING DATA")
print("=" * 60)

# Check if data exists
data_path = Path("data/processed/trm_historical.csv")
if data_path.exists():
    df = pd.read_csv(data_path)
    print(f"\n✓ Data file found")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"\n  First row:")
    print(df.iloc[0].to_dict())
    print(f"\n  Data types:")
    print(df.dtypes)
else:
    print(f"\n✗ No data file found at {data_path}")
    print("  Creating sample data...")
    
    from utils import create_sample_data
    df = create_sample_data(200)
    df.to_csv(data_path, index=False)
    print(f"  Sample data created with {len(df)} rows")

print("\n" + "=" * 60)
