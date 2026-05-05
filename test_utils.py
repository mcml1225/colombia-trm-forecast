import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd() / 'src'))

from utils import (
    get_colombia_holidays, 
    is_trading_day, 
    get_next_business_day,
    validate_trm_data,
    create_sample_data,
    test_colombian_holidays
)
from datetime import datetime

print("=" * 60)
print("TESTING UTILS MODULE")
print("=" * 60)

# Test 1: Colombian holidays
print("\n1. Testing Colombian holidays for 2024...")
holidays = test_colombian_holidays()
print(f"   Total holidays in 2024: {len(holidays)}")

# Test 2: Trading day check
print("\n2. Testing trading day check...")
today = datetime.now()
is_trading = is_trading_day(today)
print(f"   Is {today.strftime('%Y-%m-%d')} a trading day? {is_trading}")

# Test 3: Next business day
print("\n3. Testing next business day calculation...")
next_biz = get_next_business_day(today)
print(f"   Next business day after {today.strftime('%Y-%m-%d')}: {next_biz.strftime('%Y-%m-%d')}")

# Test 4: Sample data creation
print("\n4. Testing sample data creation...")
sample_df = create_sample_data(100)
print(f"   Created DataFrame with {len(sample_df)} rows")
print(f"   Columns: {sample_df.columns.tolist()}")

# Test 5: Data validation
print("\n5. Testing data validation...")
is_valid, issues = validate_trm_data(sample_df)
print(f"   Data valid: {is_valid}")
if issues:
    print(f"   Issues: {issues[:3]}")  # Show first 3 issues

print("\n" + "=" * 60)
print("UTILS MODULE TESTS COMPLETED")
print("=" * 60)
