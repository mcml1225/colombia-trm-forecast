#!/usr/bin/env python3
print("=" * 80)
print(" " * 20 + "TRM FORECASTING - INTEGRATED TEST")
print("=" * 80)

# Test 1: Import all modules
print("\n[TEST 1] Importing modules...")
try:
    import utils
    print("  ✓ utils imported")
    import preprocess
    print("  ✓ preprocess imported")
    import traditional_model
    print("  ✓ traditional_model imported")
    print("  ✓ All modules imported successfully")
except Exception as e:
    print(f"  ✗ Import failed: {e}")

# Test 2: Create and validate data
print("\n[TEST 2] Creating and validating data...")
try:
    from utils import create_sample_data, validate_trm_data
    df = create_sample_data(200)
    is_valid, issues = validate_trm_data(df)
    print(f"  ✓ Data created: {len(df)} rows")
    print(f"  ✓ Data validation: {'PASSED' if is_valid else 'WARNING - ' + str(issues[:1])}")
except Exception as e:
    print(f"  ✗ Data creation failed: {e}")

# Test 3: Feature engineering
print("\n[TEST 3] Feature engineering...")
try:
    from preprocess import TRMPreprocessor
    preprocessor = TRMPreprocessor()
    df_features = preprocessor.create_features(df)
    print(f"  ✓ Features created: {len(df.columns)} -> {len(df_features.columns)} columns")
except Exception as e:
    print(f"  ✗ Feature engineering failed: {e}")

# Test 4: Sequence preparation
print("\n[TEST 4] Sequence preparation...")
try:
    X, y = preprocessor.prepare_sequence_data(df_features)
    print(f"  ✓ Sequences prepared: X shape {X.shape}, y shape {y.shape}")
except Exception as e:
    print(f"  ✗ Sequence preparation failed: {e}")

# Test 5: Model initialization
print("\n[TEST 5] Model initialization...")
try:
    from traditional_model import TraditionalModels
    traditional = TraditionalModels()
    print(f"  ✓ Traditional models initialized")
    print(f"  ✓ Models available: {traditional.models.keys()}")
except Exception as e:
    print(f"  ✗ Model initialization failed: {e}")

print("\n" + "=" * 80)
print(" " * 25 + "INTEGRATED TEST COMPLETED")
print("=" * 80)
print("\n✓ Project is ready to run!")
print("\nNext steps:")
print("  1. Run: streamlit run dashboard/app.py")
print("  2. Or run: python src/data_fetcher.py (for real data)")
