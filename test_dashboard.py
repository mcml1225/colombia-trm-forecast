import streamlit as st
import sys
from pathlib import Path

print("=" * 60)
print("TESTING STREAMLIT DASHBOARD")
print("=" * 60)

# Check if dashboard file exists
dashboard_path = Path('dashboard/app.py')
if dashboard_path.exists():
    print(f"\n✓ Dashboard file found at: {dashboard_path}")
    print(f"  File size: {dashboard_path.stat().st_size} bytes")
    
    # Read first few lines
    with open(dashboard_path, 'r') as f:
        lines = f.readlines()[:10]
    print("\n  First 10 lines of dashboard:")
    for line in lines:
        print(f"    {line.rstrip()}")
else:
    print(f"\n✗ Dashboard file not found at {dashboard_path}")

print("\n" + "=" * 60)
print("To run the dashboard, use:")
print("streamlit run dashboard/app.py")
print("=" * 60)
