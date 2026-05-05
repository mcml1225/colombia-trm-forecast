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
    
    # Read file with UTF-8 encoding
    try:
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:10]
        print("\n  First 10 lines of dashboard:")
        for i, line in enumerate(lines, 1):
            print(f"    {i}: {line.rstrip()}")
    except UnicodeDecodeError:
        print("\n  Warning: File has encoding issues, trying alternative...")
        with open(dashboard_path, 'r', encoding='latin-1') as f:
            lines = f.readlines()[:10]
        print("  First 10 lines (simplified):")
        for i, line in enumerate(lines, 1):
            # Remove special characters for display
            clean_line = line.encode('ascii', 'ignore').decode('ascii')
            print(f"    {i}: {clean_line[:80]}")
    
else:
    print(f"\n✗ Dashboard file not found at {dashboard_path}")

print("\n" + "=" * 60)
print("To run the dashboard, use:")
print("streamlit run dashboard/app.py")
print("=" * 60)
