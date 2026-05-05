# This file is for Streamlit Cloud deployment
import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Import and run the main dashboard
from dashboard.app import main

if __name__ == "__main__":
    main()
