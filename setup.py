from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="colombia-trm-forecast",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Automated TRM forecasting system for Colombia",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/colombia-trm-forecast",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "streamlit>=1.20.0",
        "plotly>=5.10.0",
        "scikit-learn>=1.1.0",
        "statsmodels>=0.13.0",
    ],
    entry_points={
        "console_scripts": [
            "trm-fetch=src.data_fetcher:main",
            "trm-dashboard=dashboard.app:main",
        ],
    },
)
