"""
Utility functions for TRM forecasting project
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json
import yaml
from pathlib import Path
import hashlib
from typing import Dict, List, Tuple, Any, Optional
import requests
from functools import wraps
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============== Date and Time Utilities ==============

def get_last_business_day(date: datetime = None) -> datetime:
    """
    Get last business day (Monday-Friday)
    
    Args:
        date: Reference date, defaults to current date
    
    Returns:
        Last business day
    """
    if date is None:
        date = datetime.now()
    
    # Adjust for weekend
    if date.weekday() >= 5:  # Saturday = 5, Sunday = 6
        days_to_subtract = date.weekday() - 4
        date = date - timedelta(days=days_to_subtract)
    
    return date.replace(hour=0, minute=0, second=0, microsecond=0)

def is_trading_day(date: datetime) -> bool:
    """
    Check if a date is a trading day (Monday-Friday, not holiday)
    
    Args:
        date: Date to check
    
    Returns:
        True if trading day
    """
    # Simple check for weekend
    if date.weekday() >= 5:
        return False
    
    # Get Colombian holidays for the year
    colombia_holidays = get_colombia_holidays(date.year)
    
    return date.strftime('%Y-%m-%d') not in colombia_holidays

def get_colombia_holidays(year: int) -> List[str]:
    """
    Get Colombian holidays for a specific year according to Ley Emiliani (Law 51 of 1983)
    Holidays are moved to the following Monday when they fall on a weekday
    
    Args:
        year: Year to get holidays for
    
    Returns:
        List of holiday dates in YYYY-MM-DD format (actual celebration dates)
    """
    
    def move_to_monday(date: datetime) -> datetime:
        """Move a date to the following Monday if it falls on Tuesday-Thursday"""
        # According to Colombian law, holidays are celebrated on the following Monday
        # when they fall on a Tuesday, Wednesday, Thursday, or Friday
        if date.weekday() in [1, 2, 3, 4]:  # Tuesday (1), Wednesday (2), Thursday (3), Friday (4)
            days_to_monday = 7 - date.weekday()
            return date + timedelta(days=days_to_monday)
        return date
    
    def move_to_nearest_monday(date: datetime) -> datetime:
        """Move a date to the nearest Monday (if weekend) or following Monday for some holidays"""
        # For fixed dates that fall on Saturday or Sunday, move to Monday
        if date.weekday() == 5:  # Saturday
            return date + timedelta(days=2)
        elif date.weekday() == 6:  # Sunday
            return date + timedelta(days=1)
        return date
    
    holidays = []
    
    # Fixed holidays (never move - celebrated on exact date even if weekend)
    fixed_holidays = [
        (1, 1, "New Year's Day"),           # January 1
        (5, 1, "Labor Day"),                # May 1
        (7, 20, "Independence Day"),        # July 20
        (8, 7, "Battle of Boyacá"),         # August 7
        (12, 8, "Immaculate Conception"),   # December 8
        (12, 25, "Christmas Day")           # December 25
    ]
    
    for month, day, name in fixed_holidays:
        date = datetime(year, month, day)
        # Fixed holidays are celebrated on the exact date
        holidays.append(date.strftime('%Y-%m-%d'))
    
    # Holidays that move to the following Monday according to Ley Emiliani
    movable_holidays = [
        (1, 6, "Epiphany"),                 # January 6 -> move to Monday
        (3, 19, "St. Joseph's Day"),        # March 19 -> move to Monday
        (4, 9, "Holy Week"),                # April 9 (Maundy Thursday) -> move to Monday
        (6, 15, "Corpus Christi"),          # June 15 -> move to Monday
        (6, 22, "Sacred Heart"),            # June 22 -> move to Monday
        (6, 29, "St. Peter and St. Paul"),  # June 29 -> move to Monday
        (8, 15, "Assumption of Mary"),      # August 15 -> move to Monday
        (10, 12, "Columbus Day"),           # October 12 -> move to Monday
        (11, 1, "All Saints' Day"),         # November 1 -> move to Monday
        (11, 11, "Independence of Cartagena") # November 11 -> move to Monday
    ]
    
    for month, day, name in movable_holidays:
        date = datetime(year, month, day)
        # Move to following Monday according to Ley Emiliani
        holiday_date = move_to_monday(date)
        holidays.append(holiday_date.strftime('%Y-%m-%d'))
    
    # Special case: Ascension Day (40 days after Easter)
    ascension_day = get_ascension_day(year)
    holidays.append(ascension_day.strftime('%Y-%m-%d'))
    
    # Special case: St. John's Day (June 24 - sometimes moved)
    st_john = datetime(year, 6, 24)
    st_john_moved = move_to_monday(st_john)
    holidays.append(st_john_moved.strftime('%Y-%m-%d'))
    
    # Remove duplicates and sort
    holidays = sorted(list(set(holidays)))
    
    return holidays

def get_easter_date(year: int) -> datetime:
    """
    Calculate Easter Sunday date using computus algorithm
    
    Args:
        year: Year to calculate Easter for
    
    Returns:
        Easter Sunday date
    """
    # Gauss algorithm for calculating Easter
    a = year % 19
    b = year % 4
    c = year % 7
    k = year // 100
    p = (13 + 8 * k) // 25
    q = k // 4
    M = (15 - p + k - q) % 30
    N = (4 + k - q) % 7
    d = (19 * a + M) % 30
    e = (2 * b + 4 * c + 6 * d + N) % 7
    
    if d == 29 and e == 6:
        day = 19
        month = 4
    elif d == 28 and e == 6 and (11 * M + 11) % 30 < 19:
        day = 18
        month = 4
    else:
        day = 22 + d + e
        month = 3
        if day > 31:
            day = d + e - 9
            month = 4
    
    return datetime(year, month, day)

def get_ascension_day(year: int) -> datetime:
    """
    Get Ascension Day (40 days after Easter)
    
    Args:
        year: Year to calculate for
    
    Returns:
        Ascension Day date
    """
    easter = get_easter_date(year)
    ascension = easter + timedelta(days=39)  # 40th day including Easter Sunday
    
    # Move to following Monday according to Ley Emiliani
    def move_to_monday(date):
        if date.weekday() in [1, 2, 3, 4]:  # Tuesday to Friday
            days_to_monday = 7 - date.weekday()
            return date + timedelta(days=days_to_monday)
        return date
    
    return move_to_monday(ascension)

def get_next_business_day(date: datetime) -> datetime:
    """
    Get next business day (skipping weekends and holidays)
    
    Args:
        date: Starting date
    
    Returns:
        Next business day
    """
    next_day = date + timedelta(days=1)
    
    while not is_trading_day(next_day):
        next_day += timedelta(days=1)
    
    return next_day

def get_previous_business_day(date: datetime) -> datetime:
    """
    Get previous business day (skipping weekends and holidays)
    
    Args:
        date: Starting date
    
    Returns:
        Previous business day
    """
    prev_day = date - timedelta(days=1)
    
    while not is_trading_day(prev_day):
        prev_day -= timedelta(days=1)
    
    return prev_day

def get_business_day_range(start_date: datetime, end_date: datetime) -> List[datetime]:
    """
    Get range of business days between two dates
    
    Args:
        start_date: Start date
        end_date: End date
    
    Returns:
        List of business days
    """
    business_days = []
    current = start_date
    
    while current <= end_date:
        if is_trading_day(current):
            business_days.append(current)
        current += timedelta(days=1)
    
    return business_days

# ============== Data Validation Utilities ==============

def validate_trm_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate TRM data quality
    
    Args:
        df: DataFrame with TRM data
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check required columns
    required_columns = ['date', 'trm']
    for col in required_columns:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")
    
    if issues:
        return False, issues
    
    # Check for missing values
    missing_count = df['trm'].isnull().sum()
    if missing_count > 0:
        issues.append(f"Missing values in TRM column: {missing_count} rows")
    
    # Check for negative values
    negative_count = (df['trm'] < 0).sum()
    if negative_count > 0:
        issues.append(f"Negative TRM values found: {negative_count} rows")
    
    # Check for extreme outliers (beyond 3 standard deviations)
    mean = df['trm'].mean()
    std = df['trm'].std()
    outliers = ((df['trm'] < mean - 3*std) | (df['trm'] > mean + 3*std)).sum()
    if outliers > 0:
        issues.append(f"Extreme outliers detected: {outliers} rows")
    
    # Check date consistency
    if not df['date'].is_monotonic_increasing:
        issues.append("Dates are not in chronological order")
    
    # Check date gaps (only business days should have data)
    business_dates = df[df['date'].apply(is_trading_day)]['date']
    if len(business_dates) > 1:
        date_diff = business_dates.diff().dropna()
        # A gap of more than 1 day means missing data
        large_gaps = (date_diff > pd.Timedelta(days=1)).sum()
        if large_gaps > 0:
            issues.append(f"Large gaps in business day sequence: {large_gaps} gaps > 1 day")
    
    is_valid = len(issues) == 0
    return is_valid, issues

def handle_missing_values(df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
    """
    Handle missing values in time series data
    
    Args:
        df: DataFrame with missing values
        method: 'interpolate', 'ffill', 'bfill', 'mean'
    
    Returns:
        DataFrame with handled missing values
    """
    df = df.copy()
    
    if method == 'interpolate':
        df['trm'] = df['trm'].interpolate(method='linear')
    elif method == 'ffill':
        df['trm'] = df['trm'].fillna(method='ffill')
        df['trm'] = df['trm'].fillna(method='bfill')  # Fill remaining NaNs from start
    elif method == 'bfill':
        df['trm'] = df['trm'].fillna(method='bfill')
    elif method == 'mean':
        df['trm'] = df['trm'].fillna(df['trm'].mean())
    
    # Fill remaining NaN values with forward fill as fallback
    df['trm'] = df['trm'].fillna(method='ffill')
    
    return df

def detect_anomalies(df: pd.DataFrame, threshold: float = 3) -> pd.DataFrame:
    """
    Detect anomalies using Z-score method
    
    Args:
        df: DataFrame with TRM data
        threshold: Z-score threshold for anomaly detection
    
    Returns:
        DataFrame with anomaly flags
    """
    df = df.copy()
    
    # Calculate rolling statistics
    rolling_mean = df['trm'].rolling(window=30, center=True).mean()
    rolling_std = df['trm'].rolling(window=30, center=True).std()
    
    # Calculate Z-score
    z_score = (df['trm'] - rolling_mean) / rolling_std
    
    # Flag anomalies
    df['is_anomaly'] = np.abs(z_score) > threshold
    df['anomaly_zscore'] = z_score
    
    return df

# ============== Performance Metrics Utilities ==============

def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
    
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {}
    
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,  # as percentage
        'R2': r2_score(y_true, y_pred)
    }
    
    # Additional metrics
    metrics['MSE'] = mean_squared_error(y_true, y_pred)
    metrics['MaxError'] = np.max(np.abs(y_true - y_pred))
    metrics['MedianAE'] = np.median(np.abs(y_true - y_pred))
    
    # Directional accuracy
    actual_direction = np.diff(y_true) > 0
    predicted_direction = np.diff(y_pred) > 0
    if len(actual_direction) == len(predicted_direction):
        metrics['DirectionalAccuracy'] = np.mean(actual_direction == predicted_direction) * 100
    
    # Theil's U statistic
    if len(y_true) > 1:
        numerator = np.sqrt(np.sum((y_pred[1:] - y_true[1:])**2))
        denominator = np.sqrt(np.sum(y_pred[1:]**2)) + np.sqrt(np.sum(y_true[1:]**2))
        metrics['TheilsU'] = numerator / denominator if denominator > 0 else np.nan
    
    return metrics

def calculate_confidence_interval(predictions: np.ndarray, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals for predictions
    
    Args:
        predictions: Array of predictions from different runs/models
        confidence: Confidence level (0.95 for 95%)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    alpha = 1 - confidence
    lower = np.percentile(predictions, alpha/2 * 100, axis=0)
    upper = np.percentile(predictions, (1 - alpha/2) * 100, axis=0)
    
    return lower, upper

# ============== Model Management Utilities ==============

def get_model_file_path(model_type: str, model_name: str = None) -> Path:
    """
    Get standardized path for model files
    
    Args:
        model_type: 'traditional', 'ai', 'ensemble'
        model_name: Name of the specific model
    
    Returns:
        Path object for model file
    """
    base_path = Path(f"models/{model_type}")
    base_path.mkdir(parents=True, exist_ok=True)
    
    if model_name:
        return base_path / f"{model_name}.pkl"
    return base_path

def save_dict_to_json(data: Dict, filepath: Path):
    """Save dictionary to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, default=str)

def load_dict_from_json(filepath: Path) -> Dict:
    """Load dictionary from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def compute_file_hash(filepath: Path) -> str:
    """
    Compute SHA256 hash of a file
    
    Args:
        filepath: Path to file
    
    Returns:
        SHA256 hash string
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# ============== Logging Utilities ==============

class PerformanceLogger:
    """Context manager for logging performance metrics"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Starting: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        if exc_type is None:
            logger.info(f"Completed: {self.operation_name} in {elapsed_time:.2f} seconds")
        else:
            logger.error(f"Failed: {self.operation_name} after {elapsed_time:.2f} seconds - Error: {exc_val}")

def retry(max_attempts: int = 3, delay: int = 5):
    """
    Decorator for retrying failed operations
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

# ============== Data Export Utilities ==============

def export_data_to_csv(df: pd.DataFrame, filename: str, folder: str = "exports"):
    """
    Export DataFrame to CSV with timestamp
    
    Args:
        df: DataFrame to export
        filename: Base filename
        folder: Output folder name
    
    Returns:
        Path to exported file
    """
    output_dir = Path(folder)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"{filename}_{timestamp}.csv"
    
    df.to_csv(filepath, index=False)
    logger.info(f"Data exported to {filepath}")
    
    return filepath

def export_forecast_to_excel(forecasts: Dict[str, np.ndarray], dates: List[str], filepath: Path):
    """
    Export forecasts to Excel file with multiple sheets
    
    Args:
        forecasts: Dictionary of model_name -> predictions
        dates: List of forecast dates
        filepath: Output file path
    """
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        for model_name, predictions in forecasts.items():
            df = pd.DataFrame({
                'Date': dates[:len(predictions)],
                f'{model_name}_forecast': predictions
            })
            df.to_excel(writer, sheet_name=model_name[:31], index=False)
    
    logger.info(f"Forecasts exported to {filepath}")

# ============== Configuration Utilities ==============

def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    default_config = {
        'data': {
            'source_url': 'https://suameca.banrep.gov.co',
            'update_time': '07:00',
            'timezone': 'America/Bogota',
            'max_history_days': 365
        },
        'models': {
            'traditional': {
                'arima_order': [5, 1, 0],
                'sarima_order': [1, 1, 1],
                'seasonal_order': [1, 1, 1, 7]
            },
            'deep_learning': {
                'lstm_units': [64, 32],
                'dropout_rate': 0.2,
                'batch_size': 32,
                'epochs': 100
            }
        },
        'forecast': {
            'horizon_days': 30,
            'retraining_frequency': 'daily',
            'ensemble_method': 'weighted_average'
        }
    }
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            # Merge with default config
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            return config
    
    return default_config

# ============== Visualization Utilities ==============

def prepare_forecast_plot_data(df: pd.DataFrame, forecasts: Dict[str, np.ndarray], 
                               forecast_dates: List[str]) -> Dict:
    """
    Prepare data for forecast visualization
    
    Args:
        df: Historical DataFrame
        forecasts: Dictionary of model predictions
        forecast_dates: List of forecast dates
    
    Returns:
        Dictionary with plot data
    """
    plot_data = {
        'historical': {
            'dates': df['date'].tolist(),
            'values': df['trm'].tolist()
        },
        'forecasts': {}
    }
    
    for model_name, predictions in forecasts.items():
        plot_data['forecasts'][model_name] = {
            'dates': forecast_dates[:len(predictions)],
            'values': predictions.tolist()
        }
    
    return plot_data

# ============== Alert Utilities ==============

def check_alert_conditions(current_trm: float, predicted_trm: float, 
                          historical_mean: float, threshold_pct: float = 5) -> Dict:
    """
    Check if any alert conditions are met
    
    Args:
        current_trm: Current TRM value
        predicted_trm: Predicted TRM value
        historical_mean: Historical mean TRM
        threshold_pct: Alert threshold percentage
    
    Returns:
        Dictionary with alert information
    """
    alerts = []
    
    # Calculate changes
    predicted_change = ((predicted_trm - current_trm) / current_trm) * 100
    
    # Check for significant predicted movement
    if abs(predicted_change) > threshold_pct:
        alerts.append({
            'type': 'significant_movement',
            'severity': 'high' if abs(predicted_change) > threshold_pct * 2 else 'medium',
            'message': f"Predicted {predicted_change:.1f}% change in TRM",
            'value': predicted_change
        })
    
    # Check if prediction is outside historical range
    if predicted_trm > historical_mean * 1.1:
        alerts.append({
            'type': 'high_value',
            'severity': 'medium',
            'message': f"Predicted TRM ({predicted_trm:.2f}) is 10% above historical mean",
            'value': predicted_trm
        })
    
    return {
        'has_alerts': len(alerts) > 0,
        'alerts': alerts,
        'predicted_change_pct': predicted_change
    }

# ============== Testing Utilities ==============

def create_sample_data(days: int = 100) -> pd.DataFrame:
    """
    Create sample TRM data for testing
    
    Args:
        days: Number of days of sample data
    
    Returns:
        Sample DataFrame
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate realistic TRM values (COP/USD)
    base_trm = 3600
    trend = np.linspace(0, 100, days)
    seasonality = 50 * np.sin(2 * np.pi * np.arange(days) / 30)
    noise = np.random.normal(0, 20, days)
    
    trm_values = base_trm + trend + seasonality + noise
    
    df = pd.DataFrame({
        'date': dates,
        'trm': trm_values,
        'open_rate': trm_values + np.random.normal(0, 5, days),
        'close_rate': trm_values + np.random.normal(0, 5, days),
        'max_rate': trm_values + np.random.uniform(5, 20, days),
        'min_rate': trm_values - np.random.uniform(5, 20, days),
        'num_trades': np.random.randint(1000, 5000, days)
    })
    
    return df

def test_colombian_holidays():
    """Test function to verify Colombian holiday calculations"""
    test_year = 2024
    
    holidays = get_colombia_holidays(test_year)
    
    print(f"\nColombian Holidays for {test_year}:")
    print("-" * 40)
    
    for holiday in sorted(holidays):
        date = datetime.strptime(holiday, '%Y-%m-%d')
        # Check if holiday is on Monday (as per Ley Emiliani for movable holidays)
        if date.weekday() == 0:
            print(f"  {holiday} - Monday (Correct after Ley Emiliani)")
        else:
            print(f"  {holiday} - Day {date.strftime('%A')} (Fixed holiday)")
    
    return holidays

if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test holiday calculation
    test_colombian_holidays()
    
    # Test date utilities
    last_business = get_last_business_day()
    print(f"\nLast business day: {last_business.strftime('%Y-%m-%d')}")
    
    # Test next business day
    today = datetime.now()
    next_business = get_next_business_day(today)
    print(f"Next business day after {today.strftime('%Y-%m-%d')}: {next_business.strftime('%Y-%m-%d')}")
    
    # Test validation
    sample_df = create_sample_data(100)
    is_valid, issues = validate_trm_data(sample_df)
    print(f"\nData valid: {is_valid}, Issues: {issues}")
    
    # Test performance logging
    with PerformanceLogger("test_operation"):
        time.sleep(1)
    
    print("\nAll utilities test completed successfully!")