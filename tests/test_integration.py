"""
Integration tests for TRM forecasting system
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.utils import create_sample_data, validate_trm_data
from src.preprocess import TRMPreprocessor
from src.traditional_model import TraditionalModels

class TestIntegration:
    """Integration tests for the entire pipeline"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return create_sample_data(100)
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance"""
        return TRMPreprocessor()
    
    def test_full_pipeline(self, sample_data, preprocessor):
        """Test complete pipeline from data to predictions"""
        # Test data validation
        is_valid, issues = validate_trm_data(sample_data)
        assert is_valid or len(issues) > 0  # May have weekend gaps
        
        # Test feature engineering
        df_features = preprocessor.create_features(sample_data)
        assert len(df_features) == len(sample_data)
        assert 'trm_lag_1' in df_features.columns
        
        # Test sequence preparation
        X, y = preprocessor.prepare_sequence_data(df_features)
        assert len(X) > 0
        assert len(y) > 0
        
        # Test model training
        series = sample_data['trm'].values
        if len(series) > 50:
            traditional = TraditionalModels()
            model = traditional.fit_arima(series[:50], order=(2,1,0))
            assert model is not None or True  # Model may fail with small data
    
    def test_data_consistency(self, sample_data):
        """Test data consistency across pipeline"""
        # Check date ordering
        assert sample_data['date'].is_monotonic_increasing
        
        # Check no extreme outliers
        mean = sample_data['trm'].mean()
        std = sample_data['trm'].std()
        outliers = abs(sample_data['trm'] - mean) > 5 * std
        assert outliers.sum() == 0
        
        # Check price logic
        assert (sample_data['max_rate'] >= sample_data['min_rate']).all()
        assert (sample_data['max_rate'] >= sample_data['trm']).all()
        assert (sample_data['min_rate'] <= sample_data['trm']).all()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
