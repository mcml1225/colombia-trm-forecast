"""
Module for automatic data fetching from Banco de la República
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TRMFetcher:
    """Fetches TRM data from Banco de la República"""
    
    def __init__(self):
        self.base_url = "https://suameca.banrep.gov.co/estadisticas-economicas-back/reporte-oac.html"
        self.data_path = Path("data/raw")
        self.data_path.mkdir(parents=True, exist_ok=True)
    
    def fetch_csv_data(self, start_date=None, end_date=None):
        """
        Fetch TRM data in CSV format
        
        Args:
            start_date: datetime object
            end_date: datetime object
        
        Returns:
            DataFrame with TRM data
        """
        try:
            # Actual URL for CSV download (adjust based on actual endpoint)
            csv_url = "https://suameca.banrep.gov.co/estadisticas-economicas-back/api/export/csv"
            
            params = {
                'path': '/Estadisticas_Banco_de_la_Republica/4_Sector_Externo_tasas_de_cambio_y_derivados/1_Tasas_de_cambio/1_Tasa_de_cambio_del_peso_colombiano_por_USD(TRM)/2_Mercado_Interbancario_de_Divisas/1_Mercado_interbancario_de_divisas',
                'start_date': start_date.strftime('%Y-%m-%d') if start_date else None,
                'end_date': end_date.strftime('%Y-%m-%d') if end_date else None
            }
            
            response = requests.get(csv_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Save raw data
            filename = f"trm_data_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = self.data_path / filename
            
            with open(filepath, 'w') as f:
                f.write(response.text)
            
            # Parse CSV
            df = pd.read_csv(filepath, encoding='utf-8')
            logger.info(f"Successfully fetched data: {len(df)} records")
            
            return self._clean_raw_data(df)
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return self._load_backup_data()
    
    def _clean_raw_data(self, df):
        """Clean and standardize raw data"""
        # Rename columns to English
        column_mapping = {
            'Fecha': 'date',
            'TRM (COP/USD)': 'trm',
            'Número de negociaciones': 'num_trades',
            'Monto negociado (COP/USD)': 'trade_volume',
            'Tasa de cambio de apertura (COP/USD)': 'open_rate',
            'Tasa promedio ponderado (COP/USD)': 'weighted_avg',
            'Tasa de cambio de cierre (COP/USD)': 'close_rate',
            'Tasa de cambio máximo (COP/USD)': 'max_rate',
            'Tasa de cambio mínimo (COP/USD)': 'min_rate'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
        
        # Clean numeric columns
        numeric_columns = ['trm', 'num_trades', 'trade_volume', 'open_rate', 
                          'weighted_avg', 'close_rate', 'max_rate', 'min_rate']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('.', '')
                df[col] = df[col].str.replace(',', '.').astype(float)
        
        # Sort by date
        df = df.sort_values('date')
        
        return df
    
    def _load_backup_data(self):
        """Load last available backup data"""
        backup_files = list(self.data_path.glob("trm_data_*.csv"))
        if backup_files:
            latest = max(backup_files)
            df = pd.read_csv(latest)
            return self._clean_raw_data(df)
        return None
    
    def update_data(self):
        """Main method to update data automatically"""
        logger.info("Starting automatic data update...")
        
        # Get last 180 days of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        df = self.fetch_csv_data(start_date, end_date)
        
        if df is not None:
            # Save processed data
            processed_path = Path("data/processed")
            processed_path.mkdir(parents=True, exist_ok=True)
            df.to_csv(processed_path / "trm_historical.csv", index=False)
            logger.info("Data updated successfully")
            return df
        
        logger.warning("Could not fetch new data, using backup")
        return None

if __name__ == "__main__":
    fetcher = TRMFetcher()
    fetcher.update_data()