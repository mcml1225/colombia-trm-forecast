# Colombia TRM Forecast

[![GitHub stars](https://img.shields.io/github/stars/tu-usuario/colombia-trm-forecast)](https://github.com/tu-usuario/colombia-trm-forecast/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/tu-usuario/colombia-trm-forecast)](https://github.com/tu-usuario/colombia-trm-forecast/issues)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Colombia TRM Forecasting System

Automated forecasting system for Colombia's Representative Market Rate (TRM) using hybrid models (ARIMA, SARIMA, Prophet, LSTM, BiLSTM, GRU) with automatic daily updates.

## Features

- Automatic daily data download from Banco de la República
- Traditional time series models (ARIMA, SARIMA, ETS, Prophet)
- Deep learning models (LSTM, BiLSTM, GRU with Attention)
- Ensemble model combining all predictions
- Interactive Streamlit dashboard
- GitHub Actions for daily automation at 7:00 AM (GMT-5)

## Data Source

The system automatically fetches data from:
https://suameca.banrep.gov.co/estadisticas-economicas-back/reporte-oac.html?path=%2FEstadisticas_Banco_de_la_Republica%2F4_Sector_Externo_tasas_de_cambio_y_derivados%2F1_Tasas_de_cambio%2F1_Tasa_de_cambio_del_peso_colombiano_por_USD(TRM)%2F2_Mercado_Interbancario_de_Divisas%2F1_Mercado_interbancario_de_divisas


Available formats: XLSX, PDF, CSV

## Dashboard Features

- Real-time TRM visualization with historical data
- Model performance metrics (MAE, RMSE, MAPE)
- 7-day and 30-day forecasts from all models
- Interactive model comparison charts
- Volatility analysis and trend detection
- Automatic daily updates at 7:00 AM (GMT-5)

## Project Structure
colombia-trm-forecast/
├── .github/workflows/ # GitHub Actions for automation
├── data/ # Raw and processed data
├── models/ # Saved trained models
├── src/ # Source code
│ ├── data_fetcher.py # Automatic data download
│ ├── preprocess.py # Data preprocessing
│ ├── traditional_model.py # ARIMA, SARIMA, Prophet
│ ├── ai_model.py # LSTM, BiLSTM, GRU
│ └── ensemble_model.py # Ensemble predictions
├── dashboard/ # Streamlit dashboard
├── tests/ # Unit tests
└── requirements.txt # Dependencies


## Installation

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Clone Repository

git clone https://github.com/mcml1225/colombia-trm-forecast.git
cd colombia-trm-forecast

## Create Virtual Environment
Windows:
python -m venv venv
venv\Scripts\activate
Linux/Mac:
python3 -m venv venv
source venv/bin/activate
### Install Dependencies

pip install -r requirements.txt
### Usage
#### Run Dashboard Locally

streamlit run dashboard/app.py
The dashboard will open in your browser at http://localhost:8501

#### Manual Data Update
python src/data_fetcher.py

#### Train Traditional Models
python src/traditional_model.py

#### Train Deep Learning Models
python src/ai_model.py

#### Train Ensemble Model
python src/ensemble_model.py

#### Run Tests
pytest tests/

### Model Descriptions
#### Traditional Models
Model	Description
ARIMA	Autoregressive Integrated Moving Average
SARIMA	Seasonal ARIMA with weekly seasonality
ETS	Exponential Smoothing with trend and seasonality
Prophet	Facebook's Prophet with additional regressors

#### Deep Learning Models
Model	Description
LSTM	Long Short-Term Memory networks
BiLSTM	Bidirectional LSTM for better sequence learning
GRU	Gated Recurrent Units (faster than LSTM)
Attention-LSTM	LSTM with attention mechanism

#### Ensemble Methods
Weighted Average with optimized weights

Median combination

Stacking with meta-learner (Random Forest/Gradient Boosting)

#### Evaluation Metrics
**MAE:** Mean Absolute Error

**RMSE:** Root Mean Square Error

**MAPE:** Mean Absolute Percentage Error

#### Automation
The system automatically updates daily at 7:00 AM (GMT-5) using GitHub Actions:

Fetches latest TRM data from Banco de la República

Retrains all models with updated data

Generates new forecasts

Updates dashboard metrics

To modify the automation schedule, edit .github/workflows/daily_update.yml.
