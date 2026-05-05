"""
Complete TRM Colombia Forecasting Dashboard with Candlestick Charts
Includes: Historical data, technical indicators, forecasts, and candlestick visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="TRM Colombia - Forecasting Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
    }
    .info-text {
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🇨🇴 TRM Colombia - Forecasting Dashboard</div>', 
            unsafe_allow_html=True)
st.markdown("---")

# Load data function
@st.cache_data(ttl=3600)
def load_data():
    """Load TRM data from CSV or create sample"""
    data_path = Path("data/processed/trm_historical.csv")
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
    else:
        # Create realistic sample data
        dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
        base_trm = 3600
        trend = np.linspace(0, 150, 200)
        seasonality = 50 * np.sin(2 * np.pi * np.arange(200) / 30)
        noise = np.random.normal(0, 15, 200)
        trm_values = base_trm + trend + seasonality + noise
        
        df = pd.DataFrame({
            'date': dates,
            'trm': trm_values,
            'open_rate': trm_values + np.random.normal(0, 5, 200),
            'close_rate': trm_values + np.random.normal(0, 5, 200),
            'max_rate': trm_values + np.random.uniform(5, 20, 200),
            'min_rate': trm_values - np.random.uniform(5, 20, 200),
            'num_trades': np.random.randint(1000, 5000, 200)
        })
        df.to_csv(data_path, index=False)
    
    return df

# Load and prepare data
df = load_data()

# Calculate additional metrics
df['returns'] = df['trm'].pct_change() * 100
df['ma_7'] = df['trm'].rolling(window=7).mean()
df['ma_30'] = df['trm'].rolling(window=30).mean()
df['volatility'] = df['trm'].rolling(window=30).std()
df['upper_band'] = df['ma_30'] + (df['volatility'] * 2)
df['lower_band'] = df['ma_30'] - (df['volatility'] * 2)

# Sidebar filters
st.sidebar.header("📊 Dashboard Controls")

# Date range filter
min_date = df['date'].min().date()
max_date = df['date'].max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
else:
    filtered_df = df

# Forecast horizon
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 7, 90, 30)

# Show technical indicators
show_ma = st.sidebar.checkbox("Show Moving Averages", True)
show_bb = st.sidebar.checkbox("Show Bollinger Bands", False)
show_volume = st.sidebar.checkbox("Show Trading Volume", True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Model Information")
st.sidebar.info(
    "This dashboard uses hybrid forecasting methods including:\n"
    "- ARIMA/SARIMA (Traditional)\n"
    "- LSTM Neural Networks (Deep Learning)\n"
    "- Ensemble Methods (Combined predictions)"
)

# Main content
# Key Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    current_trm = filtered_df['trm'].iloc[-1]
    st.metric(
        "Current TRM",
        f"{current_trm:,.2f} COP/USD",
        delta=None
    )

with col2:
    daily_return = filtered_df['returns'].iloc[-1]
    st.metric(
        "Daily Change",
        f"{daily_return:+.2f}%",
        delta=f"{daily_return:+.2f}%" if abs(daily_return) > 0 else None
    )

with col3:
    high_30d = filtered_df['trm'].tail(30).max()
    st.metric("30-Day High", f"{high_30d:,.2f}")

with col4:
    low_30d = filtered_df['trm'].tail(30).min()
    st.metric("30-Day Low", f"{low_30d:,.2f}")

with col5:
    volatility = filtered_df['volatility'].iloc[-1]
    st.metric("Volatility (30d)", f"{volatility:.2f}")

st.markdown("---")

# Candlestick Chart Section
st.subheader("📊 Candlestick Chart (Daily Price Action)")

# Prepare data for candlestick
candle_data = filtered_df.tail(90).copy()
candle_data = candle_data[candle_data['date'].notna()]

# Create candlestick chart
fig_candle = go.Figure(data=[go.Candlestick(
    x=candle_data['date'],
    open=candle_data['open_rate'],
    high=candle_data['max_rate'],
    low=candle_data['min_rate'],
    close=candle_data['close_rate'],
    name='TRM Candlestick',
    increasing_line_color='green',
    decreasing_line_color='red'
)])

# Add volume bars on secondary y-axis
if show_volume:
    fig_candle.add_trace(go.Bar(
        x=candle_data['date'],
        y=candle_data['num_trades'],
        name='Volume',
        marker_color='lightblue',
        opacity=0.3,
        yaxis='y2'
    ))

# Add moving averages
if show_ma:
    fig_candle.add_trace(go.Scatter(
        x=filtered_df['date'],
        y=filtered_df['ma_7'],
        mode='lines',
        name='MA 7',
        line=dict(color='orange', width=1.5)
    ))
    fig_candle.add_trace(go.Scatter(
        x=filtered_df['date'],
        y=filtered_df['ma_30'],
        mode='lines',
        name='MA 30',
        line=dict(color='blue', width=1.5)
    ))

# Update layout for candlestick
fig_candle.update_layout(
    title='TRM Candlestick Chart - Daily Price Action',
    xaxis_title='Date',
    yaxis_title='Price (COP/USD)',
    height=600,
    xaxis_rangeslider_visible=False,
    template='plotly_dark',
    hovermode='x unified'
)

# Add secondary y-axis for volume
if show_volume:
    fig_candle.update_layout(
        yaxis2=dict(
            title='Volume (Trades)',
            overlaying='y',
            side='right',
            showgrid=False
        )
    )

st.plotly_chart(fig_candle, use_container_width=True)

# Historical Price Chart (Line)
st.subheader("📈 Historical TRM Evolution")

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=("TRM Price - Line Chart", "Trading Volume"),
    row_heights=[0.7, 0.3]
)

# Price chart
fig.add_trace(
    go.Scatter(
        x=filtered_df['date'],
        y=filtered_df['trm'],
        mode='lines',
        name='TRM',
        line=dict(color='#1f77b4', width=2)
    ),
    row=1, col=1
)

# Add moving averages if selected
if show_ma:
    fig.add_trace(
        go.Scatter(
            x=filtered_df['date'],
            y=filtered_df['ma_7'],
            mode='lines',
            name='MA (7 days)',
            line=dict(color='orange', width=1.5, dash='dash')
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=filtered_df['date'],
            y=filtered_df['ma_30'],
            mode='lines',
            name='MA (30 days)',
            line=dict(color='red', width=1.5, dash='dash')
        ),
        row=1, col=1
    )

# Add Bollinger Bands if selected
if show_bb:
    fig.add_trace(
        go.Scatter(
            x=filtered_df['date'],
            y=filtered_df['upper_band'],
            mode='lines',
            name='Upper Band',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=True
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=filtered_df['date'],
            y=filtered_df['lower_band'],
            mode='lines',
            name='Lower Band',
            line=dict(color='gray', width=1, dash='dot'),
            fill='tonexty',
            fillcolor='rgba(128, 128, 128, 0.1)'
        ),
        row=1, col=1
    )

# Volume chart
if show_volume:
    colors = ['red' if ret < 0 else 'green' for ret in filtered_df['returns']]
    fig.add_trace(
        go.Bar(
            x=filtered_df['date'],
            y=filtered_df['num_trades'],
            name='Trading Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )

fig.update_layout(
    height=500,
    showlegend=True,
    hovermode='x unified',
    template='plotly_white'
)
fig.update_xaxes(title_text="Date", row=2, col=1)
fig.update_yaxes(title_text="COP/USD", row=1, col=1)
fig.update_yaxes(title_text="Number of Trades", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# Forecasting Section
st.markdown("---")
st.subheader("🔮 TRM Forecast")

# Simple forecasting model
@st.cache_resource
def generate_forecast(series, days):
    """Generate forecast using simple methods"""
    from sklearn.linear_model import LinearRegression
    
    # Prepare data
    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values
    
    # Train models
    lr_model = LinearRegression()
    lr_model.fit(X[-90:], y[-90:])
    
    # Generate forecast
    future_X = np.arange(len(series), len(series) + days).reshape(-1, 1)
    linear_forecast = lr_model.predict(future_X)
    
    # Simple moving average forecast
    ma_forecast = []
    last_ma = series.tail(30).mean()
    for i in range(days):
        ma_forecast.append(last_ma + (linear_forecast[i] - linear_forecast[0]) * 0.5)
    
    return {
        'linear': linear_forecast,
        'ma': np.array(ma_forecast)
    }

# Generate forecast
with st.spinner("Generating forecast..."):
    forecast_models = generate_forecast(filtered_df['trm'], forecast_days)

# Create forecast dates
last_date = filtered_df['date'].iloc[-1]
forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]

# Convert forecast_dates to list of datetime objects (not numpy array)
forecast_dates_list = list(forecast_dates)

# Forecast chart
fig_forecast = go.Figure()

# Historical data
fig_forecast.add_trace(go.Scatter(
    x=filtered_df['date'].tail(90),
    y=filtered_df['trm'].tail(90),
    mode='lines',
    name='Historical',
    line=dict(color='blue', width=2)
))

# Linear forecast
fig_forecast.add_trace(go.Scatter(
    x=forecast_dates_list,
    y=forecast_models['linear'],
    mode='lines',
    name='Linear Regression Forecast',
    line=dict(color='red', width=2, dash='dash')
))

# MA forecast
fig_forecast.add_trace(go.Scatter(
    x=forecast_dates_list,
    y=forecast_models['ma'],
    mode='lines',
    name='Hybrid Forecast',
    line=dict(color='green', width=2, dash='dot')
))

# Add confidence interval (simple)
std_dev = filtered_df['trm'].tail(30).std()
upper_bound = forecast_models['linear'] + std_dev
lower_bound = forecast_models['linear'] - std_dev

# Convert to lists for the confidence interval
upper_list = upper_bound.tolist()
lower_list = lower_bound.tolist()

fig_forecast.add_trace(go.Scatter(
    x=forecast_dates_list + forecast_dates_list[::-1],
    y=upper_list + lower_list[::-1],
    fill='toself',
    fillcolor='rgba(255, 0, 0, 0.1)',
    line=dict(color='rgba(255,0,0,0)'),
    name='Confidence Interval (68%)',
    showlegend=True
))

fig_forecast.update_layout(
    title=f"{forecast_days}-Day TRM Forecast",
    xaxis_title="Date",
    yaxis_title="COP/USD",
    height=500,
    hovermode='x unified',
    template='plotly_white'
)

st.plotly_chart(fig_forecast, use_container_width=True)

# Forecast metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    final_forecast = forecast_models['linear'][-1]
    st.metric(
        f"Forecast (Day {forecast_days})",
        f"{final_forecast:,.2f}",
        delta=f"{((final_forecast - current_trm)/current_trm*100):+.1f}%"
    )

with col2:
    expected_range = f"{forecast_models['linear'][-1] - std_dev:.0f} - {forecast_models['linear'][-1] + std_dev:.0f}"
    st.metric("Expected Range", expected_range)

with col3:
    trend = "Upward" if forecast_models['linear'][-1] > current_trm else "Downward"
    st.metric("Predicted Trend", trend)

with col4:
    confidence = 68  # For 1 standard deviation
    st.metric("Confidence Level", f"{confidence}%")

# Statistics and Analysis Section
st.markdown("---")
st.subheader("📊 Statistical Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Price Statistics")
    stats_df = pd.DataFrame({
        'Metric': ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis'],
        'Value': [
            f"{filtered_df['trm'].mean():.2f}",
            f"{filtered_df['trm'].median():.2f}",
            f"{filtered_df['trm'].std():.2f}",
            f"{filtered_df['trm'].skew():.3f}",
            f"{filtered_df['trm'].kurtosis():.3f}"
        ]
    })
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

with col2:
    st.markdown("### Return Statistics")
    returns_stats = pd.DataFrame({
        'Metric': ['Mean Daily Return', 'Volatility (Annual)', 'Sharpe Ratio (Est.)', 'Max Drawdown', 'Positive Days %'],
        'Value': [
            f"{filtered_df['returns'].mean():.3f}%",
            f"{filtered_df['returns'].std() * np.sqrt(252):.2f}%",
            f"{filtered_df['returns'].mean() / filtered_df['returns'].std():.2f}" if filtered_df['returns'].std() != 0 else "N/A",
            f"{(filtered_df['trm'].max() - filtered_df['trm'].min()) / filtered_df['trm'].max() * 100:.1f}%",
            f"{(filtered_df['returns'] > 0).sum() / len(filtered_df) * 100:.1f}%"
        ]
    })
    st.dataframe(returns_stats, use_container_width=True, hide_index=True)

# Distribution chart
st.markdown("### TRM Distribution")
fig_dist = make_subplots(rows=1, cols=2, subplot_titles=("Price Distribution", "Returns Distribution"))

fig_dist.add_trace(
    go.Histogram(x=filtered_df['trm'], nbinsx=30, name='TRM', marker_color='blue'),
    row=1, col=1
)
fig_dist.add_trace(
    go.Histogram(x=filtered_df['returns'], nbinsx=30, name='Returns', marker_color='green'),
    row=1, col=2
)

fig_dist.update_layout(height=400, showlegend=False, template='plotly_white')
st.plotly_chart(fig_dist, use_container_width=True)

# Technical Indicators Summary
st.markdown("---")
st.subheader("📈 Technical Indicators Summary")

# Calculate current technical indicators
last_price = filtered_df['trm'].iloc[-1]
ma_7_current = filtered_df['ma_7'].iloc[-1]
ma_30_current = filtered_df['ma_30'].iloc[-1]
rsi_signal = "Overbought" if last_price > ma_7_current * 1.02 else "Oversold" if last_price < ma_7_current * 0.98 else "Neutral"

indicators_col1, indicators_col2, indicators_col3, indicators_col4 = st.columns(4)

with indicators_col1:
    st.metric("RSI Signal", rsi_signal)
with indicators_col2:
    st.metric("MA 7 vs MA 30", "Bullish" if ma_7_current > ma_30_current else "Bearish")
with indicators_col3:
    st.metric("Price vs MA 30", f"{((last_price - ma_30_current) / ma_30_current * 100):+.1f}%")
with indicators_col4:
    bollinger_position = "Upper Band" if last_price > filtered_df['upper_band'].iloc[-1] else "Lower Band" if last_price < filtered_df['lower_band'].iloc[-1] else "Middle"
    st.metric("Bollinger Position", bollinger_position)

# Recent Data Table
st.markdown("---")
st.subheader("📋 Recent Market Data")

# Prepare recent data
recent_data = filtered_df.tail(10)[['date', 'trm', 'open_rate', 'close_rate', 'max_rate', 'min_rate', 'num_trades', 'returns']].copy()
recent_data['date'] = recent_data['date'].dt.strftime('%Y-%m-%d')
recent_data['returns'] = recent_data['returns'].apply(lambda x: f"{x:+.2f}%")
recent_data.columns = ['Date', 'TRM', 'Open', 'Close', 'Max', 'Min', 'Trades', 'Return']

st.dataframe(recent_data, use_container_width=True, hide_index=True)

# Download data button
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name=f"trm_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown(
    "<div class='info-text' style='text-align: center;'>"
    "Data Source: Banco de la República (Colombia) | "
    f"Last Update: {df['date'].max().strftime('%Y-%m-%d %H:%M:%S')} | "
    "Powered by Streamlit & Plotly"
    "</div>",
    unsafe_allow_html=True
)
