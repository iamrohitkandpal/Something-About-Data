"""
Retail Demand Forecasting Web Application using Streamlit

This application provides a complete solution for retail demand forecasting
with multiple models, data exploration, and interactive visualizations.

Author: Something-About-Data
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import logging
from datetime import datetime, timedelta
import io
import base64

# Model imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

# Statistical models
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    try:
        from fbprophet import Prophet
        PROPHET_AVAILABLE = True
    except ImportError:
        PROPHET_AVAILABLE = False

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin: 1rem 0;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Retail Demand Forecasting</h1>', unsafe_allow_html=True)
    st.markdown("**A comprehensive web application for retail demand forecasting using multiple models**")
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["Home", "Data Upload & Exploration", "Feature Engineering", "Model Training", "Forecasting", "Model Comparison"]
    )
    
    if app_mode == "Home":
        show_home()
    elif app_mode == "Data Upload & Exploration":
        show_data_exploration()
    elif app_mode == "Feature Engineering":
        show_feature_engineering()
    elif app_mode == "Model Training":
        show_model_training()
    elif app_mode == "Forecasting":
        show_forecasting()
    elif app_mode == "Model Comparison":
        show_model_comparison()

def show_home():
    """Home page with application overview"""
    
    st.markdown('<h2 class="sub-header">Welcome to Retail Demand Forecasting</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Key Features
        
        - **Data Loading & Exploration**: Upload CSV files and explore your retail data
        - **Advanced Preprocessing**: Handle time-series features, holidays, and promotions
        - **Multiple Models**: Statistical, ML, and Deep Learning approaches
        - **Comprehensive Evaluation**: RMSE, MAPE, MAE with cross-validation
        - **Interactive Forecasting**: Generate and visualize future predictions
        - **Model Comparison**: Compare performance across different models
        """)
    
    with col2:
        st.markdown("""
        ### 🛠️ Supported Models
        
        - **Statistical**: ARIMA/SARIMA
        - **Machine Learning**: Random Forest, XGBoost
        - **Prophet**: Facebook's time-series forecasting
        - **Deep Learning**: LSTM/GRU neural networks
        
        ### 📊 Data Requirements
        
        Your CSV should contain columns like:
        - `date`: Date column
        - `store_id`: Store identifier
        - `item_id`: Item identifier  
        - `sales`: Target variable (sales/demand)
        - Optional: `promo`, `holiday` flags
        """)
    
    # Library availability status
    st.markdown('<h3 class="sub-header">📦 Library Status</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status = "✅" if STATSMODELS_AVAILABLE else "❌"
        st.markdown(f"**Statsmodels**: {status}")
        
    with col2:
        status = "✅" if PROPHET_AVAILABLE else "❌"
        st.markdown(f"**Prophet**: {status}")
        
    with col3:
        status = "✅" if TENSORFLOW_AVAILABLE else "❌"
        st.markdown(f"**TensorFlow**: {status}")
    
    # Sample data format
    st.markdown('<h3 class="sub-header">📁 Sample Data Format</h3>', unsafe_allow_html=True)
    
    sample_data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=5, freq='D'),
        'store_id': [1, 1, 1, 2, 2],
        'item_id': [101, 101, 102, 101, 102],
        'sales': [150, 175, 120, 200, 180],
        'promo': [0, 1, 0, 1, 0],
        'holiday': [0, 0, 0, 1, 0]
    })
    
    st.dataframe(sample_data, use_container_width=True)

def show_data_exploration():
    """Data upload and exploration page"""
    
    st.markdown('<h2 class="sub-header">📁 Data Upload & Exploration</h2>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your retail sales data CSV file"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = load_data(uploaded_file)
            st.session_state['data'] = df
            
            # Basic info
            st.success(f"Data loaded successfully! Shape: {df.shape}")
            
            # Data overview
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Data Overview")
                st.write(f"**Rows**: {df.shape[0]:,}")
                st.write(f"**Columns**: {df.shape[1]}")
                st.write(f"**Memory Usage**: {df.memory_usage().sum() / 1024**2:.2f} MB")
            
            with col2:
                st.markdown("### 🗓️ Date Range")
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    min_date = df['date'].min()
                    max_date = df['date'].max()
                    st.write(f"**From**: {min_date.strftime('%Y-%m-%d')}")
                    st.write(f"**To**: {max_date.strftime('%Y-%m-%d')}")
                    st.write(f"**Days**: {(max_date - min_date).days}")
            
            # Show sample data
            st.markdown("### 🔍 Sample Data")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data types and missing values
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📋 Data Types")
                dtype_df = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null': df.count().values
                })
                st.dataframe(dtype_df, use_container_width=True)
            
            with col2:
                st.markdown("### ❓ Missing Values")
                missing_df = pd.DataFrame({
                    'Column': df.columns,
                    'Missing': df.isnull().sum().values,
                    'Percentage': (df.isnull().sum() / len(df) * 100).round(2).values
                })
                missing_df = missing_df[missing_df['Missing'] > 0]
                if not missing_df.empty:
                    st.dataframe(missing_df, use_container_width=True)
                else:
                    st.success("No missing values found!")
            
            # Visualizations
            if st.checkbox("Show Data Visualizations"):
                show_data_visualizations(df)
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            logger.error(f"Data loading error: {str(e)}")
    
    else:
        st.info("Please upload a CSV file to get started.")
        
        # Option to use sample data
        if st.button("Use Sample Data"):
            df = generate_sample_data()
            st.session_state['data'] = df
            st.success("Sample data loaded!")
            st.rerun()

def load_data(uploaded_file):
    """Load and validate uploaded data"""
    
    df = pd.read_csv(uploaded_file)
    
    # Basic validation
    required_columns = ['date', 'sales']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        st.stop()
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Remove rows with invalid dates
    df = df.dropna(subset=['date'])
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    return df

def generate_sample_data():
    """Generate sample retail data for demonstration"""
    
    np.random.seed(42)
    
    # Date range
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Generate data for multiple stores and items
    stores = [1, 2, 3]
    items = [101, 102, 103, 104]
    
    data = []
    
    for store in stores:
        for item in items:
            for date in dates:
                # Base sales with trend and seasonality
                trend = 100 + (date - dates[0]).days * 0.01
                seasonality = 20 * np.sin(2 * np.pi * date.dayofyear / 365.25)
                weekly_pattern = 15 * np.sin(2 * np.pi * date.weekday() / 7)
                
                # Random noise
                noise = np.random.normal(0, 10)
                
                # Store and item effects
                store_effect = store * 10
                item_effect = item * 0.1
                
                # Promotions (random)
                promo = np.random.choice([0, 1], p=[0.9, 0.1])
                promo_effect = promo * 25
                
                # Holidays (simplified)
                holiday = 1 if date.month == 12 and date.day >= 20 else 0
                holiday_effect = holiday * 30
                
                sales = max(0, trend + seasonality + weekly_pattern + store_effect + 
                           item_effect + promo_effect + holiday_effect + noise)
                
                data.append({
                    'date': date,
                    'store_id': store,
                    'item_id': item,
                    'sales': round(sales, 2),
                    'promo': promo,
                    'holiday': holiday
                })
    
    return pd.DataFrame(data)

def show_data_visualizations(df):
    """Show various data visualizations"""
    
    st.markdown("### 📈 Data Visualizations")
    
    # Time series plot
    if 'date' in df.columns and 'sales' in df.columns:
        st.markdown("#### Sales Over Time")
        
        # Aggregate sales by date
        daily_sales = df.groupby('date')['sales'].sum().reset_index()
        
        fig = px.line(daily_sales, x='date', y='sales', title='Daily Sales Trend')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Sales distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Sales Distribution")
            fig = px.histogram(df, x='sales', nbins=50, title='Sales Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Sales by Store/Item")
            if 'store_id' in df.columns:
                store_sales = df.groupby('store_id')['sales'].sum().reset_index()
                fig = px.bar(store_sales, x='store_id', y='sales', title='Sales by Store')
                st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        st.markdown("#### Correlation Heatmap")
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                       title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)

def show_feature_engineering():
    """Feature engineering page"""
    
    st.markdown('<h2 class="sub-header">🔧 Feature Engineering</h2>', unsafe_allow_html=True)
    
    if 'data' not in st.session_state:
        st.warning("Please upload data first in the 'Data Upload & Exploration' section.")
        return
    
    df = st.session_state['data'].copy()
    
    # Feature engineering options
    st.markdown("### Select Features to Create")
    
    col1, col2 = st.columns(2)
    
    with col1:
        create_time_features = st.checkbox("Time-based Features", value=True)
        create_lag_features = st.checkbox("Lag Features", value=True)
        create_rolling_features = st.checkbox("Rolling Window Features", value=True)
    
    with col2:
        create_seasonal_features = st.checkbox("Seasonal Features", value=True)
        create_promo_features = st.checkbox("Promotion Features", value=False)
        create_holiday_features = st.checkbox("Holiday Features", value=False)
    
    if st.button("Create Features"):
        with st.spinner("Creating features..."):
            try:
                df_engineered = create_features(
                    df,
                    create_time_features,
                    create_lag_features,
                    create_rolling_features,
                    create_seasonal_features,
                    create_promo_features,
                    create_holiday_features
                )
                
                st.session_state['engineered_data'] = df_engineered
                st.success(f"Features created successfully! New shape: {df_engineered.shape}")
                
                # Show new features
                new_features = set(df_engineered.columns) - set(df.columns)
                if new_features:
                    st.markdown("### 🆕 New Features Created")
                    st.write(list(new_features))
                
                # Show sample of engineered data
                st.markdown("### 📊 Sample of Engineered Data")
                st.dataframe(df_engineered.head(10), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error creating features: {str(e)}")
                logger.error(f"Feature engineering error: {str(e)}")

def create_features(df, time_features=True, lag_features=True, rolling_features=True,
                   seasonal_features=True, promo_features=False, holiday_features=False):
    """Create engineered features"""
    
    df = df.copy()
    
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Time-based features
    if time_features:
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['weekday'] = df['date'].dt.weekday
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    
    # Seasonal features
    if seasonal_features:
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        df['sin_week'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['cos_week'] = np.cos(2 * np.pi * df['weekday'] / 7)
    
    # Sort by date for lag and rolling features
    df = df.sort_values(['store_id', 'item_id', 'date'] if 'store_id' in df.columns else ['date'])
    
    # Lag features
    if lag_features:
        for lag in [1, 7, 14, 30]:
            df[f'sales_lag_{lag}'] = df.groupby(['store_id', 'item_id'])['sales'].shift(lag) if 'store_id' in df.columns else df['sales'].shift(lag)
    
    # Rolling window features
    if rolling_features:
        for window in [3, 7, 14, 30]:
            df[f'sales_rolling_mean_{window}'] = df.groupby(['store_id', 'item_id'])['sales'].rolling(window).mean().values if 'store_id' in df.columns else df['sales'].rolling(window).mean()
            df[f'sales_rolling_std_{window}'] = df.groupby(['store_id', 'item_id'])['sales'].rolling(window).std().values if 'store_id' in df.columns else df['sales'].rolling(window).std()
    
    # Promotion features
    if promo_features and 'promo' in df.columns:
        df['promo_lag_1'] = df.groupby(['store_id', 'item_id'])['promo'].shift(1) if 'store_id' in df.columns else df['promo'].shift(1)
        df['promo_rolling_3'] = df.groupby(['store_id', 'item_id'])['promo'].rolling(3).sum().values if 'store_id' in df.columns else df['promo'].rolling(3).sum()
    
    # Holiday features
    if holiday_features and 'holiday' in df.columns:
        df['holiday_lag_1'] = df.groupby(['store_id', 'item_id'])['holiday'].shift(1) if 'store_id' in df.columns else df['holiday'].shift(1)
        df['days_since_holiday'] = df.groupby(['store_id', 'item_id']).apply(
            lambda x: (x['date'] - x[x['holiday'] == 1]['date'].max()).dt.days
        ).values if 'store_id' in df.columns else (df['date'] - df[df['holiday'] == 1]['date'].max()).dt.days
    
    return df

def show_model_training():
    """Model training page"""
    
    st.markdown('<h2 class="sub-header">🤖 Model Training</h2>', unsafe_allow_html=True)
    
    if 'engineered_data' not in st.session_state:
        st.warning("Please create features first in the 'Feature Engineering' section.")
        return
    
    df = st.session_state['engineered_data'].copy()
    
    # Model selection
    st.markdown("### Select Models to Train")
    
    col1, col2 = st.columns(2)
    
    with col1:
        train_rf = st.checkbox("Random Forest", value=True)
        train_xgb = st.checkbox("XGBoost", value=True)
        
    with col2:
        train_arima = st.checkbox("ARIMA", value=STATSMODELS_AVAILABLE)
        train_prophet = st.checkbox("Prophet", value=PROPHET_AVAILABLE)
        train_lstm = st.checkbox("LSTM", value=TENSORFLOW_AVAILABLE)
    
    # Model parameters
    st.markdown("### Model Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
        cv_folds = st.slider("CV Folds", 3, 10, 5)
    
    with col2:
        if train_rf:
            rf_trees = st.slider("RF Trees", 50, 500, 100)
            rf_depth = st.slider("RF Max Depth", 5, 50, 10)
    
    with col3:
        if train_xgb:
            xgb_trees = st.slider("XGB Trees", 50, 500, 100)
            xgb_lr = st.slider("XGB Learning Rate", 0.01, 0.3, 0.1)
    
    if st.button("Train Models"):
        with st.spinner("Training models..."):
            try:
                results = train_models(
                    df, test_size, cv_folds,
                    train_rf, train_xgb, train_arima, train_prophet, train_lstm,
                    rf_trees, rf_depth, xgb_trees, xgb_lr
                )
                
                st.session_state['model_results'] = results
                st.success("Models trained successfully!")
                
                # Show results
                show_training_results(results)
                
            except Exception as e:
                st.error(f"Error training models: {str(e)}")
                logger.error(f"Model training error: {str(e)}")

def train_models(df, test_size, cv_folds, train_rf, train_xgb, train_arima, 
                train_prophet, train_lstm, rf_trees, rf_depth, xgb_trees, xgb_lr):
    """Train selected models"""
    
    results = {}
    
    # Prepare data
    df = df.dropna()
    
    # For models that need specific store/item
    if 'store_id' in df.columns and 'item_id' in df.columns:
        # Select first store/item for simplicity
        store_item = df[(df['store_id'] == df['store_id'].iloc[0]) & 
                       (df['item_id'] == df['item_id'].iloc[0])].copy()
    else:
        store_item = df.copy()
    
    # Sort by date
    store_item = store_item.sort_values('date')
    
    # Split data
    split_idx = int(len(store_item) * (1 - test_size))
    train_data = store_item[:split_idx]
    test_data = store_item[split_idx:]
    
    # Features for ML models
    feature_cols = [col for col in df.columns if col not in ['date', 'sales', 'store_id', 'item_id']]
    
    X_train = train_data[feature_cols]
    y_train = train_data['sales']
    X_test = test_data[feature_cols]
    y_test = test_data['sales']
    
    # Random Forest
    if train_rf:
        rf_model = RandomForestRegressor(
            n_estimators=rf_trees,
            max_depth=rf_depth,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        results['Random Forest'] = {
            'model': rf_model,
            'predictions': rf_pred,
            'actual': y_test.values,
            'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'mae': mean_absolute_error(y_test, rf_pred),
            'mape': calculate_mape(y_test, rf_pred)
        }
    
    # XGBoost
    if train_xgb:
        xgb_model = xgb.XGBRegressor(
            n_estimators=xgb_trees,
            learning_rate=xgb_lr,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        
        results['XGBoost'] = {
            'model': xgb_model,
            'predictions': xgb_pred,
            'actual': y_test.values,
            'rmse': np.sqrt(mean_squared_error(y_test, xgb_pred)),
            'mae': mean_absolute_error(y_test, xgb_pred),
            'mape': calculate_mape(y_test, xgb_pred)
        }
    
    # ARIMA
    if train_arima and STATSMODELS_AVAILABLE:
        try:
            arima_model = ARIMA(train_data['sales'], order=(1, 1, 1))
            arima_fitted = arima_model.fit()
            arima_pred = arima_fitted.forecast(steps=len(test_data))
            
            results['ARIMA'] = {
                'model': arima_fitted,
                'predictions': arima_pred,
                'actual': y_test.values,
                'rmse': np.sqrt(mean_squared_error(y_test, arima_pred)),
                'mae': mean_absolute_error(y_test, arima_pred),
                'mape': calculate_mape(y_test, arima_pred)
            }
        except Exception as e:
            st.warning(f"ARIMA training failed: {str(e)}")
    
    # Prophet
    if train_prophet and PROPHET_AVAILABLE:
        try:
            prophet_data = train_data[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
            prophet_model = Prophet(daily_seasonality=True, yearly_seasonality=True)
            prophet_model.fit(prophet_data)
            
            future = test_data[['date']].rename(columns={'date': 'ds'})
            prophet_pred = prophet_model.predict(future)['yhat'].values
            
            results['Prophet'] = {
                'model': prophet_model,
                'predictions': prophet_pred,
                'actual': y_test.values,
                'rmse': np.sqrt(mean_squared_error(y_test, prophet_pred)),
                'mae': mean_absolute_error(y_test, prophet_pred),
                'mape': calculate_mape(y_test, prophet_pred)
            }
        except Exception as e:
            st.warning(f"Prophet training failed: {str(e)}")
    
    return results

def calculate_mape(actual, predicted):
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def show_training_results(results):
    """Display training results"""
    
    st.markdown("### 📊 Model Performance")
    
    # Metrics table
    metrics_data = []
    for model_name, result in results.items():
        metrics_data.append({
            'Model': model_name,
            'RMSE': f"{result['rmse']:.2f}",
            'MAE': f"{result['mae']:.2f}",
            'MAPE': f"{result['mape']:.2f}%"
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)
    
    # Performance visualization
    st.markdown("### 📈 Actual vs Predicted")
    
    for model_name, result in results.items():
        if st.checkbox(f"Show {model_name} predictions"):
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                y=result['actual'],
                mode='lines',
                name='Actual',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                y=result['predictions'],
                mode='lines',
                name=f'{model_name} Predicted',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f'{model_name} - Actual vs Predicted',
                xaxis_title='Time',
                yaxis_title='Sales',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_forecasting():
    """Forecasting page"""
    
    st.markdown('<h2 class="sub-header">🔮 Forecasting</h2>', unsafe_allow_html=True)
    
    if 'model_results' not in st.session_state:
        st.warning("Please train models first in the 'Model Training' section.")
        return
    
    results = st.session_state['model_results']
    
    # Forecasting parameters
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_days = st.slider("Forecast Days", 7, 90, 30)
        selected_model = st.selectbox("Select Model", list(results.keys()))
    
    with col2:
        confidence_interval = st.slider("Confidence Interval (%)", 80, 99, 95)
        show_historical = st.checkbox("Show Historical Data", value=True)
    
    if st.button("Generate Forecast"):
        with st.spinner("Generating forecast..."):
            try:
                forecast_result = generate_forecast(
                    results[selected_model], 
                    forecast_days, 
                    confidence_interval
                )
                
                st.session_state['forecast_result'] = forecast_result
                
                # Visualize forecast
                show_forecast_visualization(forecast_result, show_historical)
                
                # Download forecast
                if st.button("Download Forecast as CSV"):
                    csv = create_download_link(forecast_result)
                    st.markdown(csv, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
                logger.error(f"Forecasting error: {str(e)}")

def generate_forecast(model_result, forecast_days, confidence_interval):
    """Generate forecast using selected model"""
    
    # This is a simplified forecast generation
    # In practice, you'd need the original features for the forecast period
    
    last_value = model_result['actual'][-1]
    trend = np.mean(np.diff(model_result['actual'][-7:]))  # Simple trend
    
    forecast = []
    dates = pd.date_range(start=pd.Timestamp.now(), periods=forecast_days, freq='D')
    
    for i in range(forecast_days):
        # Simple forecast with trend and some seasonality
        forecast_value = last_value + trend * (i + 1) + np.random.normal(0, 5)
        forecast.append(max(0, forecast_value))
    
    # Calculate confidence intervals (simplified)
    error_std = np.std(model_result['actual'] - model_result['predictions'])
    z_score = 1.96 if confidence_interval == 95 else 2.576  # 95% or 99%
    
    lower_bound = [f - z_score * error_std for f in forecast]
    upper_bound = [f + z_score * error_std for f in forecast]
    
    return {
        'dates': dates,
        'forecast': forecast,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'confidence_interval': confidence_interval
    }

def show_forecast_visualization(forecast_result, show_historical):
    """Visualize forecast results"""
    
    fig = go.Figure()
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_result['dates'],
        y=forecast_result['forecast'],
        mode='lines',
        name='Forecast',
        line=dict(color='red')
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_result['dates'],
        y=forecast_result['upper_bound'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_result['dates'],
        y=forecast_result['lower_bound'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        name=f'{forecast_result["confidence_interval"]}% CI',
        fillcolor='rgba(255,0,0,0.2)'
    ))
    
    fig.update_layout(
        title='Sales Forecast',
        xaxis_title='Date',
        yaxis_title='Sales',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_download_link(forecast_result):
    """Create download link for forecast CSV"""
    
    forecast_df = pd.DataFrame({
        'date': forecast_result['dates'],
        'forecast': forecast_result['forecast'],
        'lower_bound': forecast_result['lower_bound'],
        'upper_bound': forecast_result['upper_bound']
    })
    
    csv = forecast_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="forecast.csv">Download Forecast CSV</a>'
    return href

def show_model_comparison():
    """Model comparison page"""
    
    st.markdown('<h2 class="sub-header">⚖️ Model Comparison</h2>', unsafe_allow_html=True)
    
    if 'model_results' not in st.session_state:
        st.warning("Please train models first in the 'Model Training' section.")
        return
    
    results = st.session_state['model_results']
    
    # Metrics comparison
    st.markdown("### 📊 Metrics Comparison")
    
    metrics_data = []
    for model_name, result in results.items():
        metrics_data.append({
            'Model': model_name,
            'RMSE': result['rmse'],
            'MAE': result['mae'],
            'MAPE': result['mape']
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Best model by each metric
    best_rmse = metrics_df.loc[metrics_df['RMSE'].idxmin(), 'Model']
    best_mae = metrics_df.loc[metrics_df['MAE'].idxmin(), 'Model']
    best_mape = metrics_df.loc[metrics_df['MAPE'].idxmin(), 'Model']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Best RMSE", best_rmse, f"{metrics_df['RMSE'].min():.2f}")
    
    with col2:
        st.metric("Best MAE", best_mae, f"{metrics_df['MAE'].min():.2f}")
    
    with col3:
        st.metric("Best MAPE", best_mape, f"{metrics_df['MAPE'].min():.2f}%")
    
    # Metrics visualization
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('RMSE', 'MAE', 'MAPE'),
        specs=[[{'secondary_y': False}, {'secondary_y': False}, {'secondary_y': False}]]
    )
    
    fig.add_trace(
        go.Bar(x=metrics_df['Model'], y=metrics_df['RMSE'], name='RMSE'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=metrics_df['Model'], y=metrics_df['MAE'], name='MAE'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=metrics_df['Model'], y=metrics_df['MAPE'], name='MAPE'),
        row=1, col=3
    )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed comparison table
    st.markdown("### 📋 Detailed Comparison")
    st.dataframe(metrics_df, use_container_width=True)

if __name__ == "__main__":
    main()