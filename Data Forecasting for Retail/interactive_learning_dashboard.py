# Library Imports 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="📈 Retail Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/',
        'Report a bug': None,
        'About': "Learning Streamlit with Real Data Forecasting!"
    }
)

# Custom CSS Styling
st.markdown("""
<style>
    /* Ye CSS rules hain jo page ko style karte hain */
    
    /* Main container styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Custom metric cards */
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Success message styling */
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {  /* Streamlit sidebar class */
        background-color: #f0f2f6;
    }
    
    /* Button hover effects */
    .stButton > button {
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Caching Functions
@st.cache_data
def load_and_prepare_data():
    try:
        df = pd.read_csv('train_data.csv')
        st.success(f"✅ Data loaded successfully! {len(df)} records found.")
        
        df['date'] = pd.to_datetime(df['week'], format='%d-%m-%Y')
        
        daily_sales = df.groupby('date')['units_sold'].sum().reset_index()
        daily_sales.columns = ['ds', 'y']
        
        return df, daily_sales
    
    except FileNotFoundError:
        st.error("❌ Can't find train_data.csv! Check file location.")
        st.info("💡 File should be in same directory as this dashboard script")
        return None, None
    except Exception as e:
        st.error(f"❌ Data loading error: {str(e)}")
        return None, None

@st.cache_resource
def train_forecasting_model(daily_sales):
    split_point = int(len(daily_sales) * 0.8)
    train_data = daily_sales[:split_point]
    
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.95
    )
    
    with st.spinner("🤖 Training Prophet Model... Please wait"):
        model.fit(train_data)
    
    return model, train_data

# Sidebar COntrols Function
def create_sidebar_controls():
    st.sidebar.header("🎛️ Dashboard Controls")
    st.sidebar.markdown("*Customize your forecasting experience")
    
    st.sidebar.subheader("📅 Forecast Settings")
    forecast_days = st.sidebar.slider(
        "Duration of the forecast?",
        min_value=7,
        max_value=90,
        value=30,
        step=7,
        help="Move the slider for predictions"
    )
    
    st.sidebar.subheader("👁️ Display Options")
    
    show_confidence = st.sidebar.checkbox(
        "Show Confidence Intervals",
        value=True,
        help="For Prediction Uncertainty"
    )
    
    show_raw_data = st.sidebar.checkbox(
        "Show Raw Data Tables",
        value=False,
        help="View Original CSV data tables"
    )
    
    show_model_details = st.sidebar.checkbox(
        "Show Model Performance",
        value=True,
        help="RMSE, accuracy and test results"
    )
    
    st.sidebar.subheader("🎨 Chart Styling")
    
    chart_theme = st.sidebar.selectbox(
        "Chart Color Theme:",
        ["Default", "Dark", "Colorful", "Minimal"],
        help="Select Visual Theme for Chart"
    )
    
    chart_height = st.sidebar.slider(
        "Chert Height (picels):",
        min_value=300,
        max_value=800,
        value=500
        step=50
    )
    
    return {
        'forecast_days': forecast_days,
        'show_confidence': show_confidence,
        'show_raw_data': show_raw_data,
        'show_model_details': show_model_details,
        'chart_theme': chart_theme,
        'chart_height': chart_height,
    }
    
# Metrics Display Function
def displat_key_metrics(df, daily_sales):
    st.subheader("📊 Key Business Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_sales = df['units_sold'].sum()
    avg_daily_sales = daily_sales['y'].means()
    total_stores = df['store_id'].nunique()
    total_products = df['sku_id'].nunique()
    
    mid_point = len(daily_sales) // 2
    recent_avg = daily_sales['y'][mid_point:].mean()
    old_avg = daily_sales['y'][:mid_point].mean()
    growth_rate = ((recent_avg - old_avg) / old_avg) * 100
    
    with col1:
        st.metric(
            label="📦 Total Sales",
            value=f"{total_sales:,} units",
            delta=f"{growth_rate:+.1f}% trend",
            help="Overall Sales across all stores and products",
        )
        
    with col2:
        st.metric(
            label="📈 Daily Average",
            value=f"{avg_daily_sales:,.0f} units",
            delta=f"{recent_avg-old_avg :,.0f} recent",
            help="Average Daily Sales Volume",
        )
        
    with col3:
        st.metric(
            label="📦 Total Stores",
            value=f"{total_stores}",
            help="Number of Unique Retail Locations",
        )
        
    with col4:
        st.metric(
            label="🛍️ Product SKUs",
            value=f"{total_products:,} units",
            help="Number of Unique Products",
        )
        
# Forecasting Chart Fucntion
def create_forecast_chart(daily_sales, model, controls):
    st.subheader("🔮 Sales Forecasting with Prophet Model")
    
    future = model.make_furniture_dataframe(periods=controls['forecast_days'])
    forecast = model.predict(future)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_sales['ds'],
        y=daily_sales['y'],
        mode='lines+markers',
        name="📊 Historical"
    ))
        