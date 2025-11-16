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
        
        if df.empty:
            st.error("❌ Dataset is empty!")
            return None, None
        
        required_columns = ['week', 'units_sold', 'store_id', 'sku_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            return None, None
        
        if (df['units_sold'] < 0).any():
            st.warning("⚠️ Found negative sales values. Cleaning data...")
            df = df[df['units_sold'] >= 0]
        
        if len(df) < 100:
            st.warning("⚠️ Dataset too small for reliable forecasting")
            
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
    
def load_and_prepare_data_with_upload(uploaded_file):
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.info(f"📂 Using uploaded file: {uploaded_file.name}")
        else:
            df = pd.read_csv('train_data.csv')
            st.info("📂 Using default training dataset")
        
        if df.empty:
            st.error("❌ Dataset is empty!")
            return None, None
        
        required_columns = ['week', 'units_sold', 'store_id', 'sku_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.info("💡 Your CSV must contain: week, units_sold, store_id, sku_id")
            
            with st.expander("📋 See Required Data Format"):
                example_data = pd.DataFrame({
                    'record_ID': [1, 2, 3],
                    'week': ['17-01-2011', '17-01-2011', '17-01-2011'],
                    'store_id': [8091, 8091, 8095],
                    'sku_id': [216418, 216419, 216418],
                    'total_price': [99.04, 99.04, 99.04],
                    'base_price': [111.86, 99.04, 99.04],
                    'is_featured_sku': [0, 0, 0],
                    'is_display_sku': [0, 0, 0],
                    'units_sold': [20, 28, 99]
                })
                st.dataframe(example_data)
            return None, None
        
        if len(df) < 30:
            st.warning("⚠️ Dataset too small for reliable forecasting (minimum 30 records)")
            
        if len(df) > 1000000:
            st.warning("⚠️ Large dataset detected. Pracessing may take longer...)")
            
        if(df['units_sold'] < 0).any():
            st.warning("⚠️ Found negative sales values. Cleaning data...")
            df = df[df['units_sold'] >= 0]
        
        try:
            df['date'] = pd.to_datetime(df['week'], format='%d-%m-%Y')
        except:
            try:
                df['date'] = pd.to_datetime(df['week'])
            except:
                try:
                    df['date'] = pd.to_datetime(df['week'])
                except:
                    st.error("❌ Cannot parse date format. Please use DD-MM-YYYY format")
                    return None, None
        daily_sales = df.groupby('date')['units_sold'].sum().reset_index()
        daily_sales.columns = ['ds', 'y']
        
        st.success(f"✅ Data processed successfully! {len(df)} records, {len(daily_sales)} days")
        
        return df, daily_sales
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
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

# Sidebar Controls Function
def create_sidebar_controls():
    st.sidebar.header("🎛️ Dashboard Controls")
    st.sidebar.markdown("*Customize your forecasting experience*")
    
    st.sidebar.subheader("📅 Forecast Settings")
    forecast_days = st.sidebar.slider(
        "Duration of the forecast?",
        min_value=7,
        max_value=90,
        value=30,
        step=7,
        help="Move the slider for predictions"
    )
    
    # Advanced Forecasting Options
    st.sidebar.subheader("🔬 Advanced Options")
    
    model_type = st.sidebar.selectbox(
        "Choose Forecasting Model:",
        ["Prophet (Default)", "Prophet with Holidays", "Prophet Enhanced"],
        help="Select the forecasting algorithm"
    )
    
    confidence_level = st.sidebar.slider(
        "Confidence Interval:",
        min_value=80,
        max_value=99,
        value=95,
        help="Statistical confidence level for predictions"
    )
    
    include_holidays = st.sidebar.checkbox(
        "Include Holiday Effects",
        help="Account for holiday sales patterns"
    )
    
    seasonal_adjustment = st.sidebar.selectbox(
        "Seasonal Adjustment:",
        ["Auto", "Weekly", "Monthly", "Quarterly"],
        help="Type of seasonality to emphasize"
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
    
    show_business_dashboard = st.sidebar.checkbox(
        "Show Business Dashboard",
        value=True,
        help="Executive KPIs and business metrics"
    )
    
    show_data_quality = st.sidebar.checkbox(
        "Show Data Quality Report",
        value=False,
        help="Data completeness and quality assessment"
    )
    
    show_alerts = st.sidebar.checkbox(
        "Show Business Alerts",
        value=True,
        help="Automated alerts for business anomalies"
    )
    
    st.sidebar.subheader("🎨 Chart Styling")
    
    chart_theme = st.sidebar.selectbox(
        "Chart Color Theme:",
        ["Default", "Dark", "Colorful", "Minimal"],
        help="Select Visual Theme for Chart"
    )
    
    chart_height = st.sidebar.slider(
        "Chart Height (pixels):",  
        min_value=300,
        max_value=800,
        value=500,
        step=50
    )
    
    return {
        'forecast_days': forecast_days,
        'model_type': model_type,
        'confidence_level': confidence_level,
        'include_holidays': include_holidays,
        'seasonal_adjustment': seasonal_adjustment,
        'show_confidence': show_confidence,
        'show_raw_data': show_raw_data,
        'show_model_details': show_model_details,
        'show_business_dashboard': show_business_dashboard,
        'show_data_quality': show_data_quality,
        'show_alerts': show_alerts,
        'chart_theme': chart_theme,
        'chart_height': chart_height,
    }
    
# Metrics Display Function
def display_key_metrics(df, daily_sales):
    st.subheader("📊 Key Business Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_sales = df['units_sold'].sum()
    avg_daily_sales = daily_sales['y'].mean()
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
            delta=f"{recent_avg-old_avg:,.0f} recent",
            help="Average Daily Sales Volume",
        )
        
    with col3:
        st.metric(
            label="🏪 Total Stores",
            value=f"{total_stores}",
            help="Number of Unique Retail Locations",
        )
        
    with col4:
        st.metric(
            label="🛍️ Product SKUs",
            value=f"{total_products:,}", 
            help="Number of Unique Products",
        )
        
# Forecasting Chart Function
def create_forecast_chart(daily_sales, model, controls):
    st.subheader("🔮 Sales Forecasting with Prophet Model")
    
    future = model.make_future_dataframe(periods=controls['forecast_days'])
    forecast = model.predict(future)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_sales['ds'],
        y=daily_sales['y'],
        mode='lines+markers',
        name="📊 Historical Sales",
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=4),
        hovertemplate='<b>Date:</b> %{x}<br><b>Sales:</b> %{y:,.0f} units<extra></extra>'
    ))
    
    forecast_start = len(daily_sales)
    fig.add_trace(go.Scatter(
        x=forecast['ds'][forecast_start:],
        y=forecast['yhat'][forecast_start:],
        mode='lines+markers',
        name="🔮 Forecast",
        line=dict(color='#ff7f0e', width=3, dash='dash'),
        marker=dict(size=6, symbol='diamond'),
        hovertemplate='<b>Predicted Date:</b> %{x}<br><b>Forecast:</b> %{y:,.0f} units<extra></extra>'
    ))
    
    if controls['show_confidence']:
        fig.add_trace(go.Scatter(
            x=forecast['ds'][forecast_start:],
            y=forecast['yhat_upper'][forecast_start:],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['ds'][forecast_start:],
            y=forecast['yhat_lower'][forecast_start:],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255,127,14,0.2)',
            name='🎯 95% Confidence',
            hovertemplate='<b>Range:</b> %{y:,.0f} - upper bound<extra></extra>'
        ))
        
    if controls['chart_theme'] == 'Dark':
        template = 'plotly_dark'
        bg_color = '#2F3349'
    elif controls['chart_theme'] == 'Colorful':
        template = 'plotly'
        bg_color = '#F0F8FF'
    elif controls['chart_theme'] == 'Minimal':
        template = 'simple_white'
        bg_color = 'white'
    else:
        template = 'plotly'
        bg_color = 'white'
        
    fig.update_layout(
        title=f"📈 Sales Forecast for Next {controls['forecast_days']} Days",
        xaxis_title="📅 Date",
        yaxis_title="📦 Units Sold",
        height=controls['chart_height'],
        template=template,
        plot_bgcolor=bg_color,
        hovermode='x unified',
        showlegend=True,
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    return forecast

# Business Insights Function
def display_business_insights(forecast, daily_sales, controls):
    st.subheader("💼 Business Intelligence & Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📅 Next Week Detailed Forecast")
        
        next_week = forecast.tail(7)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].round(0)
        
        for idx, row in next_week.iterrows():
            day_name = row['ds'].strftime('%A')
            date = row['ds'].strftime('%d-%m-%Y')
            predicted_sales = row['yhat']
            
            if predicted_sales > daily_sales['y'].quantile(0.8):
                emoji = "🔥"  # High sales day
                color = "🟢"
            elif predicted_sales < daily_sales['y'].quantile(0.2):
                emoji = "📉"  # Low sales day
                color = "🔴"
            else:
                emoji = "📈"  # Normal sales day
                color = "🟡"
                
            st.markdown(f"""
            **{emoji} {day_name} ({date})**
            {color} Predicted: **{predicted_sales:,.0f} units**
            Range: {row['yhat_lower']:,.0f} - {row['yhat_upper']:,.0f}
            """)
            
        weekly_total = next_week['yhat'].sum()
        st.success(f"📊 **Total Next Week**: {weekly_total:,.0f} units")
        
    with col2:
        st.markdown("### 🎯 Actionable Business Recommendations")
        
        avg_forecast = forecast.tail(controls['forecast_days'])['yhat'].mean()
        historical_avg = daily_sales['y'].mean()
        
        peak_day = forecast.tail(controls['forecast_days']).loc[
            forecast.tail(controls['forecast_days'])['yhat'].idxmax()
        ]
        
        low_day = forecast.tail(controls['forecast_days']).loc[
            forecast.tail(controls['forecast_days'])['yhat'].idxmin()
        ]
        
        # Trend analysis
        if avg_forecast > historical_avg * 1.05:
            trend_msg = "📈 **Growing Trend** - Sales increasing!"
            trend_color = "success"
        elif avg_forecast < historical_avg * 0.95:
            trend_msg = "📉 **Declining Trend** - Need attention!"
            trend_color = "warning"  
        else:
            trend_msg = "➡️ **Stable Trend** - Consistent performance"
            trend_color = "info"
            
        eval(f"st.{trend_color}(trend_msg)")
        
        st.info(f"""
        **📊 Forecast Summary:**
        - Average daily: {avg_forecast:,.0f} units
        - vs Historical: {((avg_forecast-historical_avg)/historical_avg)*100:+.1f}%
        - Total {controls['forecast_days']} days: {forecast.tail(controls['forecast_days'])['yhat'].sum():,.0f} units        
        """)
        
        st.warning(f"""
        **🎯 Peak Sales Day:**
        📅 {peak_day['ds'].strftime('%A, %d %B')}
        📈 Expected: {peak_day['yhat']:,.0f} units
        
        💡 *Recommendation: Ensure adequate inventory!*
        """)
        
        st.error(f"""
        **📉 Lowest Sales Day:**
        📅 {low_day['ds'].strftime('%A, %d %B')} 
        📉 Expected: {low_day['yhat']:,.0f} units
        
        💡 *Recommendation: Plan promotions or marketing!*
        """)
        
# Business Intelligence Dashboard
def create_business_dashboard(df, forecast, controls):
    if not controls['show_business_dashboard']:
        return
    
    st.subheader("🎯 Executive Business Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    avg_price = df['total_price'].mean()
    forecast_revenue = forecast.tail(controls['forecast_days'])['yhat'].sum() * avg_price
    
    with col1:
        st.metric(
            "💰 Projected Revenue",
            f"${forecast_revenue:,.0f}",
            help="Revenue forecast based on predicted sales"
        )
    with col2:
        max_daily = forecast.tail(controls['forecast_days'])['yhat'].max()
        st.metric(
            "📦 Peak Inventory Need",
            f"{max_daily:,.0f} units",
            help="Maximum single-day inventory requirement"
        )    
    with col3:
        avg_forecast = forecast.tail(controls['forecast_days'])['yhat'].mean()
        historical_avg = df.groupby('date')['units_sold'].sum().mean()
        growth = ((avg_forecast - historical_avg) / historical_avg) * 100
        st.metric(
            "📈 Forecast Growth",
            f"{growth:+.1f}%",
            delta=f"vs historical_avg",
            help="Growth compare to historical average"
        )
    with col4:
        total_forecast = forecast.tail(controls['forecast_days'])['yhat'].sum()
        st.metric(
            "📊 Total Forecast",
            f"{total_forecast:,.0f} units",
            help=f"Total units for next {controls['forecast_days']} days"
        )
        
    st.markdown("#### 🏪 Store Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        store_performance = df.groupby('store_id').agg({
            'units_sold': 'sum',
            'total_price': 'mean'
        }).round(2)

        fig = px.scatter(
            store_performance,
            x='total_price',
            y='units_sold',
            title="Store Performance: Price vs Volume",
            labels={'total_price': 'Average Price', 'units_sold': 'Total Units Sold'},
            hover_data={'total_price': ':2f', 'units_sold': ':,'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        top_stores = df.groupby('store_id') ['units_sold'].sum().sort_values(ascending=False).head(10)
        
        fig = px.bar(
            x=top_stores.index.astype(str),
            y = top_stores.values,
            title="🏆 Top 10 Performing Stores",
            labels={'x': 'Sore ID', 'y': 'Total Units Sold'},
            color=top_stores.values,
            color_continuous_scale='Blues'
        )   
        fig.update_layout(height=400, showLegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 🛍️ Product Performance Indights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        featured_performance = df.groupby('is_featured_sku').agg({
            'units_sold': ['sum', 'mean'],
            'total_price': 'mean'
        }).round(2)
        
        featured_data = pd.DataFrame({
            'Category': ['Regular Products', 'Featured Products'],
            'Total_Sales': [
                featured_performance.loc[0, ('units_sold', 'sum')],
                featured_performance.loc[1, ('units_sold', 'sum')] if 1 in featured_performance.index else 0,
            ],
            'Avg_Sales': [
                featured_performance.loc[0, ('units_sold', 'mean')],
                featured_performance.loc[1, ('units_sold', 'mean')] if 1 in featured_performance.index else 0,
            ]
        })
        
        fig = px.bar(
            featured_data,
            x="Category",
            y='Total_Sales',
            title='📊 Featured vs Regular Products',
            color='Category',
            color_discrete_map={
                'Regular Products': '#3498db',
                'Featured Products': '#e74c3c',
            }
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        price_sales = df.groupby(pd.cut(df['total_price'], bins=5)).agg({
            'units_sold': 'sum'
        }).reset_index()
        price_sales['price_range'] = price_sales['total_price'].astype(str)
        
        fig = px.line(
            price_sales,
            x='price_range',
            y='units_sold',
            title="💰 Price Range vs Sales Volume",
            markers=True
        )
        fig.update_layout(height=350)
        fig.update_xaxes(title="Price Range")
        fig.update_yaxes(title="Units Sold")
        st.plotly_chart(fig, use_container_width=True)
            
# Model performance Function
def display_model_performance(model, daily_sales, controls):
    if not controls['show_model_details']:
        return
    
    st.subheader("🎯 Model Performance Analytics")
    
    split_point = int(len(daily_sales) * 0.8)
    test_data = daily_sales[split_point:].copy()
    
    if len(test_data) == 0:
        st.warning("⚠️ Not enough data for performance testing")
        return 
    
    # Test Predictions
    test_future = model.make_future_dataframe(periods=0)
    test_forecast = model.predict(test_future)
    test_predictions = test_forecast.tail(len(test_data))['yhat'].values
    test_actual = test_data['y'].values
    
    # Performance Metrics
    rmse = np.sqrt(mean_squared_error(test_actual, test_predictions))
    mae = np.mean(np.abs(test_actual - test_predictions))
    mape = np.mean(np.abs((test_actual - test_predictions) / test_actual)) * 100
    accuracy = max(0, 100 - mape)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📏 RMSE", f"{rmse:,.0f}", help="Root Mean Square Error - lower is better")
    with col2:
        st.metric("📊 MAE", f"{mae:,.0f}", help="Mean Absolute Error")
    with col3:
        # Color based on accuracy
        if accuracy > 90:
            st.metric("🎯 Accuracy", f"{accuracy:.1f}%", delta="Excellent", delta_color="normal")
        elif accuracy > 80:
            st.metric("🎯 Accuracy", f"{accuracy:.1f}%", delta="Good", delta_color="normal")
        elif accuracy > 70:
            st.metric("🎯 Accuracy", f"{accuracy:.1f}%", delta="Fair", delta_color="inverse")
        else:
            st.metric("🎯 Accuracy", f"{accuracy:.1f}%", delta="Needs Improvement", delta_color="inverse")
    with col4:
        st.metric("📅 Test Days", f"{len(test_data)}")
        
    st.markdown("### 📊 Actual vs Predicted Comparison")
    
    comparison_fig = go.Figure()
    
    comparison_fig.add_trace(go.Scatter(
        x=test_data['ds'],
        y=test_actual,
        mode='lines+markers',
        name=" ✅Actual Sales",
        line=dict(color='green', width=3),
        marker=dict(size=8)
    ))
    
    comparison_fig.add_trace(go.Scatter(
        x=test_data['ds'],
        y=test_predictions,
        mode='lines+markers',
        name="🔮 Predicted Sales",
        line=dict(color='red', width=3, dash='dot'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    comparison_fig.update_layout(
        title=f"Model Accuracy Test (RMSE: {rmse:.1f}, Accuracy: {accuracy:.1f}%)",
        xaxis_title="Date",
        yaxis_title="Sales Units",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(comparison_fig, use_container_width=True)
    
# Data Explorer Function
def display_data_explorer(df, daily_sales, controls):
    if not controls['show_raw_data']:
        return
    
    st.subheader("📋 Data Explorer & Raw Tables")
    
    tab1, tab2, tab3 = st.tabs(["📈 Daily Sales Data", "🛍️ Original Dataset", "🔍 Data Analysis"])
    
    with tab1:
        st.markdown("**📊 Aggregated daily sales data (prepared for Prophet model):**")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            date_range = st.date_input(
                "Select Date Range:",
                value=(daily_sales['ds'].min(), daily_sales['ds'].max()),
                min_value=daily_sales['ds'].min(),
                max_value=daily_sales['ds'].max(),
            )
        with col2:
            sales_threshold = st.number_input(
                "Min Sales Filter:",
                min_value=0,
                value=0,
                step=1000                             
            )
        
        if len(date_range) == 2:
            filtered_daily = daily_sales[
                (daily_sales['ds'] >= pd.Timestamp(date_range[0])) &
                (daily_sales['ds'] <= pd.Timestamp(date_range[1])) &
                (daily_sales['y'] >= sales_threshold)
            ]    
        else:
            filtered_daily = daily_sales[daily_sales['y'] >= sales_threshold]
            
        st.dataframe(filtered_daily, use_container_width=True, height=300)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(filtered_daily))
        with col2:
            st.metric("Avg Sales", f"{filtered_daily['y'].mean():,.0f}")
        with col3:
            st.metric("Total Sales", f"{filtered_daily['y'].sum():,}")
            
        csv_daily = filtered_daily.to_csv(index=False)
        st.download_button(
            label="💾 Download Daily Sales CSV",
            data=csv_daily,
            file_name=f"daily_sales_filtered.csv",
            mime="text/csv"
        )
        
    with tab2:
        st.markdown("**🛍️ Original transaction-level dataset:**")
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            selected_stores = st.multiselect(
                "Filter by Stores:",
                options=sorted(df['store_id'].unique()),
                default=sorted(df['store_id'].unique())[:5],
            )
        with col2:
            search_sku = st.text_input("Search SKU ID:", "")
            
        filtered_df = df[df['store_id'].isin(selected_stores)]
        if search_sku:
            filtered_df = filtered_df[filtered_df['sku_id'].str.contains(search_sku, case=False, na=False)]
            
        display_df = filtered_df.head(1000)
        st.dataframe(display_df, use_container_width=True, height=300)
        
        if len(filtered_df) > 1000:
            st.info(f"📊 Showing first 1000 rows of {len(filtered_df):,} total records")
        
        st.markdown("**📊 Dataset Summary:**")
        col1, col2, col3, col4 = st.columns(4)
    
        with col1:
            st.metric("Total Records", f"{len(filtered_df):,}")
        with col2:
            st.metric("Unique Stores", filtered_df['store_id'].nunique())
        with col3:
            st.metric("Unique SKUs", filtered_df['sku_id'].nunique())
        with col4:
            st.metric("Total Units", f"{filtered_df['units_sold'].sum():,}")
            
    with tab3:
        st.markdown("**🔍 Quick Data Analysis**")
        
        store_sales = df.groupby('store_id')['units_sold'].sum().sort_values(ascending=False).head(10)
        
        fig_stores = px.bar(
            x=store_sales.index,
            y=store_sales.values,
            title="🏪 Top 10 Stores by Total Sales",
            labels={'x': "Store ID", 'y': 'Total Units Sold'}
        )
        st.plotly_chart(fig_stores, use_container_width=True)
        
        fig_dist = px.histogram(
            df,
            x='units_sold',
            nbins=50,
            title="📊 Sales Volume Distribution",
            labels={'units_sold': 'Units Sold per Transaction'}
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
# Enhanced Sidebar Controls
def create_enhanced_sidebar_controls():
    st.sidebar.header("🎛️ Dashboard Controls")
    st.sidebar.markdown("*Customize your forecasting experience*")
    
    # File Upload Section
    st.sidebar.subheader("📂 Data Source")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your CSV file",
        type=['csv'],
        help="Upload retail sales data or use default dataset"
    )
    
    st.sidebar.subheader("📅 Forecast Configuration")
    st.sidebar.info("📍 **Current Data Range**: Based on your latest sales data")
    
    forecast_method = st.sidebar.radio(
        "Choose Forecasting Method:",
        ["📊 Days from Last Date", "📅 Specific Date Range", "🎯 Next N Business Days"]
    )
    
    if forecast_method == "📊 Days from Last Date":
        forecast_days = st.sidebar.slider(
            "Forecast Duration (days from last data point):",
            min_value=7,
            max_value=90,
            value=30,
            step=7,
            help="Predict next N days from your latest sales date"
        )
        start_date = None
        end_date = None
    elif forecast_method == "📅 Specific Date Range":
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            start_date = st.sidebar.date_input("Start Date:")
        with col2:
            end_date = st.sidebar.date_input("End Date:")
            
        if start_date and end_date:
            forecast_days = (end_date - start_date).days
            if forecast_days <= 0:
                st.sidebar.error("End date must be after start date!")
                forecast_days = 7
        else:
            forecast_days = 7
    else:
        business_days = st.sidebar.slider(
            "Business Days to Forecast:",
            min_value=5,
            max_value=60,
            value=20,
            step=5,
            help="Exclude weekends from forecast period"
        )
        forecast_days = int(business_days * 1.4)
        start_date = None
        end_date = None
    
    return {
        'uploaded_file': uploaded_file,
        'forecast_method': forecast_method,
        'forecast_days': forecast_days,
        'start_date': start_date,
        'end_date': end_date
    }
    

# Export Functionality    
def create_export_section(forecast, daily_sales, controls):
    st.subheader("💾 Export Your Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Download Forecast Data")
        
        if st.button("📈 Generate Forecast CSV", type='primary'):
            forecast_export = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(controls['forecast_days'])
            forecast_export = forecast_export.round(0)
            forecast_export.columns = ['Date', 'Predicted_Sales', 'Lower_Confidence', 'Upper_Confidence']  # Fixed space
            
            csv_data = forecast_export.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv_data,
                file_name=f"sales_forecast_{controls['forecast_days']}_days.csv",
                mime="text/csv",
                key="forecast_csv"
            )
            
            st.success("✅ Forecast CSV ready for download!")  
            st.dataframe(forecast_export.head(), use_container_width=True)
            
    with col2:
        st.markdown("#### 📋 Business Report")
        
        if st.button("📄 Generate Business Report", type="secondary"):
            # Business report text banao
            avg_forecast = forecast.tail(controls['forecast_days'])['yhat'].mean()
            total_forecast = forecast.tail(controls['forecast_days'])['yhat'].sum()

            report = f"""
RETAIL SALES FORECAST REPORT
=====================================
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {controls['model_type']}
Confidence Level: {controls['confidence_level']}%

📊 EXECUTIVE SUMMARY:
• Forecast Period: {controls['forecast_days']} days
• Average Daily Forecast: {avg_forecast:,.0f} units
• Total Forecast Sales: {total_forecast:,.0f} units
• Historical Daily Average: {daily_sales['y'].mean():,.0f} units

📈 TREND ANALYSIS:
• Growth vs Historical: {((avg_forecast - daily_sales['y'].mean()) / daily_sales['y'].mean()) * 100:+.1f}%

🎯 NEXT WEEK FORECAST:
"""

            next_week = forecast.tail(7)[['ds', 'yhat']].round(0)
            for idx, row in next_week.iterrows():
                day_name = row['ds'].strftime('%A')
                date = row['ds'].strftime('%d-%m-%Y')
                report += f"\n● {day_name} ({date}): {row['yhat']:,.0f} units"

            report += f"\n\nTotal Next Week: {next_week['yhat'].sum():,.0f} units"
            report += f"""

💼 BUSINESS RECOMMENDATIONS:
• Monitor inventory levels for peak days
• Plan marketing campaigns for low-sales periods
• Ensure adequate staffing during high-demand forecasts
• Review supplier agreements for demand spikes

📊 MODEL PERFORMANCE:
• Algorithm: Facebook Prophet
• Confidence Interval: {controls['confidence_level']}%
• Seasonality: {controls['seasonal_adjustment']}
• Holiday Effects: {'Included' if controls['include_holidays'] else 'Not Included'}

---
Report generated by Retail Sales Forecasting Dashboard
Built with Python, Prophet, and Streamlit
"""
            st.download_button(
                label="📥 Download Business Report",
                data=report,
                file_name=f"business_report_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                key="business_report"
            )

            st.success("✅ Business report ready for download!")

# Data Quality Report
def create_data_quality_report(df):
    st.subheader("🔍 Data Quality Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('#### 📊 Data Completeness')
        missing_data = df.isnull().sum()
        completeness = (1 - missing_data / len(df)) * 100
        
        for col, pct in completeness.items():
            color = "🟢" if pct > 95 else "🟡" if pct > 90 else "🔴"
            st.write(f"{color} {col}: {pct:.1f}% complete")
    
    with col2:
        st.markdown("#### 📈 Statistical Summary")
        st.write("**Sales Distribution:**")
        st.write(f"Mean: {df['units_sold'].mean():.1f}")
        st.write(f"Median: {df['units_sold'].median():.1f}")
        st.write(f"Std Dev: {df['units_sold'].std():.1f}")
        st.write(f"Min: {df['units_sold'].min()}")
        st.write(f"Max: {df['units_sold'].max()}")
        
        st.write("**Price Statistics:**")
        st.write(f"Avg Price: ${df['total_price'].mean():.2f}")
        st.write(f"Price Range: {df['total_price'].min():.2f} - ${df['total_price'].max():.2f}")
        
    with col3:
        st.markdown('#### ⚠️ Data Issues')
        issues = []
        
        if(df['units_sold'] == 0).sum() > 0:
            issues.append(f"🟡 {(df['units_sold'] == 0).sum()} zero sales records")
            
        if df.duplicated().sum() > 0:
            issues.append(f"🟡 {df.duplicated().sum()} duplicate records")
            
        if (df['total_price'] <= 0).sum() > 0:
            issues.append(f"🔴 {(df['total_price'] <= 0).sum()} invalid price records")
            
        q1 =df['units_sold'].quantile(0.25)
        q3 =df['units_sold'].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df['units_sold'] < (q1 - 1.5 * iqr)) | (df['units_sold'] > (q3 + 1.5 * iqr))]
        
        if len(outliers) > 0:
            issues.append(f"🟡 {len(outliers)} potential outliers detected")
        
        if not issues:
            issues.append("🟢 No major data quality issues")
            
        for issue in issues:
            st.write(issue)
            
    st.markdown("#### 🎯 Overall Data Quality Score")
    
    completeness_score = completeness.mean()
    outlier_penalty = min(20, (len(outliers) / len(df)) * 100)
    missing_penalty = min(10, (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100)
    
    quality_score = max(0, completeness_score - outlier_penalty - missing_penalty)
    
    if quality_score >= 90:
        color = "🟢"
        status = "Excellent"
    elif quality_score >= 80:
        color = "🟡"
        status = "Good"
    else:
        color = "🔴"
        status = "Needs Improvement"
    
    st.metric(
        f"{color} Data Quality Score",
        f"{quality_score:.1f}/100",
        delta=status,
        help="Based on completeness, outliers, and missing data"
    )

            
# Real-Time Alert System
def create_alert_system(forecast, daily_sales, controls):
    if not controls['show_alerts']:
        return
    
    st.subheader("🚨 Business Alerts & Recommendations")
    
    alerts = []
    
    recent_forecast = forecast.tail(controls['forecast_days'])['yhat']
    historical_avg = daily_sales['y'].mean()
    
    # Alert for significant drops
    if recent_forecast.min() < historical_avg * 0.5:
        alerts.append({
            "type": "error",
            "title": "📉 Significant Sales Drop Predicted",
            "message": f"Sales may drop to {recent_forecast.min():,.0f} units (vs avg {historical_avg:,.0f})",
            "action": "Consider promotional campaigns or inventory adjustments"
        })
    
    # Alert for high demand
    if recent_forecast.max() > historical_avg * 1.5:
        alerts.append({
            "type": "warning",
            "title": "🔥 High Demand Period Ahead",
            "message": f"Peak demand of {recent_forecast.max():,.0f} units expected",
            "action": "Ensure adequate inventory and staffing"
        })
    
    # Alert for unusual variance
    forecast_std = recent_forecast.std()
    historical_std = daily_sales['y'].std()
    
    if forecast_std > historical_std * 1.5:
        alerts.append({
            "type": "info",
            "title": "📊 High Volatility Period",
            "message": f"Sales volatility is {(forecast_std/historical_std):.1f}x higher than historical",
            "action": "Prepare for variable demand patterns"
        })
        
    forecast_with_day = forecast.tail(controls['forecast_days']).copy()
    forecast_with_day['day_of_week'] = forecast_with_day['ds'].dt.day_name()
    
    weekend_avg = forecast_with_day[forecast_with_day['day_of_week'].isin(['Saturday', 'Sunday'])]['yhat'].mean()
    weekday_avg = forecast_with_day[~forecast_with_day['day_of_week'].isin(['Saturday', 'Sunday'])]['yhat'].mean()
    
    if not np.isnan(weekday_avg) and not np.isnan(weekend_avg):
        if weekend_avg > weekday_avg * 1.3:
            alerts.append({
                "type": "info",
                "title": "🎉 Strong Weekend Performance Expected",
                "message": f"Weekend sales ({weekend_avg:.0f}) vs weekday ({weekday_avg:.0f})",
                "action": "Optimize weekend staffing and inventory"
            })
        
    # Display alerts
    if alerts:
        for alert in alerts:
            if alert["type"] == "error":
                st.error(f"**{alert['title']}**\n\n{alert['message']}\n\n💡 *Action Required: {alert['action']}*")
            elif alert["type"] == "warning":
                st.warning(f"**{alert['title']}**\n\n{alert['message']}\n\n💡 *Recommended Action: {alert['action']}*")
            else:
                st.info(f"**{alert['title']}**\n\n{alert['message']}\n\n💡 *Consider: {alert['action']}*")
    else:
        st.success("✅ **All Clear!** No significant business alerts at this time.")


# MAIN FUNCTION
def main():
    st.title("📈 Retail Sales Forecasting Dashboard")
    st.markdown("---")
    
    # 📊LOAD DATA
    with st.spinner("📂 Loading data from CSV..."):
        df, daily_sales = load_and_prepare_data()
    
    # Check if data loaded successfully
    if df is None or daily_sales is None:
        st.stop() # Stop Dashboard if no data available
    
    # 🎛️ GET USER CONTROLS
    controls = create_sidebar_controls()
    
    # 📊 DISPLAY METRICS 
    display_key_metrics(df, daily_sales)
    
    st.markdown("---")
    
    # 🤖 TRAIN MODEL 
    with st.spinner("🧠 Training forecasting model..."):
        model, train_data = train_forecasting_model(daily_sales)
        
    st.success("✅ Model trained successfully! Ready for forecasting.")
    
    # 📈 CREATE FORECAST CHART
    forecast = create_forecast_chart(daily_sales, model, controls)
    
    st.markdown("---")
    
    # 💼 BUSINESS INSIGHTS
    display_business_insights(forecast, daily_sales, controls)
    
    st.markdown("---")
    
    # 🎯 MODEL PERFORMANCE
    display_model_performance(model, daily_sales, controls)
    
    st.markdown("---")
    
    # 📋 DATA EXPLORER
    display_data_explorer(df, daily_sales, controls)
    
    st.markdown("---")
    
    # 💾 EXPORT SECTION
    create_export_section(forecast, daily_sales, controls)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <p style='color: #666; font-size: 14px;'>
            💡 Built with: Streamlit 🎨 • Prophet 🔮 • Plotly 📊 • Python 🐍 • Love ❤️
        </p>
    </div>
    """, unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()