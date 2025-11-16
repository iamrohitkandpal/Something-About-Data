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
            st.error(f"❌ Missing reuired columns: {missing_columns}")
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
        "Chart Height (pixels):",  
        min_value=300,
        max_value=800,
        value=500,
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
        
# Advanced Forecasting Option
def create_advanced_forecasting(dail_sales, controls):
    st.subheader("🔬 Advanced Forecasting Options")

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
    st.subheader("🎯 Executive Business Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    avg_price = df['total_price'].mean()
    forecast_revenue = forecast.tail(controls['forecast_days'])['yhat']
    
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
            help="Maximum single-dya inventory requirement"
        )    
        
    st.markdown("#### 🏪 Store Performance Analysis")
    store_performance = df.groupby('store_id').agg({
        'units_sold': 'sum',
        'total_price': 'mean'
    }).round(2)
    
    fig = px.scatter(
        store_performance,
        x='total_price',
        y='units_sold',
        title="Store Performance: Price vs Volume",
        labels={'total_price': 'Average Price', 'units_sold': 'Total Units Sold'}
    )
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

📊 EXECUTIVE SUMMARY:
● Forecast Period: {controls['forecast_days']} days
● Average Daily Forecast: {avg_forecast:,.0f} units
● Total Forecast Sales: {total_forecast:,.0f} units
● Historical Daily Average: {daily_sales['y'].mean():,.0f} units

📈 TREND ANALYSIS:
● Growth vs Historical: {((avg_forecast - daily_sales['y'].mean()) / daily_sales['y'].mean()) * 100:+.1f}%

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

---
Report generated by Streamlit Forecasting Dashboard
Built with ❤️ using Python, Prophet, and Streamlit
"""
            st.download_button(
                label="📥 Download Business Report",
                data=report,
                file_name=f"business_report_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                key="business_report"
            )

            st.success("✅ Business report ready for download!")
        
            
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