"""
Example usage of the Demand Forecasting Pipeline

This script demonstrates how to use the demand forecasting system
with sample data and various models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import the forecasting pipeline
from demand_forecasting.main_pipeline import DemandForecastingPipeline, quick_forecast
from demand_forecasting.utils.metrics import plot_predictions, compare_models


def create_sample_data(n_stores: int = 3, n_items: int = 5, n_days: int = 365*2) -> pd.DataFrame:
    """
    Create sample retail sales data for demonstration.
    
    Args:
        n_stores: Number of stores
        n_items: Number of items per store
        n_days: Number of days of data
        
    Returns:
        Sample DataFrame with sales data
    """
    print(f"Creating sample data: {n_stores} stores, {n_items} items, {n_days} days")
    
    # Create date range
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    data = []
    
    for store in range(1, n_stores + 1):
        for item in range(1, n_items + 1):
            # Base sales level for this store-item combination
            base_sales = np.random.uniform(50, 200)
            
            for i, date in enumerate(dates):
                # Seasonal pattern (yearly)
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * i / 365)
                
                # Weekly pattern (lower sales on weekdays)
                weekly_factor = 1.2 if date.weekday() >= 5 else 0.9
                
                # Holiday effects
                holiday_factor = 1.0
                if date.month == 12 and date.day in [24, 25]:  # Christmas
                    holiday_factor = 2.0
                elif date.month == 11 and date.day == 24:  # Black Friday (approx)
                    holiday_factor = 2.5
                elif date.month == 1 and date.day == 1:  # New Year
                    holiday_factor = 0.5
                
                # Random noise
                noise = np.random.normal(0, 0.1)
                
                # Calculate final sales
                sales = base_sales * seasonal_factor * weekly_factor * holiday_factor * (1 + noise)
                sales = max(0, sales)  # Ensure non-negative
                
                data.append({
                    'date': date,
                    'store': store,
                    'item': item,
                    'sales': round(sales, 2)
                })
    
    df = pd.DataFrame(data)
    print(f"Sample data created with {len(df)} records")
    return df


def example_basic_usage():
    """Example of basic usage with sample data."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Usage with Sample Data")
    print("="*60)
    
    # Create sample data
    sample_data = create_sample_data(n_stores=2, n_items=3, n_days=730)
    
    # Use quick forecast function
    results = quick_forecast(
        data_df=sample_data,
        target_col='sales',
        models=['random_forest', 'xgboost']
    )
    
    print("\nQuick forecast completed!")
    print(f"Models trained: {list(results['models'].keys())}")
    print(f"Model performance: {results['results']}")


def example_advanced_usage():
    """Example of advanced usage with custom configuration."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Advanced Usage with Custom Configuration")
    print("="*60)
    
    # Create larger sample dataset
    sample_data = create_sample_data(n_stores=2, n_items=2, n_days=1095)  # 3 years
    
    # Custom configuration
    config = {
        'test_size': 0.15,  # Use 15% for testing
        'lags': [1, 7, 14, 30, 60],  # More lag features
        'rolling_windows': [7, 14, 30, 60, 90],  # More rolling features
        'models_to_use': ['random_forest', 'xgboost', 'prophet'],
        'random_forest': {
            'n_estimators': 200,
            'max_depth': 15,
            'random_state': 42
        },
        'xgboost': {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.05,
            'random_state': 42
        }
    }
    
    # Create pipeline with custom configuration
    pipeline = DemandForecastingPipeline(config)
    
    # Run the full pipeline
    results = pipeline.run_full_pipeline(
        data_df=sample_data,
        target_col='sales'
    )
    
    # Compare models
    pipeline.compare_all_models()
    
    # Generate and plot forecasts
    forecasts = results['forecasts']
    if forecasts:
        print(f"\nGenerated forecasts for next 30 days:")
        for model_name, forecast in forecasts.items():
            print(f"{model_name}: {forecast[:5]}... (showing first 5 values)")


def example_single_store_item():
    """Example focusing on a single store-item combination."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Single Store-Item Analysis")
    print("="*60)
    
    # Create sample data
    sample_data = create_sample_data(n_stores=1, n_items=1, n_days=1095)
    
    # Filter to single store-item for simpler analysis
    single_series = sample_data[(sample_data['store'] == 1) & (sample_data['item'] == 1)].copy()
    single_series = single_series.sort_values('date').reset_index(drop=True)
    
    print(f"Analyzing single time series with {len(single_series)} data points")
    
    # Configuration for time series models
    config = {
        'test_size': 0.2,
        'models_to_use': ['arima', 'prophet', 'random_forest'],
        'lags': [1, 7, 14, 30],
        'rolling_windows': [7, 14, 30]
    }
    
    pipeline = DemandForecastingPipeline(config)
    results = pipeline.run_full_pipeline(data_df=single_series, target_col='sales')
    
    # Plot predictions for the best model
    if results['results']:
        best_model_name = min(results['results'].items(), key=lambda x: x[1]['RMSE'])[0]
        print(f"\nBest model: {best_model_name}")
        
        # Get test predictions
        test_df = results['test_data']
        if best_model_name in results['models']:
            model = results['models'][best_model_name]
            
            if best_model_name in ['random_forest', 'xgboost']:
                feature_cols = pipeline.feature_columns
                X_test = test_df[feature_cols]
                y_pred = model.predict(X_test)
                y_true = test_df['sales'].values
                
                # Plot results
                plot_predictions(
                    y_true, y_pred, 
                    test_df['date'], 
                    f"{best_model_name.replace('_', ' ').title()}"
                )


def example_data_exploration():
    """Example of data exploration and visualization."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Data Exploration")
    print("="*60)
    
    # Create sample data
    sample_data = create_sample_data(n_stores=3, n_items=2, n_days=730)
    
    print("Data shape:", sample_data.shape)
    print("\nFirst few rows:")
    print(sample_data.head())
    
    print("\nBasic statistics:")
    print(sample_data.describe())
    
    # Plot sales over time for one store-item
    plt.figure(figsize=(12, 6))
    store1_item1 = sample_data[(sample_data['store'] == 1) & (sample_data['item'] == 1)]
    plt.plot(store1_item1['date'], store1_item1['sales'])
    plt.title('Sales Over Time - Store 1, Item 1')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Sales by store
    plt.figure(figsize=(10, 6))
    store_sales = sample_data.groupby('store')['sales'].sum()
    plt.bar(store_sales.index, store_sales.values)
    plt.title('Total Sales by Store')
    plt.xlabel('Store')
    plt.ylabel('Total Sales')
    plt.grid(True, alpha=0.3)
    plt.show()


def example_kaggle_data_format():
    """Example showing expected Kaggle data format."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Kaggle Data Format")
    print("="*60)
    
    print("For Kaggle Store Item Demand Forecasting Challenge, your data should have columns:")
    print("- date: Date of the sale (YYYY-MM-DD format)")
    print("- store: Store identifier (integer)")
    print("- item: Item identifier (integer)")
    print("- sales: Number of items sold (target variable)")
    
    print("\nExample data format:")
    sample_kaggle = pd.DataFrame({
        'date': ['2013-01-01', '2013-01-02', '2013-01-03'],
        'store': [1, 1, 1],
        'item': [1, 1, 1],
        'sales': [13, 11, 14]
    })
    print(sample_kaggle)
    
    print("\nTo use with actual Kaggle data:")
    print("```python")
    print("from demand_forecasting.main_pipeline import quick_forecast")
    print("")
    print("# Load your Kaggle data")
    print("results = quick_forecast(")
    print("    data_path='path/to/your/kaggle_data.csv',")
    print("    target_col='sales',")
    print("    models=['random_forest', 'xgboost', 'prophet']")
    print(")")
    print("```")


if __name__ == "__main__":
    """Run all examples."""
    print("Demand Forecasting Pipeline - Examples")
    print("="*60)
    
    try:
        # Run examples
        example_basic_usage()
        example_advanced_usage()
        example_single_store_item()
        example_data_exploration()
        example_kaggle_data_format()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")