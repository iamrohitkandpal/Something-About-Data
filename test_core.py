"""
Simple test script to demonstrate core functionality
without requiring all optional dependencies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import core modules
from demand_forecasting.data_processing.preprocessor import DataPreprocessor
from demand_forecasting.utils.metrics import calculate_rmse, calculate_mape, evaluate_model

def create_test_data(n_days=100):
    """Create simple test data for demonstration."""
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    data = []
    for i, date in enumerate(dates):
        # Simple pattern with some seasonality
        base_sales = 50 + 20 * np.sin(2 * np.pi * i / 7)  # Weekly pattern
        noise = np.random.normal(0, 5)
        sales = max(0, base_sales + noise)
        
        data.append({
            'date': date,
            'store': 1,
            'item': 1,
            'sales': round(sales, 2)
        })
    
    return pd.DataFrame(data)

def test_preprocessing():
    """Test data preprocessing functionality."""
    print("Testing Data Preprocessing")
    print("-" * 30)
    
    # Create test data
    df = create_test_data(100)
    print(f"Created test data with {len(df)} records")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Test time features
    df_with_time = preprocessor.create_time_features(df)
    print(f"Added time features: {len(df_with_time.columns)} total columns")
    
    # Test holiday features
    df_with_holidays = preprocessor.create_holiday_features(df_with_time)
    print(f"Added holiday features: {len(df_with_holidays.columns)} total columns")
    
    # Test lag features
    df_with_lags = preprocessor.create_lag_features(df_with_holidays, 'sales', [1, 7])
    print(f"Added lag features: {len(df_with_lags.columns)} total columns")
    
    # Test rolling features
    df_with_rolling = preprocessor.create_rolling_features(df_with_lags, 'sales', [7, 14])
    print(f"Added rolling features: {len(df_with_rolling.columns)} total columns")
    
    print("✓ All preprocessing tests passed!\n")
    return df_with_rolling

def test_metrics():
    """Test metrics calculation."""
    print("Testing Metrics Calculation")
    print("-" * 30)
    
    # Create sample predictions
    np.random.seed(42)
    y_true = np.random.randint(10, 100, 20)
    y_pred = y_true + np.random.normal(0, 5, 20)  # Add some noise
    
    # Test individual metrics
    rmse = calculate_rmse(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Test evaluation function
    results = evaluate_model(y_true, y_pred, "Test Model")
    
    print("✓ All metrics tests passed!\n")
    return results

def test_basic_ml_model():
    """Test basic machine learning functionality."""
    print("Testing Basic ML Model")
    print("-" * 30)
    
    try:
        from demand_forecasting.models.ml_models import RandomForestForecaster
        
        # Create test data
        df = create_test_data(200)
        preprocessor = DataPreprocessor()
        
        # Add features
        df = preprocessor.create_time_features(df)
        df = preprocessor.create_lag_features(df, 'sales', [1, 7])
        df = preprocessor.create_rolling_features(df, 'sales', [7])
        
        # Clean data
        df_clean = df.dropna().reset_index(drop=True)
        
        # Get features
        feature_cols = preprocessor.get_feature_columns(df_clean, ['date', 'sales'])
        
        # Split data
        split_idx = int(len(df_clean) * 0.8)
        train_df = df_clean.iloc[:split_idx]
        test_df = df_clean.iloc[split_idx:]
        
        # Prepare training data
        X_train = train_df[feature_cols]
        y_train = train_df['sales']
        X_test = test_df[feature_cols]
        y_test = test_df['sales']
        
        # Train model
        model = RandomForestForecaster()
        model.fit(X_train, y_train, n_estimators=50, max_depth=5)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        results = evaluate_model(y_test.values, y_pred, "Random Forest")
        
        print("✓ Random Forest model test passed!")
        
    except ImportError as e:
        print(f"✗ Skipping ML model test due to missing dependency: {e}")
    except Exception as e:
        print(f"✗ ML model test failed: {e}")
    
    print()

def main():
    """Run all tests."""
    print("Demand Forecasting - Core Functionality Test")
    print("=" * 50)
    
    # Test preprocessing
    processed_data = test_preprocessing()
    
    # Test metrics
    test_metrics()
    
    # Test basic ML model
    test_basic_ml_model()
    
    print("Core functionality testing completed!")
    print("\nTo see more advanced examples with full dependencies, run:")
    print("pip install -r requirements.txt")
    print("python example_usage.py")

if __name__ == "__main__":
    main()