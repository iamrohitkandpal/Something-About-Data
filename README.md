# Demand Forecasting for Retail

A comprehensive Python solution for retail demand forecasting using multiple machine learning and time series models. This project implements various forecasting techniques including Random Forest, XGBoost, ARIMA, Facebook Prophet, and LSTM/GRU neural networks.

## 🎯 Features

- **Multiple Forecasting Models**: Random Forest, XGBoost, ARIMA, Prophet, LSTM, GRU
- **Automated Feature Engineering**: Time-based features, holidays, lag features, rolling statistics
- **Model Evaluation**: RMSE, MAPE, MAE, SMAPE metrics
- **Easy-to-Use Pipeline**: Complete end-to-end forecasting pipeline
- **Dataset Support**: Compatible with Kaggle datasets (Store Item Demand, M5 Forecasting)
- **Visualization**: Comprehensive plots for model comparison and forecast results

## 📊 Supported Datasets

### Free Dataset Sources
- **Kaggle – Store Item Demand Forecasting Challenge**
- **M5 Forecasting Dataset (Walmart sales)**
- **Custom retail datasets** with date, store, item, and sales columns

### Dataset Requirements
- Daily sales data for products across stores
- Historical data with promotions and holidays (optional but recommended)
- Columns: `date`, `store`, `item`, `sales`

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/iamrohitkandpal/Something-About-Data.git
cd Something-About-Data
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from demand_forecasting.main_pipeline import quick_forecast
import pandas as pd

# Load your data
data = pd.read_csv('your_sales_data.csv')

# Quick forecasting with default models
results = quick_forecast(
    data_df=data,
    target_col='sales',
    models=['random_forest', 'xgboost', 'prophet']
)

print("Model Performance:", results['results'])
```

### Advanced Usage

```python
from demand_forecasting.main_pipeline import DemandForecastingPipeline

# Custom configuration
config = {
    'test_size': 0.2,
    'lags': [1, 7, 14, 30],
    'rolling_windows': [7, 14, 30],
    'models_to_use': ['random_forest', 'xgboost', 'arima', 'prophet', 'lstm'],
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 15
    }
}

# Create and run pipeline
pipeline = DemandForecastingPipeline(config)
results = pipeline.run_full_pipeline(
    data_path='path/to/data.csv',
    target_col='sales'
)

# Compare models
pipeline.compare_all_models()
```

## 📁 Project Structure

```
demand_forecasting/
├── __init__.py
├── main_pipeline.py          # Main forecasting pipeline
├── data_processing/
│   ├── __init__.py
│   └── preprocessor.py       # Data preprocessing and feature engineering
├── models/
│   ├── __init__.py
│   ├── ml_models.py          # Random Forest and XGBoost
│   ├── time_series_models.py # ARIMA and Prophet
│   └── deep_learning_models.py # LSTM and GRU
└── utils/
    ├── __init__.py
    └── metrics.py            # Evaluation metrics and visualization

example_usage.py              # Example scripts
requirements.txt              # Dependencies
```

## 🤖 Available Models

### Machine Learning Models
- **Random Forest**: Ensemble method with hyperparameter optimization
- **XGBoost**: Gradient boosting with feature importance

### Time Series Models
- **ARIMA**: AutoRegressive Integrated Moving Average
- **Prophet**: Facebook's time series forecasting tool with seasonality

### Deep Learning Models
- **LSTM**: Long Short-Term Memory networks
- **GRU**: Gated Recurrent Units

## 📊 Evaluation Metrics

- **RMSE**: Root Mean Square Error
- **MAPE**: Mean Absolute Percentage Error
- **MAE**: Mean Absolute Error
- **SMAPE**: Symmetric Mean Absolute Percentage Error

## 🔧 Feature Engineering

### Automated Features
- **Time Features**: Year, month, day, weekday, quarter
- **Cyclical Encoding**: Sine/cosine transformations for seasonality
- **Holiday Features**: Major holidays and proximity indicators
- **Lag Features**: Previous sales values (1, 7, 14, 30 days)
- **Rolling Statistics**: Moving averages, std, min, max

### Custom Features
- Weekend indicators
- Month start/end indicators
- Holiday proximity (within 7 days)
- Seasonal decomposition

## 📈 Example Results

```python
# Model Performance Comparison
Model Performance:
├── Random Forest: RMSE=15.23, MAPE=12.45%
├── XGBoost: RMSE=14.87, MAPE=11.98%
├── Prophet: RMSE=16.45, MAPE=13.12%
├── ARIMA: RMSE=18.23, MAPE=14.67%
└── LSTM: RMSE=15.67, MAPE=12.78%

Best Model: XGBoost
```

## 💡 Usage Examples

### Example 1: Kaggle Store Item Demand Data
```python
from demand_forecasting.main_pipeline import quick_forecast

# For Kaggle Store Item Demand Forecasting Challenge
results = quick_forecast(
    data_path='train.csv',
    target_col='sales',
    models=['random_forest', 'xgboost', 'prophet']
)
```

### Example 2: M5 Forecasting (Walmart) Data
```python
pipeline = DemandForecastingPipeline({
    'models_to_use': ['xgboost', 'prophet', 'lstm'],
    'test_size': 0.15
})

results = pipeline.run_full_pipeline(
    data_path='m5_sales_data.csv',
    target_col='sales'
)
```

### Example 3: Custom Retail Data
```python
import pandas as pd

# Your custom data format
data = pd.DataFrame({
    'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'store': [1, 1, 1],
    'item': [1, 1, 1],
    'sales': [100, 120, 95]
})

results = quick_forecast(data_df=data, target_col='sales')
```

## 🔬 Running Examples

Run the example script to see the pipeline in action:

```bash
python example_usage.py
```

This will demonstrate:
- Basic usage with sample data
- Advanced configuration options
- Single time series analysis
- Data exploration and visualization
- Kaggle data format examples

## 📋 Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning models and metrics
- **xgboost**: Gradient boosting framework
- **statsmodels**: Statistical models (ARIMA)
- **prophet**: Time series forecasting
- **tensorflow**: Deep learning models (LSTM/GRU)
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization

## 🎛️ Configuration Options

```python
config = {
    'test_size': 0.2,                    # Train/test split ratio
    'lags': [1, 7, 14, 30],             # Lag features to create
    'rolling_windows': [7, 14, 30],      # Rolling window sizes
    'models_to_use': [                   # Models to train
        'random_forest', 'xgboost', 
        'arima', 'prophet', 'lstm'
    ],
    'random_forest': {                   # Random Forest parameters
        'n_estimators': 100,
        'max_depth': 10
    },
    'xgboost': {                        # XGBoost parameters
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1
    },
    'lstm': {                           # LSTM parameters
        'sequence_length': 30,
        'units': [50, 50],
        'epochs': 50,
        'batch_size': 32
    }
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🎯 Use Cases

This forecasting solution is perfect for:
- **Retail demand forecasting**
- **Inventory management**
- **Sales planning**
- **Supply chain optimization**
- **Revenue forecasting**
- **Data science competitions** (Kaggle, etc.)

## 📞 Support

If you encounter any issues or have questions:
1. Check the [example_usage.py](example_usage.py) file for detailed examples
2. Review the model documentation in each module
3. Open an issue on GitHub for bug reports or feature requests

---

**Happy Forecasting!** 📈✨