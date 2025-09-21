# 📊 Retail Demand Forecasting Web Application

A comprehensive Python-based web application for retail demand forecasting using Streamlit. This application provides an end-to-end solution for analyzing retail sales data, engineering features, training multiple forecasting models, and generating future predictions.

## 🎯 Features

### Data Management
- **CSV Upload**: Support for uploading retail sales datasets
- **Sample Data**: Built-in sample data generator for demonstration
- **Data Validation**: Automatic validation and preprocessing of uploaded data
- **Data Exploration**: Interactive visualizations and summary statistics

### Feature Engineering
- **Time-based Features**: Year, month, day, weekday, quarter, weekend flags
- **Seasonal Features**: Cyclical encoding of temporal patterns
- **Lag Features**: Historical sales values (1, 7, 14, 30 days)
- **Rolling Features**: Moving averages and standard deviations
- **Promotion & Holiday Features**: External factor handling

### Forecasting Models
- **Statistical Models**: ARIMA/SARIMA using statsmodels
- **Machine Learning**: Random Forest and XGBoost
- **Prophet**: Facebook's time-series forecasting library
- **Deep Learning**: LSTM/GRU neural networks with TensorFlow/Keras

### Model Evaluation
- **Metrics**: RMSE, MAE, MAPE
- **Cross-validation**: Time-series split validation
- **Visualization**: Actual vs predicted plots
- **Model Comparison**: Side-by-side performance analysis

### Forecasting & Visualization
- **Future Predictions**: Generate forecasts for specified periods
- **Confidence Intervals**: Statistical uncertainty estimation
- **Interactive Charts**: Plotly-based visualizations
- **Export**: Download forecasts as CSV files

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn, XGBoost
- **Statistical Models**: statsmodels
- **Time Series**: Prophet (fbprophet)
- **Deep Learning**: TensorFlow/Keras
- **Deployment**: Docker, Heroku, Streamlit Sharing

## 📋 Data Requirements

Your CSV file should contain the following columns:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `date` | datetime | Yes | Date of the sales record |
| `sales` | numeric | Yes | Sales/demand value (target variable) |
| `store_id` | numeric | Optional | Store identifier |
| `item_id` | numeric | Optional | Item/product identifier |
| `promo` | binary | Optional | Promotion flag (0/1) |
| `holiday` | binary | Optional | Holiday flag (0/1) |

### Sample Data Format
```csv
date,store_id,item_id,sales,promo,holiday
2020-01-01,1,101,150,0,0
2020-01-02,1,101,175,1,0
2020-01-03,1,102,120,0,0
```

## 🚀 Quick Start

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/iamrohitkandpal/Something-About-Data.git
   cd Something-About-Data
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run main.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t retail-forecasting .
   ```

2. **Run the container**
   ```bash
   docker run -p 8501:8501 retail-forecasting
   ```

### Streamlit Sharing Deployment

1. Fork this repository
2. Connect your GitHub account to [Streamlit Sharing](https://share.streamlit.io/)
3. Deploy the `main.py` file

### Heroku Deployment

1. **Create Heroku app**
   ```bash
   heroku create your-app-name
   ```

2. **Deploy**
   ```bash
   git push heroku main
   ```

## 📱 Application Usage

### 1. Home Page
- Overview of features and capabilities
- Library availability status
- Sample data format reference

### 2. Data Upload & Exploration
- Upload CSV files or use sample data
- View data summary and statistics
- Explore data with interactive visualizations
- Check data quality and missing values

### 3. Feature Engineering
- Select features to create:
  - Time-based features
  - Lag features
  - Rolling window features
  - Seasonal features
  - Promotion/holiday features
- Preview engineered dataset

### 4. Model Training
- Select models to train:
  - Random Forest
  - XGBoost
  - ARIMA
  - Prophet
  - LSTM
- Configure hyperparameters
- View training results and metrics

### 5. Forecasting
- Generate future predictions
- Set forecast horizon
- Configure confidence intervals
- Visualize results
- Download forecasts

### 6. Model Comparison
- Compare model performance
- View metrics side-by-side
- Identify best performing models

## 🔧 Configuration

### Model Parameters
- **Random Forest**: Number of trees, max depth
- **XGBoost**: Number of trees, learning rate
- **ARIMA**: Automatic parameter selection
- **Prophet**: Built-in seasonality detection
- **LSTM**: Configurable architecture

### Forecasting Settings
- Forecast horizon: 7-90 days
- Confidence intervals: 80-99%
- Historical data inclusion

## 📊 Model Performance

The application evaluates models using:
- **RMSE (Root Mean Square Error)**: Penalizes large errors
- **MAE (Mean Absolute Error)**: Average prediction error
- **MAPE (Mean Absolute Percentage Error)**: Percentage-based error

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [Prophet](https://facebook.github.io/prophet/) for time-series forecasting
- [Plotly](https://plotly.com/) for interactive visualizations
- Kaggle community for retail forecasting datasets

## 📧 Contact

Rohit Kandpal - [@iamrohitkandpal](https://github.com/iamrohitkandpal)

Project Link: [https://github.com/iamrohitkandpal/Something-About-Data](https://github.com/iamrohitkandpal/Something-About-Data)