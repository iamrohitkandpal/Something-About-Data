"""
Main forecasting pipeline that orchestrates data preprocessing, 
model training, and evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import os
import warnings
warnings.filterwarnings('ignore')

from .data_processing.preprocessor import DataPreprocessor
from .models.ml_models import RandomForestForecaster, XGBoostForecaster
from .models.time_series_models import ARIMAForecaster, ProphetForecaster, create_holiday_dataframe
from .models.deep_learning_models import LSTMForecaster, GRUForecaster
from .utils.metrics import evaluate_model, plot_predictions, compare_models, create_forecast_plot


class DemandForecastingPipeline:
    """
    Complete pipeline for retail demand forecasting with multiple models.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the forecasting pipeline.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config or self._default_config()
        self.preprocessor = DataPreprocessor()
        self.models = {}
        self.results = {}
        self.data = None
        self.feature_columns = None
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the pipeline."""
        return {
            'test_size': 0.2,
            'lags': [1, 7, 14, 30],
            'rolling_windows': [7, 14, 30],
            'models_to_use': ['random_forest', 'xgboost', 'arima', 'prophet', 'lstm'],
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            },
            'lstm': {
                'sequence_length': 30,
                'units': [50, 50],
                'epochs': 50,
                'batch_size': 32
            }
        }
    
    def load_data(self, data_path: str = None, data_df: pd.DataFrame = None,
                 data_type: str = 'kaggle') -> pd.DataFrame:
        """
        Load and preprocess data.
        
        Args:
            data_path: Path to data file
            data_df: DataFrame with data (alternative to file path)
            data_type: Type of data ('kaggle' or 'custom')
            
        Returns:
            Loaded DataFrame
        """
        if data_df is not None:
            self.data = data_df.copy()
        elif data_path is not None:
            if data_type == 'kaggle':
                self.data = self.preprocessor.load_kaggle_store_data(data_path)
            else:
                self.data = pd.read_csv(data_path)
                self.data['date'] = pd.to_datetime(self.data['date'])
        else:
            raise ValueError("Either data_path or data_df must be provided")
        
        print(f"Data loaded successfully. Shape: {self.data.shape}")
        print(f"Date range: {self.data['date'].min()} to {self.data['date'].max()}")
        
        return self.data
    
    def create_features(self, target_col: str = 'sales') -> pd.DataFrame:
        """
        Create features for forecasting.
        
        Args:
            target_col: Name of the target column
            
        Returns:
            DataFrame with features
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        print("Creating features...")
        
        # Create time features
        self.data = self.preprocessor.create_time_features(self.data)
        print("✓ Time features created")
        
        # Create holiday features
        self.data = self.preprocessor.create_holiday_features(self.data)
        print("✓ Holiday features created")
        
        # Create lag features
        self.data = self.preprocessor.create_lag_features(
            self.data, target_col, self.config['lags']
        )
        print("✓ Lag features created")
        
        # Create rolling features
        self.data = self.preprocessor.create_rolling_features(
            self.data, target_col, self.config['rolling_windows']
        )
        print("✓ Rolling features created")
        
        # Get feature columns
        self.feature_columns = self.preprocessor.get_feature_columns(
            self.data, exclude_cols=['date', target_col]
        )
        
        print(f"Total features created: {len(self.feature_columns)}")
        
        return self.data
    
    def prepare_data_for_modeling(self, target_col: str = 'sales') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for modeling by splitting into train/test sets.
        
        Args:
            target_col: Name of the target column
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if self.data is None:
            raise ValueError("Data must be loaded and features created first")
        
        # Remove rows with NaN values (from lag features)
        clean_data = self.data.dropna().reset_index(drop=True)
        
        # Split data
        train_df, test_df = self.preprocessor.prepare_for_modeling(
            clean_data, target_col, self.config['test_size']
        )
        
        print(f"Training data: {len(train_df)} samples")
        print(f"Test data: {len(test_df)} samples")
        
        return train_df, test_df
    
    def train_ml_models(self, train_df: pd.DataFrame, target_col: str = 'sales') -> None:
        """
        Train machine learning models.
        
        Args:
            train_df: Training data
            target_col: Name of the target column
        """
        X_train = train_df[self.feature_columns]
        y_train = train_df[target_col]
        
        models_to_train = [m for m in self.config['models_to_use'] 
                          if m in ['random_forest', 'xgboost']]
        
        for model_name in models_to_train:
            print(f"\nTraining {model_name.replace('_', ' ').title()}...")
            
            try:
                if model_name == 'random_forest':
                    model = RandomForestForecaster()
                    model.fit(X_train, y_train, **self.config['random_forest'])
                    
                elif model_name == 'xgboost':
                    model = XGBoostForecaster()
                    model.fit(X_train, y_train, **self.config['xgboost'])
                
                self.models[model_name] = model
                print(f"✓ {model_name.replace('_', ' ').title()} trained successfully")
                
            except Exception as e:
                print(f"✗ Error training {model_name}: {str(e)}")
    
    def train_time_series_models(self, train_df: pd.DataFrame, target_col: str = 'sales') -> None:
        """
        Train time series models.
        
        Args:
            train_df: Training data
            target_col: Name of the target column
        """
        models_to_train = [m for m in self.config['models_to_use'] 
                          if m in ['arima', 'prophet']]
        
        for model_name in models_to_train:
            print(f"\nTraining {model_name.upper()}...")
            
            try:
                if model_name == 'arima':
                    # For ARIMA, we use the time series data
                    series = train_df.set_index('date')[target_col]
                    model = ARIMAForecaster()
                    model.fit(series)
                    
                elif model_name == 'prophet':
                    # Create holidays dataframe
                    years = list(range(train_df['date'].dt.year.min(), 
                                     train_df['date'].dt.year.max() + 3))
                    holidays_df = create_holiday_dataframe(years)
                    
                    model = ProphetForecaster(holidays=holidays_df)
                    model.fit(train_df, 'date', target_col)
                
                self.models[model_name] = model
                print(f"✓ {model_name.upper()} trained successfully")
                
            except Exception as e:
                print(f"✗ Error training {model_name}: {str(e)}")
    
    def train_deep_learning_models(self, train_df: pd.DataFrame, target_col: str = 'sales') -> None:
        """
        Train deep learning models.
        
        Args:
            train_df: Training data
            target_col: Name of the target column
        """
        models_to_train = [m for m in self.config['models_to_use'] 
                          if m in ['lstm', 'gru']]
        
        if not models_to_train:
            return
        
        # Prepare data for deep learning
        feature_cols = [col for col in self.feature_columns if not col.startswith(target_col + '_lag')]
        
        for model_name in models_to_train:
            print(f"\nTraining {model_name.upper()}...")
            
            try:
                if model_name == 'lstm':
                    model = LSTMForecaster(self.config['lstm']['sequence_length'])
                elif model_name == 'gru':
                    model = GRUForecaster(self.config['lstm']['sequence_length'])  # Use same config
                else:
                    continue
                
                # Prepare data
                X_train, X_test, y_train, y_test = model.prepare_data(
                    train_df, target_col, feature_cols, test_size=0.2
                )
                
                # Train model
                model.fit(X_train, y_train, X_test, y_test, **self.config['lstm'])
                
                self.models[model_name] = model
                print(f"✓ {model_name.upper()} trained successfully")
                
            except Exception as e:
                print(f"✗ Error training {model_name}: {str(e)}")
    
    def evaluate_models(self, test_df: pd.DataFrame, target_col: str = 'sales') -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models.
        
        Args:
            test_df: Test data
            target_col: Name of the target column
            
        Returns:
            Dictionary with evaluation results for each model
        """
        if not self.models:
            raise ValueError("No models have been trained")
        
        print("\nEvaluating models...")
        print("=" * 50)
        
        y_true = test_df[target_col].values
        
        for model_name, model in self.models.items():
            try:
                if model_name in ['random_forest', 'xgboost']:
                    # ML models
                    X_test = test_df[self.feature_columns]
                    y_pred = model.predict(X_test)
                    
                elif model_name == 'arima':
                    # ARIMA model
                    steps = len(test_df)
                    y_pred = model.predict(steps)
                    
                elif model_name == 'prophet':
                    # Prophet model
                    periods = len(test_df)
                    forecast = model.predict(periods)
                    # Get predictions for test period
                    train_size = len(self.data) - len(test_df)
                    y_pred = forecast['yhat'].iloc[train_size:].values
                    
                elif model_name in ['lstm', 'gru']:
                    # Deep learning models
                    feature_cols = [col for col in self.feature_columns if not col.startswith(target_col + '_lag')]
                    X_train, X_test, _, _ = model.prepare_data(
                        test_df, target_col, feature_cols, test_size=1.0
                    )
                    if len(X_test) > 0:
                        y_pred = model.predict(X_test)
                    else:
                        continue
                
                # Evaluate predictions
                results = evaluate_model(y_true, y_pred, model_name.replace('_', ' ').title())
                self.results[model_name] = results
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {str(e)}")
        
        return self.results
    
    def generate_forecasts(self, horizon: int = 30, target_col: str = 'sales') -> Dict[str, np.ndarray]:
        """
        Generate future forecasts with all models.
        
        Args:
            horizon: Number of periods to forecast
            target_col: Name of the target column
            
        Returns:
            Dictionary with forecasts for each model
        """
        forecasts = {}
        
        for model_name, model in self.models.items():
            try:
                if model_name in ['random_forest', 'xgboost']:
                    # For ML models, we need to create future features
                    # This is simplified - in practice, you'd need more sophisticated approach
                    last_row = self.data.iloc[-1:].copy()
                    future_predictions = []
                    
                    for step in range(horizon):
                        X_future = last_row[self.feature_columns]
                        pred = model.predict(X_future)[0]
                        future_predictions.append(pred)
                        
                        # Update last_row for next prediction (simplified)
                        # In practice, you'd update lag and rolling features properly
                        last_row[target_col] = pred
                    
                    forecasts[model_name] = np.array(future_predictions)
                    
                elif model_name == 'arima':
                    forecasts[model_name] = model.predict(horizon)
                    
                elif model_name == 'prophet':
                    forecast = model.predict(horizon)
                    forecasts[model_name] = forecast['yhat'].tail(horizon).values
                    
                elif model_name in ['lstm', 'gru']:
                    # Use the last sequence from training data
                    feature_cols = [col for col in self.feature_columns if not col.startswith(target_col + '_lag')]
                    _, _, _, _ = model.prepare_data(self.data, target_col, feature_cols, test_size=0.1)
                    
                    # Get last sequence (this is simplified)
                    last_sequence = self.data[feature_cols].tail(model.sequence_length).values
                    if len(last_sequence) == model.sequence_length:
                        last_sequence = model.feature_scaler.transform(last_sequence)
                        forecasts[model_name] = model.predict_future(last_sequence, horizon)
                
            except Exception as e:
                print(f"Error generating forecast for {model_name}: {str(e)}")
        
        return forecasts
    
    def run_full_pipeline(self, data_path: str = None, data_df: pd.DataFrame = None,
                         target_col: str = 'sales', data_type: str = 'kaggle') -> Dict[str, Any]:
        """
        Run the complete forecasting pipeline.
        
        Args:
            data_path: Path to data file
            data_df: DataFrame with data
            target_col: Name of the target column
            data_type: Type of data
            
        Returns:
            Dictionary with results including model performance and forecasts
        """
        print("Starting Demand Forecasting Pipeline")
        print("=" * 50)
        
        # Load data
        self.load_data(data_path, data_df, data_type)
        
        # Create features
        self.create_features(target_col)
        
        # Prepare data for modeling
        train_df, test_df = self.prepare_data_for_modeling(target_col)
        
        # Train models
        self.train_ml_models(train_df, target_col)
        self.train_time_series_models(train_df, target_col)
        self.train_deep_learning_models(train_df, target_col)
        
        # Evaluate models
        results = self.evaluate_models(test_df, target_col)
        
        # Generate future forecasts
        forecasts = self.generate_forecasts(30, target_col)
        
        # Summary
        print(f"\nPipeline completed successfully!")
        print(f"Models trained: {list(self.models.keys())}")
        print(f"Best model by RMSE: {min(results.items(), key=lambda x: x[1]['RMSE'])[0] if results else 'None'}")
        
        return {
            'models': self.models,
            'results': results,
            'forecasts': forecasts,
            'train_data': train_df,
            'test_data': test_df
        }
    
    def compare_all_models(self) -> None:
        """Compare performance of all models."""
        if self.results:
            compare_models(self.results)
        else:
            print("No results available. Run evaluation first.")
    
    def save_results(self, output_dir: str = 'results') -> None:
        """
        Save results and models to disk.
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        if self.results:
            results_df = pd.DataFrame(self.results).T
            results_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'))
            print(f"Model comparison saved to {output_dir}/model_comparison.csv")
        
        # Save models (only ML models for now)
        for model_name, model in self.models.items():
            if model_name in ['random_forest', 'xgboost']:
                model_path = os.path.join(output_dir, f'{model_name}_model.pkl')
                model.save_model(model_path)
                print(f"{model_name} model saved to {model_path}")


# Convenience function for quick forecasting
def quick_forecast(data_path: str = None, data_df: pd.DataFrame = None,
                  target_col: str = 'sales', models: List[str] = None) -> Dict[str, Any]:
    """
    Quick forecasting function with minimal configuration.
    
    Args:
        data_path: Path to data file
        data_df: DataFrame with data
        target_col: Name of the target column
        models: List of models to use
        
    Returns:
        Dictionary with results
    """
    if models is None:
        models = ['random_forest', 'xgboost', 'prophet']
    
    config = {
        'models_to_use': models,
        'test_size': 0.2
    }
    
    pipeline = DemandForecastingPipeline(config)
    return pipeline.run_full_pipeline(data_path, data_df, target_col)