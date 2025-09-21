"""
Utility functions for model evaluation and metrics calculation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAPE value as percentage
    """
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAE value
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        SMAPE value as percentage
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Avoid division by zero
    mask = denominator != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, 
                  model_name: str = "Model") -> Dict[str, float]:
    """
    Evaluate model performance using multiple metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model for display
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        'RMSE': calculate_rmse(y_true, y_pred),
        'MAE': calculate_mae(y_true, y_pred),
        'MAPE': calculate_mape(y_true, y_pred),
        'SMAPE': calculate_smape(y_true, y_pred)
    }
    
    print(f"\n{model_name} Performance:")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                    dates: pd.Series = None, model_name: str = "Model",
                    figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Plot actual vs predicted values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        dates: Optional date series for x-axis
        model_name: Name of the model for title
        figsize: Figure size
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping plot.")
        return
    
    plt.figure(figsize=figsize)
    
    if dates is not None:
        plt.plot(dates, y_true, label='Actual', alpha=0.7)
        plt.plot(dates, y_pred, label='Predicted', alpha=0.7)
        plt.xlabel('Date')
    else:
        plt.plot(y_true, label='Actual', alpha=0.7)
        plt.plot(y_pred, label='Predicted', alpha=0.7)
        plt.xlabel('Time')
    
    plt.ylabel('Sales')
    plt.title(f'{model_name} - Actual vs Predicted')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                  model_name: str = "Model",
                  figsize: Tuple[int, int] = (12, 4)) -> None:
    """
    Plot residuals analysis.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model for title
        figsize: Figure size
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping plot.")
        return
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.6)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title(f'{model_name} - Residuals vs Predicted')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{model_name} - Residuals Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def compare_models(results: Dict[str, Dict[str, float]], 
                  figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Compare multiple models' performance.
    
    Args:
        results: Dictionary with model names as keys and metrics as values
        figsize: Figure size
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Showing text comparison only.")
    
    # Convert to DataFrame for easier plotting
    df_results = pd.DataFrame(results).T
    
    # Plot comparison if matplotlib is available
    if MATPLOTLIB_AVAILABLE:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        
        metrics = ['RMSE', 'MAE', 'MAPE', 'SMAPE']
        
        for i, metric in enumerate(metrics):
            if metric in df_results.columns:
                df_results[metric].plot(kind='bar', ax=axes[i])
                axes[i].set_title(f'{metric} Comparison')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Print summary table
    print("\nModel Comparison Summary:")
    print("=" * 50)
    print(df_results.round(4))


def plot_feature_importance(feature_importance: Dict[str, float],
                          top_n: int = 20,
                          figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot feature importance.
    
    Args:
        feature_importance: Dictionary with feature names as keys and importance as values
        top_n: Number of top features to display
        figsize: Figure size
    """
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Get top N features
    top_features = sorted_features[:top_n]
    features, importance = zip(*top_features)
    
    # Create plot
    plt.figure(figsize=figsize)
    y_pos = np.arange(len(features))
    
    plt.barh(y_pos, importance)
    plt.yticks(y_pos, features)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_forecast_plot(train_data: pd.DataFrame, 
                        test_data: pd.DataFrame,
                        predictions: np.ndarray,
                        date_col: str = 'date',
                        target_col: str = 'sales',
                        model_name: str = "Model",
                        figsize: Tuple[int, int] = (14, 6)) -> None:
    """
    Create a comprehensive forecast plot showing train, test, and predictions.
    
    Args:
        train_data: Training data DataFrame
        test_data: Test data DataFrame
        predictions: Predictions array
        date_col: Name of the date column
        target_col: Name of the target column
        model_name: Name of the model for title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Plot training data
    plt.plot(train_data[date_col], train_data[target_col], 
             label='Training Data', alpha=0.7, color='blue')
    
    # Plot actual test data
    plt.plot(test_data[date_col], test_data[target_col], 
             label='Actual Test Data', alpha=0.8, color='green')
    
    # Plot predictions
    plt.plot(test_data[date_col], predictions, 
             label='Predictions', alpha=0.8, color='red', linestyle='--')
    
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title(f'{model_name} - Forecast Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()