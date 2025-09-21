"""
Data preprocessing and feature engineering for demand forecasting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional


class DataPreprocessor:
    """
    A class for preprocessing retail sales data and creating features
    for demand forecasting models.
    """
    
    def __init__(self):
        self.holiday_dates = self._get_common_holidays()
    
    def _get_common_holidays(self) -> List[str]:
        """
        Returns a list of common US holidays that might affect retail sales.
        """
        # Common holidays that affect retail sales
        holidays = [
            '01-01',  # New Year's Day
            '02-14',  # Valentine's Day
            '07-04',  # Independence Day
            '10-31',  # Halloween
            '11-24',  # Thanksgiving (approximate)
            '12-25',  # Christmas
            '12-31',  # New Year's Eve
        ]
        return holidays
    
    def load_kaggle_store_data(self, train_path: str, test_path: str = None) -> pd.DataFrame:
        """
        Load Kaggle Store Item Demand Forecasting Challenge data.
        
        Args:
            train_path: Path to training data CSV
            test_path: Optional path to test data CSV
            
        Returns:
            Preprocessed DataFrame
        """
        train_df = pd.read_csv(train_path)
        
        # Convert date column
        train_df['date'] = pd.to_datetime(train_df['date'])
        
        # Sort by date and store/item
        train_df = train_df.sort_values(['store', 'item', 'date']).reset_index(drop=True)
        
        return train_df
    
    def create_time_features(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """
        Create time-based features from date column.
        
        Args:
            df: Input DataFrame
            date_col: Name of the date column
            
        Returns:
            DataFrame with additional time features
        """
        df = df.copy()
        
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Basic time features
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['dayofweek'] = df[date_col].dt.dayofweek
        df['dayofyear'] = df[date_col].dt.dayofyear
        df['week'] = df[date_col].dt.isocalendar().week
        df['quarter'] = df[date_col].dt.quarter
        
        # Weekend indicator
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        # Month start/end indicators
        df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
        
        # Cyclical encoding for seasonal patterns
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        return df
    
    def create_holiday_features(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """
        Create holiday-related features.
        
        Args:
            df: Input DataFrame
            date_col: Name of the date column
            
        Returns:
            DataFrame with holiday features
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Create holiday indicator
        df['is_holiday'] = 0
        
        for year in df[date_col].dt.year.unique():
            for holiday in self.holiday_dates:
                holiday_date = pd.to_datetime(f"{year}-{holiday}")
                df.loc[df[date_col] == holiday_date, 'is_holiday'] = 1
        
        # Days before/after holidays
        df['days_to_holiday'] = 0
        df['days_from_holiday'] = 0
        
        for idx, row in df.iterrows():
            current_date = row[date_col]
            year = current_date.year
            
            # Find nearest holiday
            holiday_distances = []
            for holiday in self.holiday_dates:
                holiday_date = pd.to_datetime(f"{year}-{holiday}")
                holiday_distances.append(abs((current_date - holiday_date).days))
                
                # Also check previous and next year
                prev_holiday = pd.to_datetime(f"{year-1}-{holiday}")
                next_holiday = pd.to_datetime(f"{year+1}-{holiday}")
                holiday_distances.append(abs((current_date - prev_holiday).days))
                holiday_distances.append(abs((current_date - next_holiday).days))
            
            min_distance = min(holiday_distances)
            
            # Find if we're before or after the nearest holiday
            for holiday in self.holiday_dates:
                for year_offset in [-1, 0, 1]:
                    holiday_date = pd.to_datetime(f"{year + year_offset}-{holiday}")
                    days_diff = (holiday_date - current_date).days
                    
                    if abs(days_diff) == min_distance:
                        if days_diff > 0:
                            df.loc[idx, 'days_to_holiday'] = days_diff
                        else:
                            df.loc[idx, 'days_from_holiday'] = abs(days_diff)
                        break
        
        # Holiday proximity features (within 7 days)
        df['near_holiday'] = ((df['days_to_holiday'] <= 7) & (df['days_to_holiday'] > 0) |
                             (df['days_from_holiday'] <= 7) & (df['days_from_holiday'] > 0)).astype(int)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'sales', 
                           lags: List[int] = [1, 7, 14, 30]) -> pd.DataFrame:
        """
        Create lag features for time series data.
        
        Args:
            df: Input DataFrame (should be sorted by date)
            target_col: Name of the target column
            lags: List of lag periods to create
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        # Group by store and item if they exist
        group_cols = []
        if 'store' in df.columns:
            group_cols.append('store')
        if 'item' in df.columns:
            group_cols.append('item')
        
        if group_cols:
            # Create lags within each group
            for lag in lags:
                df[f'{target_col}_lag_{lag}'] = df.groupby(group_cols)[target_col].shift(lag)
        else:
            # Create lags for entire series
            for lag in lags:
                df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str = 'sales',
                               windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Args:
            df: Input DataFrame (should be sorted by date)
            target_col: Name of the target column
            windows: List of window sizes for rolling features
            
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        
        # Group by store and item if they exist
        group_cols = []
        if 'store' in df.columns:
            group_cols.append('store')
        if 'item' in df.columns:
            group_cols.append('item')
        
        if group_cols:
            # Create rolling features within each group
            for window in windows:
                grouped = df.groupby(group_cols)[target_col]
                df[f'{target_col}_roll_mean_{window}'] = grouped.rolling(window=window, min_periods=1).mean().reset_index(level=group_cols, drop=True)
                df[f'{target_col}_roll_std_{window}'] = grouped.rolling(window=window, min_periods=1).std().reset_index(level=group_cols, drop=True)
                df[f'{target_col}_roll_min_{window}'] = grouped.rolling(window=window, min_periods=1).min().reset_index(level=group_cols, drop=True)
                df[f'{target_col}_roll_max_{window}'] = grouped.rolling(window=window, min_periods=1).max().reset_index(level=group_cols, drop=True)
        else:
            # Create rolling features for entire series
            for window in windows:
                df[f'{target_col}_roll_mean_{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
                df[f'{target_col}_roll_std_{window}'] = df[target_col].rolling(window=window, min_periods=1).std()
                df[f'{target_col}_roll_min_{window}'] = df[target_col].rolling(window=window, min_periods=1).min()
                df[f'{target_col}_roll_max_{window}'] = df[target_col].rolling(window=window, min_periods=1).max()
        
        return df
    
    def prepare_for_modeling(self, df: pd.DataFrame, target_col: str = 'sales',
                           test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for modeling by splitting into train/test sets.
        
        Args:
            df: Preprocessed DataFrame
            target_col: Name of the target column
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple of (train_df, test_df)
        """
        df = df.copy()
        
        # Sort by date
        if 'date' in df.columns:
            df = df.sort_values('date')
        
        # Split based on time
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        return train_df, test_df
    
    def get_feature_columns(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
        """
        Get list of feature columns for modeling.
        
        Args:
            df: DataFrame
            exclude_cols: Columns to exclude from features
            
        Returns:
            List of feature column names
        """
        if exclude_cols is None:
            exclude_cols = ['date', 'sales']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols