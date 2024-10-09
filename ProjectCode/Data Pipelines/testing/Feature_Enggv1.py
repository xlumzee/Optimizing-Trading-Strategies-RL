import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# ### Feature Engineering pipeline

def create_lagged_features(df, column_name, lags):
    """
    Create lagged features for specified time lags.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    column_name (str): Name of the column to create lag features for.
    lags (list of int): List containing the lag periods.

    Returns:
    pd.DataFrame: DataFrame with new columns for each lag feature.
    """
    for lag in lags:
        df[f'{column_name}_lag_{lag}'] = df[column_name].shift(lag)
    return df

def calculate_rolling_statistics(df, column_prefix, windows):
    """
    Calculate rolling mean and standard deviation for each window size.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    column_prefix (str): Prefix of the column names to calculate rolling stats for.
    windows (list of int): List containing the window sizes.

    Returns:
    pd.DataFrame: DataFrame with rolling mean and std added as new columns.
    """
    for window in windows:
        df[f'{column_prefix}_mean_{window}d'] = df[f'{column_prefix}_lag_{window}'].rolling(window=window, min_periods=1).mean()
        df[f'{column_prefix}_std_{window}d'] = df[f'{column_prefix}_lag_{window}'].rolling(window=window, min_periods=1).std()
    return df
