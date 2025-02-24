import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from scipy.stats import entropy

# ### Feature Engineering pipeline

def add_time_based_features(df, date_column='date'):
    """
    Add time-based features such as day of the week, day of the month, month, 
    year, is_month_start, is_month_end, and week number.

    Args:
        df (pd.DataFrame): Input DataFrame with a datetime column.
        date_column (str): The name of the column containing datetime values.

    Returns:
        df with new time-based features.
    """

    df[date_column] = pd.to_datetime(df[date_column])
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['day_of_month'] = df[date_column].dt.day
    df['month'] = df[date_column].dt.month
    df['is_month_start'] = df[date_column].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_column].dt.is_month_end.astype(int)
    df['year'] = df[date_column].dt.year
    df['week'] = df[date_column].dt.isocalendar().week
    return df

def add_cumulative_features(df, return_column='RET'):
    """
    Calculate cumulative features like Exponential Moving Averages (EMA).

    Args:
        df (pd.DataFrame): Input DataFrame with a returns column.
        return_column (str): The name of the column containing return values.

    Returns:
        df with new cumulative features added.
    """

    # Calculate EMA for different spans
    df[f'{return_column}_ema_12'] = df[return_column].ewm(span=12, adjust=False).mean()
    df[f'{return_column}_ema_26'] = df[return_column].ewm(span=26, adjust=False).mean()
    return df

def add_statistical_features(df, return_column='RET', window=10):
    """
    Calculate statistical features like skewness, kurtosis, and entropy.

    For Reference - 
    Skewness: A measure of the asymmetry of the distribution of a dataset around its mean; it indicates whether the data leans more to the left or right.

    Kurtosis: A measure of the "tailedness" or the presence of outliers in the distribution; 
    it indicates whether the distribution has heavier or lighter tails compared to a normal distribution.

    Entropy: A measure of the randomness or uncertainty in a dataset; it quantifies the unpredictability of the information contained within the data.
    Args:
        df (pd.DataFrame): Input DataFrame with a returns column.
        return_column (str): The name of the column containing return values.
        window (int): Window size for calculating rolling statistics.

    Returns:
        df with new statistical features added.
    """
    # Skewness and Kurtosis
    df[f'{return_column}_skew_{window}'] = df[return_column].rolling(window=window).skew()
    df[f'{return_column}_kurtosis_{window}'] = df[return_column].rolling(window=window).kurt()
    
    # Entropy
    #df[f'{return_column}_entropy_{window}'] = df[return_column].rolling(window=window).apply(
        #lambda x: entropy(pd.value_counts(x)), raw=False)
    df[f'{return_column}_entropy_{window}'] = df[return_column].rolling(window=window).apply(
    lambda x: entropy(pd.Series(x).value_counts()), raw=False)

    
    return df


def calculate_rsi(df, return_column='RET', window_length=14):
    """
    Calculate the Relative Strength Index (RSI).

    Args:
        df (pd.DataFrame): Input DataFrame with a returns column.
        return_column (str): The name of the column containing return values.
        window_length (int): The window length for calculating RSI. Typically 2 weeks

    Returns:
        df with RSI added as a new column.
    """
    delta = df[return_column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    return df


def calculate_obv(df, return_column='RET', volume_column='VOL_CHANGE'):
    """
    Calculate the On-Balance Volume (OBV) indicator.
    On-Balance Volume (OBV): A momentum indicator that uses volume changes to predict price movements by cumulatively adding or subtracting volume based on 
    whether the price closes higher or lower than the previous period.

    Args:
        df (pd.DataFrame): Input DataFrame with returns and volume columns.
        return_column (str): The name of the column containing return values.
        volume_column (str): The name of the column containing volume changes.

    Returns:
        pd.DataFrame: DataFrame with OBV added as a new column.
    """
    df['OBV'] = (df[return_column].diff() * df[volume_column]).cumsum()
    return df


def add_technical_indicators(df, return_column='RET', volume_column='VOL_CHANGE'):
    """
    Add technical indicators like RSI and OBV to the df.

    Args:
        df (pd.DataFrame): Input DataFrame with return and volume columns.
        return_column (str): The name of the column containing return values.
        volume_column (str): The name of the column containing volume changes.

    Returns:
        df with RSI and OBV added as new columns.
    """
    df = calculate_rsi(df, return_column=return_column)
    df = calculate_obv(df, return_column=return_column, volume_column=volume_column)
    return df


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
        # Check if lagged column exists before calculation
        lag_column = f'{column_prefix}_lag_{window}'
        if lag_column not in df.columns:
            print(f"Error: {lag_column} not found in DataFrame. Ensure create_lagged_features() includes this lag.")
            continue
        df[f'{column_prefix}_mean_{window}d'] = df[lag_column].rolling(window=window, min_periods=1).mean()
        df[f'{column_prefix}_std_{window}d'] = df[lag_column].rolling(window=window, min_periods=1).std()
    return df


if __name__ == "__main__":
    # Example file path (adjust this path based on your environment)
    filepath = '/Users/amulya/Desktop/Capstone/DSCI-601-Amy/Data/AKAM.csv'

    df = pd.read_csv(filepath)

    df = add_time_based_features(df, date_column='date')
    df = add_cumulative_features(df, return_column='RET')
    df = add_statistical_features(df, return_column='RET', window=10)
    df = add_technical_indicators(df, return_column='RET', volume_column='VOL_CHANGE')


    df = create_lagged_features(df, column_name='RET', lags=[1, 7, 14, 30])

    df = calculate_rolling_statistics(df, column_prefix='RET', windows=[7, 14, 30])

    print("DataFrame with all feature engineering steps applied:\n", df.head())

    output_path = '/Users/amulya/Desktop/Capstone/DSCI-601-Amy/Data/FeatureEngineered/AKAM_feature_engineered.csv'
    df.to_csv(output_path, index=False)
    print(f"Feature engineered data saved to: {output_path}")