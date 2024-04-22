#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm


# ### Data preprocessing and feature engineering functions

# In[4]:


def load_data(filepath):
    """
    Load data from a specified CSV file and convert the 'date' column to datetime format.

    Args:
        filepath (str): The path to the CSV file to be loaded.

    Returns:
        pd.DataFrame: A DataFrame with the 'date' column converted to datetime objects.
    """
    data = pd.read_csv(filepath)
    data['date'] = pd.to_datetime(data['date'])
    return data

def process_date_column(df, date_column):
    """
    Convert a specified column in a DataFrame to datetime and extract day, month, and year components.
    
    Args:
    df (pd.DataFrame): DataFrame containing the data.
    date_column (str): Name of the column to convert to datetime and extract components.

    Returns:
    pd.DataFrame: The original DataFrame with the date column converted to datetime and new columns for day, month, and year.
    """
    # Convert the column to datetime
    df[date_column] = pd.to_datetime(df[date_column])

    # Extract day, month, and year into separate columns
    df[f'{date_column}_day'] = df[date_column].dt.day
    df[f'{date_column}_month'] = df[date_column].dt.month
    df[f'{date_column}_year'] = df[date_column].dt.year

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
        df[f'{column_prefix}_mean_{window}d'] = df[f'{column_prefix}_lag_{window}'].rolling(window=window, min_periods=1).mean()
        df[f'{column_prefix}_std_{window}d'] = df[f'{column_prefix}_lag_{window}'].rolling(window=window, min_periods=1).std()
    return df



# In[14]:


df = load_data('/home/amy/work/RIT/TDess/DSCI-601-Amy/Data/Combined/combined_AAPL.csv')


# In[15]:


df = process_date_column(df, 'date')


# #### Setting the index

# In[16]:


df.set_index('date', inplace=True)


# In[19]:


# Define lags and window sizes
lags = [1, 7, 30]
windows = [1, 7, 30] # for rolling window

# Create lagged features
df = create_lagged_features(df, 'RET', lags)


# In[21]:


df = calculate_rolling_statistics(df, 'RET', windows)


# In[23]:


print(df.head())


# In[ ]:




