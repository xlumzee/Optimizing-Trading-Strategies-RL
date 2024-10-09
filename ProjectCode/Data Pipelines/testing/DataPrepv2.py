
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


# ### Data preprocessing pipeline

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


def clean_data(df):
    """
    Clean the given DataFrame by dropping duplicate rows and removing unnecessary white spaces.

    Returns:
        pd.DataFrame: A cleaned DataFrame with no duplicate rows and stripped white spaces.
    """
    # Remove any duplicate rows if present
    df = df.drop_duplicates()
    
    # Strip white spaces from headers and text columns
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    return df


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

def handle_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values in a DataFrame based on a specified strategy.

    Args:
        df (pd.DataFrame): The DataFrame with potential missing values.
        strategy (str): The strategy to handle missing values. Options are:
                        - 'drop': Drop rows with missing values.
                        - 'fill': Fill missing values with a specified value or method.

    Returns:
        pd.DataFrame: A DataFrame with missing values handled based on the chosen strategy.
    """
    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'fill':
        if fill_value is not None:
            df = df.fillna(fill_value)
        else:
            df = df.fillna(df.mean())  # Default: fill with mean values
    else:
        raise ValueError("Invalid strategy! Choose either 'drop' or 'fill'.")

    return df


def convert_data_types(df, conversions):
    """
    Convert specified columns to desired data types. Converts columns to desired data types based on a provided dictionary (conversions) 
    where keys are column names and values are the target data types.

    Args:
        df (pd.DataFrame): The DataFrame 
        conversions (dict): A dictionary specifying the column names as keys and desired data types as values.

    Returns:
        pd.DataFrame: A DataFrame with converted data types.
    """
    for column, dtype in conversions.items():
        df[column] = df[column].astype(dtype)
    return df

def normalize_or_scale_features(df, columns, method='standard'):
    """
    Normalize or scale specified columns in a DataFrame using a selected method. Normalizes or scales specified columns using either standard scaling (mean=0, std=1)
    or Min-Max scaling (range [0, 1])

    Args:
        df (pd.DataFrame): The DataFrame 
        columns (list): List of column names to be normalized or scaled.
        method (str): Method for normalization or scaling. Options are:
                      - 'standard': Standard scaling (mean=0, std=1)
                      - 'minmax': Min-Max scaling (range [0,1])

    Returns:
        pd.DataFrame: A DataFrame with normalized or scaled columns.
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid method! Choose either 'standard' or 'minmax'.")
    
    df[columns] = scaler.fit_transform(df[columns])
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


df = load_data('/Users/amulya/Desktop/Capstone/DSCI-601-Amy/Data/AKAM.csv')


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




