
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


# def handle_missing_values(df, strategy='drop', fill_value=None):
#     """
#     Handle missing values in a DataFrame based on a specified strategy.

#     Args:
#         df (pd.DataFrame): The DataFrame with potential missing values.
#         strategy (str): The strategy to handle missing values. Options are:
#                         - 'drop': Drop rows with missing values.
#                         - 'fill': Fill missing values with a specified value or method.

#     Returns:
#         pd.DataFrame: A DataFrame with missing values handled based on the chosen strategy.
#     """
#     if strategy == 'drop':
#         df = df.dropna()
#     elif strategy == 'fill':
#         if fill_value is not None:
#             df = df.fillna(fill_value)
#         else:
#             df = df.fillna(df.mean())  # Default: fill with mean values
#     else:
#         raise ValueError("Invalid strategy! Choose either 'drop' or 'fill'.")

#     return df

# write proper function for converting datatypes

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

def data_summary(df):
    """
    Generate a summary of the DataFrame, including data types, missing values, 
    unique values, and basic descriptive statistics.

    Args:
        df (pd.DataFrame): Input DataFrame for which the summary is to be generated.

    Returns:
        pd.DataFrame: A summary DataFrame that includes:
            - Data Type: The data type of each column.
            - Missing #: The number of missing values in each column.
            - Missing %: The percentage of missing values in each column.
            - Unique Values: The number of unique values in each column.
            - Min: The minimum value for numeric columns.
            - Max: The maximum value for numeric columns.
            - Mean: The mean value for numeric columns.
            - Std: The standard deviation for numeric columns.
    

    """
    # Create a summary DataFrame with basic information about the columns
    summ = pd.DataFrame(df.dtypes, columns=['Data Type'])
    summ['Missing #'] = df.isnull().sum()
    summ['Missing %'] = (df.isnull().sum() / len(df)) * 100
    summ['Unique Values'] = df.nunique()
    
    # Calculate descriptive statistics for numeric columns and join with the summary
    desc = df.describe().transpose()
    summ = summ.join(desc[['min', 'max', 'mean', 'std']], how='left')
    
    return summ




if __name__ == "__main__":
    # Example file path (adjust this path based on your environment)
    filepath = '/Users/amulya/Desktop/Capstone/DSCI-601-Amy/Data/AKAM.csv'

    # Load the data
    df = load_data(filepath)

    # clean data
    df = clean_data(df)
    
    # Handle missing values (e.g., fill with mean values)
    #df = handle_missing_values(df, strategy='fill')

    # Normalize or scale selected features
    df = normalize_or_scale_features(df, columns=['PRC', 'VOL_CHANGE'], method='minmax')

    df = data_summary(df)

    print(df.head(10))



