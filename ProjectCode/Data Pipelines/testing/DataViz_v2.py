#!/usr/bin/env python
# coding: utf-8

# ## Data Loading and Inital Setup

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose


# ### Data inital prep functions

# In[3]:


def load_data(filepath):
    data = pd.read_csv(filepath)
    data['date'] = pd.to_datetime(data['date'])
    return data

def preprocess_data(df):
    pd.options.display.float_format = '{:,.2f}'.format
    return df

def data_summary(df):
    summ = pd.DataFrame(df.dtypes, columns=['Data Type'])
    summ['Missing #'] = df.isnull().sum()
    summ['Missing %'] = (df.isnull().sum() / len(df)) * 100
    summ['Unique Values'] = df.nunique()
    
    desc = df.describe().transpose()
    summ = summ.join(desc[['min', 'max', 'mean', 'std']])
    
    return summ


# ### Data Visualization functions

# In[4]:


def plot_histograms(df, columns, bins=50):
    """
    Plot histograms for specified columns in the DataFrame.

    Args:
        df (DataFrame): Pandas DataFrame containing the data.
        columns (list): List of column names to plot histograms for.
        bins (int): Number of bins to use for the histograms.

    Returns:
        None: Displays the histogram plots.
    """
    num_plots = len(columns)
    fig, axes = plt.subplots(nrows=(num_plots // 2 + num_plots % 2), ncols=2, figsize=(12, num_plots * 2))
    axes = axes.flatten()
    for idx, col in enumerate(columns):
        sns.histplot(df[col], bins=bins, ax=axes[idx], color=np.random.rand(3,))
        axes[idx].set_title(f'Histogram of {col}')
    plt.tight_layout()
    plt.show()

def correlation_analysis(df):
    """
    Display a heatmap of the correlation matrix for the DataFrame.

    Args:
        df (DataFrame): Pandas DataFrame containing the data.

    Returns:
        None: Displays the heatmap of the correlation matrix.
    """
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def plot_moving_averages(df, metrics, ma_days):
    """
    Plot moving averages for specified metrics over defined days.

    Args:
        df (DataFrame): Pandas DataFrame containing the data.
        metrics (list): List of metric columns to calculate moving averages for.
        ma_days (list): List of day intervals (integers) for calculating moving averages.

    Returns:
        None: Displays the plots of the metrics with their moving averages.
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    axes = axes.flatten()
    for i, metric in enumerate(metrics):
        for ma in ma_days:
            df[f'MA for {ma} days {metric}'] = df[metric].rolling(window=ma).mean()
        df[[metric] + [f'MA for {ma} days {metric}' for ma in ma_days]].plot(ax=axes[i])
        axes[i].set_title(f'{metric.upper()} Moving Averages')
    plt.tight_layout()
    plt.show()

def plot_scatter(df, x, y, hue=None, add_line=False):
    """
    Plot a scatter plot for two variables and optionally add a regression line.

    Args:
        df (DataFrame): Pandas DataFrame containing the data.
        x (str): Column name for the x-axis.
        y (str): Column name for the y-axis.
        hue (str, optional): Column name to color data points by.
        add_line (bool, optional): Whether to add a regression line.

    Returns:
        None: Displays the scatter plot.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=hue)
    plt.title(f'Relationship between {x} and {y}')
    plt.xlabel(x)
    plt.ylabel(y)
    
    if add_line:
        sns.regplot(data=df, x=x, y=y, scatter=False, color='red')
    
    plt.show()

def plot_boxplots(df, columns):
    num_plots = len(columns)
    fig, axes = plt.subplots(nrows=(num_plots // 2 + num_plots % 2), ncols=2, figsize=(12, num_plots * 2))
    axes = axes.flatten()
    for idx, col in enumerate(columns):
        sns.boxplot(x=df[col], ax=axes[idx])
        axes[idx].set_title(f'Box Plot of {col}')
    plt.tight_layout()
    plt.show()

def plot_pairplot(df, hue=None):
    sns.pairplot(df, hue=hue)
    plt.suptitle('Pair Plot of DataFrame Variables', y=1.02)
    plt.show()



# ### Show implementation of the functions

# In[5]:


data_df = load_data('/home/amy/work/RIT/TDess/DSCI-601-Amy/Data/Combined/combined_AAPL.csv')


# In[6]:


data_df


# In[8]:


plot_histograms(data_df, ['RET', 'VOL_CHANGE', 'sprtrn','DJI_Return'])


# In[9]:


correlation_analysis(data_df)


# In[11]:


plot_moving_averages(data_df, ['RET', 'VOL_CHANGE', 'sprtrn','DJI_Return'], [10, 20, 50])


# In[12]:


plot_scatter(data_df, 'RET', 'VOL_CHANGE', add_line=True)


# In[13]:


plot_boxplots(data_df, ['RET', 'VOL_CHANGE', 'ILLIQUIDITY'])


# In[14]:


plot_pairplot(data_df[['RET', 'VOL_CHANGE', 'ILLIQUIDITY', 'sprtrn']])


# In[ ]:





# In[ ]:




