
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from DataLoader_Preprocesser import load_data


# ### Data Visualization Pipeline

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
    Display a heatmap of the correlation matrix for the DataFrame. Selects only numeric columns

    Args:
        df (DataFrame): Pandas DataFrame containing the data.

    Returns:
        None: Displays the heatmap of the correlation matrix.
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Check if there are numeric columns in the DataFrame
    if numeric_df.empty:
        print("No numeric columns found in the DataFrame for correlation analysis.")
        return

    # Calculate and display the correlation matrix
    correlation_matrix = numeric_df.corr()
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
    # Determine the number of subplots needed
    num_plots = len(metrics)
    num_rows = (num_plots // 2) + (num_plots % 2)  # Calculate number of rows dynamically
    num_cols = 2  # Fixed number of columns

    # Create the subplots dynamically based on the number of metrics
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, num_rows * 5))
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    # Loop through each metric and create the moving average plots
    for i, metric in enumerate(metrics):
        for ma in ma_days:
            df[f'MA for {ma} days {metric}'] = df[metric].rolling(window=ma).mean()
        
        # Check if the current index exceeds the number of axes; if yes, skip plotting
        if i >= len(axes):
            print(f"Skipping plot for {metric} as no more subplots are available.")
            continue

        # Plot the original metric and its moving averages
        df[[metric] + [f'MA for {ma} days {metric}' for ma in ma_days]].plot(ax=axes[i])
        axes[i].set_title(f'{metric.upper()} Moving Averages')

    # Hide any empty subplots if the number of metrics is less than available subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

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


if __name__ == "__main__":
    # Load the data
    filepath = '/Users/amulya/Desktop/Capstone/DSCI-601-Amy/Data/AKAM.csv'
    data_df = load_data(filepath)

    plot_histograms(data_df, ['PRC','RET', 'VOL_CHANGE', 'ASK','BID'])

    correlation_analysis(data_df)

    plot_moving_averages(data_df, ['PRC','RET', 'VOL_CHANGE', 'ASK','BID'], [10, 20, 50])

    plot_scatter(data_df, 'PRC', 'RET', add_line=True)

    plot_boxplots(data_df, ['RET', 'VOL_CHANGE', 'PRC'])

    plot_pairplot(data_df[['RET', 'VOL_CHANGE', 'ILLIQUIDITY', 'PRC']])




