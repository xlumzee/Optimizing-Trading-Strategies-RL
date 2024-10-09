# main_pipeline.py

# Importing necessary functions from each module
from DataLoader_Preprocesser import load_data, clean_data, normalize_or_scale_features
from DataVisualization import plot_histograms, correlation_analysis, plot_moving_averages, plot_scatter, plot_boxplots, plot_pairplot
from FeatureEngineering import (add_time_based_features, add_cumulative_features, 
                                add_statistical_features, add_technical_indicators, 
                                create_lagged_features, calculate_rolling_statistics)

import os
import pandas as pd

def run_pipeline(filepath, output_dir):
    """
    Run the complete pipeline: Data Loading -> Preprocessing -> Feature Engineering -> Visualization.
    
    Args:
        filepath (str): The file path of the dataset to load.
        output_dir (str): Directory to save the feature-engineered dataset.
    """
    print("Step 1: Loading data...")
    df = load_data(filepath)

    print("Step 2: Cleaning data...")
    df = clean_data(df)

    print("Step 3: Normalizing selected features...")
    df = normalize_or_scale_features(df, columns=['PRC', 'VOL_CHANGE'], method='minmax')

    #Feature Engineering
    print("Step 4: Adding time-based features...")
    df = add_time_based_features(df, date_column='date')

    print("Step 5: Adding cumulative features...")
    df = add_cumulative_features(df, return_column='RET')

    print("Step 6: Adding statistical features...")
    df = add_statistical_features(df, return_column='RET', window=10)

    print("Step 7: Adding technical indicators...")
    df = add_technical_indicators(df, return_column='RET', volume_column='VOL_CHANGE')

    print("Step 8: Creating lagged features...")
    df = create_lagged_features(df, column_name='RET', lags=[1, 7, 14, 30])

    print("Step 9: Calculating rolling statistics...")
    df = calculate_rolling_statistics(df, column_prefix='RET', windows=[7, 14, 30])

    # Save the feature-engineered DataFrame to a CSV file
    os.makedirs(output_dir, exist_ok=True) 
    feature_engineered_filepath = os.path.join(output_dir, 'AKAM_feature_engineeredv2.csv')
    df.to_csv(feature_engineered_filepath, index=False)
    print(f"Feature engineered data saved to: {feature_engineered_filepath}")

    #Data Visualization
    print("Step 10: Generating visualizations...")

    # Generate various plots for visualization
    plot_histograms(df, ['PRC', 'RET', 'VOL_CHANGE', 'ASK', 'BID'])
    correlation_analysis(df)
    plot_moving_averages(df, ['PRC', 'RET', 'VOL_CHANGE', 'ASK', 'BID'], [10, 20, 50])
    plot_scatter(df, 'PRC', 'RET', add_line=True)
    plot_boxplots(df, ['RET', 'VOL_CHANGE', 'PRC'])
    plot_pairplot(df[['RET', 'VOL_CHANGE', 'ILLIQUIDITY', 'PRC']])

    print("Pipeline executed successfully!")

if __name__ == "__main__":
    # Define file paths and directories
    input_filepath = '/Users/amulya/Desktop/Capstone/DSCI-601-Amy/Data/AKAM.csv' 
    output_directory = '/Users/amulya/Desktop/Capstone/DSCI-601-Amy/Data/FeatureEngineered'

    # Run the complete pipeline
    run_pipeline(input_filepath, output_directory)



