import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    # Load the CSV
    df = pd.read_csv(file_path)

    # Fill missing values
    df.fillna(method='ffill', inplace=True)

    # Drop rows with remaining missing values
    df.dropna(inplace=True)

    # Preserve the 'date' column for reference, but exclude it from processing
    date_column = df[['date']]  # Save the 'date' column for later reference

    # Select only numeric columns for scaling and training
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    df_numeric = df[numeric_cols]

    # Feature scaling
    scaler = MinMaxScaler()
    df_numeric[df_numeric.columns] = scaler.fit_transform(df_numeric)

    # Combine 'date' with processed numeric data
    processed_df = pd.concat([date_column, df_numeric], axis=1)

    return processed_df


