import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# def load_data(file_path):
#     df = pd.read_csv(file_path)
    
#     df.fillna(method='ffill', inplace=True)

#     df.dropna(inplace=True)
#     df = df.select_dtypes(include=[float, int])
#     df.reset_index(drop=True, inplace=True)
#     # Feature scaling
#     scaler = MinMaxScaler()
#     df[df.columns] = scaler.fit_transform(df[df.columns])
#     return df

def load_data(file_path):
    df = pd.read_csv(file_path)
    # Data preprocessing steps
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    print(f"Length after dropna: {len(df)}")
    # Now extract the 'date' column after dropping NaNs
    date_column = df[['date']]
    print(f"Length of date_column: {len(date_column)}")
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    df_numeric = df[numeric_cols]
    print(f"Length of df_numeric: {len(df_numeric)}")
    # Combine 'date' with numeric data
    df = pd.concat([date_column, df_numeric], axis=1)
    # Feature scaling
    scaler = MinMaxScaler()
    df[df_numeric.columns] = scaler.fit_transform(df_numeric)
    return df


