import pandas as pd
import numpy as np


def read_files(train_path,features_path,stores_path)->pd.DataFrame:
    """Reads the input files and merges them into a single DataFrame.

    Args:
        train_path (str): Path to the training data CSV file.
        features_path (str): Path to the features data CSV file.
        stores_path         (str): Path to the stores data CSV file.

    Returns:
        pd.DataFrame: Merged DataFrame containing training, features, and stores data.
    """
    # Read the CSV files into DataFrames
    df_train = pd.read_csv(train_path, parse_dates=['Date'])
    df_features = pd.read_csv(features_path, parse_dates=['Date'])
    df_stores = pd.read_csv(stores_path )

    # Merge the DataFrames
    df_merged = pd.merge(df_train, df_features, on=['Store', 'Date', 'IsHoliday'], how='inner')
    df = pd.merge(df_merged, df_stores, on='Store', how='inner')

    return df




