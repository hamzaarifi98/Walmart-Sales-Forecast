import pandas as pd
import numpy as np



def train_test_split(
    df: pd.DataFrame,
    date_col: str = "Date",
    split_date: str = "2012-01-01"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into training and testing sets based on a specified date.

    Args:
        df (pd.DataFrame): The input DataFrame containing a date column.
        date_col (str): The name of the date column in the DataFrame.
        split_date (str): The date to split the DataFrame on. Rows with dates
                          before this date go to the training set, and rows
                          on or after this date go to the testing set.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and testing DataFrames.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    train_df = df[df[date_col] < split_date].reset_index(drop=True)
    test_df = df[df[date_col] >= split_date].reset_index(drop=True)

    X_train = train_df.drop(columns=["Weekly_Sales", 'Date'])
    y_train = train_df["Weekly_Sales"]
    X_test = test_df.drop(columns=["Weekly_Sales", 'Date'])
    y_test = test_df["Weekly_Sales"]

    return X_train, y_train, X_test, y_test

