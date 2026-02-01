import numpy as np
import pandas as pd


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # make sure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month

    df["week_sin"] = np.sin(2 * np.pi * df["weekofyear"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["weekofyear"] / 52)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df = df.drop(columns=["weekofyear", "month"])
    return df


def clean_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure MarkDown1..5 exist
    markdown_cols = [f"MarkDown{i}" for i in range(1, 6)]
    for c in markdown_cols:
        if c not in df.columns:
            df[c] = np.nan

    # MarkDowns: often "no markdown" => fill 0
    for c in markdown_cols:
        df[c] = df[c].fillna(0)

    # Numeric columns: median fill (safe default)
    for c in ["CPI", "Unemployment", "Temperature", "Fuel_Price"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    # Keep Type as category (we can one-hot encode later)
    if "Type" in df.columns:
        df["Type"] = df["Type"].astype("category")
    

    return df


def sort_for_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)


def build_dataframe(
    df: pd.DataFrame,
    one_hot_type: bool = True
) -> pd.DataFrame:
    """
    Final pipeline:
      1) ensure Date is datetime + sort
      2) add cyclical date features
      3) clean columns (markdowns, medians)
      4) optional one-hot encoding for Type (recommended for RF/XGBoost)
    """
    out = df.copy()

    # 1) sort
    out = sort_for_timeseries(out)

    # 2) add date features
    out = add_date_features(out)

    # 3) clean numeric/markdown/type
    out = clean_feature_columns(out)

    # 4) encode Type for models (RF/XGBoost need numeric)
    if one_hot_type and "Type" in out.columns:
        out = pd.get_dummies(out, columns=["Type"], drop_first=True)

    return out

