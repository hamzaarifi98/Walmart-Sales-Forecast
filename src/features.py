import pandas as pd
import numpy as np
from typing import List

def add_lag_roll_features(
    df: pd.DataFrame,
    keys: List[str] = ["Store", "Dept"],
    date_col: str = "Date",
    target_cols: List[str] = ["Weekly_Sales"],
    lags: List[int] = [1, 2, 4, 8, 12],
    windows: List[int] = [4, 8, 12],
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    Leakage-safe:
      - lag_k uses shift(k) -> only past values
      - rolling uses shift(1) then rolling(window) per group -> no leakage and no cross-group mixing
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.sort_values(keys + [date_col]).reset_index(drop=True)

    for col in target_cols:
        g = out.groupby(keys, sort=False)[col]

        # Lags
        for k in lags:
            out[f"{col}_lag_{k}"] = g.shift(k)

        # Rolling stats computed on strictly past values
        shifted = g.shift(1)

        # IMPORTANT: groupwise rolling
        for w in windows:
            out[f"{col}_roll_mean_{w}"] = shifted.groupby(out[keys].apply(tuple, axis=1), sort=False).rolling(
                window=w, min_periods=min_periods
            ).mean().reset_index(level=0, drop=True)

            out[f"{col}_roll_std_{w}"] = shifted.groupby(out[keys].apply(tuple, axis=1), sort=False).rolling(
                window=w, min_periods=min_periods
            ).std().reset_index(level=0, drop=True)


    return out
