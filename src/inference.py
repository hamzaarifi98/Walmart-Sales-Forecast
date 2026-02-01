from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import joblib
import pandas as pd

from src.preprocessing import build_dataframe
from src.features import add_lag_roll_features


def load_artifact(artifact_path: str | Path) -> dict:
    artifact = joblib.load(str(artifact_path))
    if not isinstance(artifact, dict) or "model" not in artifact or "feature_cols" not in artifact:
        raise ValueError("Invalid artifact. Expected dict with keys: 'model', 'feature_cols'.")
    return artifact


def build_inference_frame(
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    *,
    date_col: str = "Date",
    target_col: str = "Weekly_Sales",
) -> pd.DataFrame:
    """
    Combine history (with target) + future (target missing) so lag/rolling features
    can be computed for future rows without leakage.
    """
    hist = history_df.copy()
    fut = future_df.copy()

    # Ensure Date is datetime
    hist[date_col] = pd.to_datetime(hist[date_col], errors="coerce")
    fut[date_col] = pd.to_datetime(fut[date_col], errors="coerce")

    # Future rows: target is unknown
    if target_col not in fut.columns:
        fut[target_col] = pd.NA

    combined = pd.concat([hist, fut], axis=0, ignore_index=True)
    combined = build_dataframe(combined)          # your preprocessing (dummies, fills, date features)
    combined = add_lag_roll_features(combined)    # your leakage-safe lag/rolling features
    return combined


def predict_batch(
    artifact_path: str | Path,
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    *,
    id_cols: list[str] = ["Store", "Dept", "Date"],
    date_col: str = "Date",
    target_col: str = "Weekly_Sales",
) -> pd.DataFrame:
    artifact = load_artifact(artifact_path)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    combined = build_inference_frame(history_df, future_df, date_col=date_col, target_col=target_col)

    # We only want predictions for the future portion (where target is NA in the combined view)
    to_pred = combined[combined[target_col].isna()].copy()

    # Build X exactly like training: drop target + Date (you do this in train_test_split.py)
    drop_cols = [target_col]
    if date_col in to_pred.columns:
        drop_cols.append(date_col)

    X = to_pred.drop(columns=drop_cols, errors="ignore")

    # Align columns to training schema (VERY IMPORTANT for one-hot / feature drift)
    X = X.reindex(columns=feature_cols, fill_value=0)

    preds = model.predict(X)

    out = to_pred[id_cols].copy()
    out["prediction"] = preds
    return out


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch inference for Walmart sales forecasting.")
    parser.add_argument("--artifact", type=str, required=True, help="Path to saved joblib artifact (.pkl).")
    parser.add_argument("--history", type=str, required=True, help="CSV with history INCLUDING Weekly_Sales.")
    parser.add_argument("--future", type=str, required=True, help="CSV with future rows (no Weekly_Sales needed).")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path.")
    args = parser.parse_args()

    history_df = pd.read_csv(args.history, parse_dates=["Date"])
    future_df = pd.read_csv(args.future, parse_dates=["Date"])

    preds = predict_batch(args.artifact, history_df, future_df)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.output, index=False)
    print(f"Wrote predictions to: {args.output}")


if __name__ == "__main__":
    main()
