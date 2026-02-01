from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    # project root (walmart-forecast/)
    root: Path = Path(__file__).resolve().parent.parent

    # raw data paths
    data_raw: Path = root / "data"
    train_path: Path = data_raw / "train.csv"
    test_path: Path = data_raw / "test.csv"
    features_path: Path = data_raw / "features.csv"
    stores_path: Path = data_raw / "stores.csv"

    # output paths
    models_dir: Path = root / "models"
    model_path: Path = models_dir / "model.pkl"
    feature_cols_path: Path = models_dir / "feature_cols.json"

    outputs_dir: Path = root / "outputs"
    submission_path: Path = outputs_dir / "submissions" / "submission.csv"

    # dataset columns
    date_col: str = "Date"
    target: str = "Weekly_Sales"
    keys: tuple[str, str] = ("Store", "Dept")

    # training
    split_date: str = "2012-01-01"

    # feature params
    lags: tuple[int, ...] = (1, 2, 4, 8, 12)
    windows: tuple[int, ...] = (4, 8, 12)

    # model choice
    model_name: str = "xgb"   # "rf" or "xgb"
