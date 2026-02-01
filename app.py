import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from src.config import Config
from src.io import read_files
from src.train_test import train_model, evaluate_model  
from src.preprocessing import build_dataframe
from src.features import add_lag_roll_features
from src.preprocessing import build_dataframe
from src.train_test_split import train_test_split

# Read data and merge files(train, features, stores)
df = read_files(
    train_path=Config.train_path,features_path=Config.features_path,stores_path=Config.stores_path)

# Preprocess data
df = build_dataframe(df)

# Feature engineering, add lag and rolling features
df = add_lag_roll_features(df)

# Split data into train and test sets
X_train, y_train, X_test, y_test = train_test_split(df)

# Train model
model = train_model(X_train, y_train)

# Evaluate model
metrics = evaluate_model(model, X_test, y_test)

# Print evaluation metrics
print("Evaluation Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")


import joblib
from pathlib import Path
from src.config import Config

cfg = Config()
cfg.models_dir.mkdir(parents=True, exist_ok=True)

artifact = {
    "model": model,
    "feature_cols": X_train.columns.tolist(),
}

joblib.dump(artifact, cfg.models_dir / "artifact.pkl")
print(f"Saved artifact to: {cfg.models_dir / 'artifact.pkl'}")










