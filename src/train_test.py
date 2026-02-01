import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)


def train_model(X_train: pd.DataFrame,
    y_train: pd.Series
):
   
    model = xgb
    model.fit(X_train, y_train)
    return model

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict[str, float]:
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "MSE": mse,
        "MAE": mae,
        "R2": r2
    }










