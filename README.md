## Walmart Sales End-to-End project 

This project forecasts Walmart Weekly Sales in Store, Department level, where as a model it uses **XGBoost** where **r^2 = 0.973**. I have created lag and rolling features to get more precise result. **Lag and rolls are shifted by k value in history data to prevent leakage**. Also I have done **split train test by date** not randomly to prevent data leakage. Forecasting is easy by writing this code for inference in terminal:


python -m src.inference \
  --artifact models/artifact.pkl \  
  --history data/train.csv \
  --future data/test.csv \
  --output outputs/predictions.csv

It will get the future data and with the model that has been trained will forecast sales for next weeks in **future** data and it exports in output folder the result. 


## The project Step by Step
- Merges raw datasets (train + features + stores)
- Builds preprocessing + engineered features:
  - cyclical date features (sin/cos)
  - leakage-safe lag features and rolling statistics per (Store, Dept)
- Uses a **time-based split** (not random) for evaluation
- Trains an **XGBoost regressor**
- Saves a reusable artifact (`model + feature schema`)
- Runs batch inference from the command line


- Metrics reported: **MSE / MAE / R²**

> Note: Sales are highly autocorrelated; lag/rolling features are strong predictors.  
> The split is time-based to avoid leakage.


## Project structure
- `src/config.py` — central paths + parameters (dataclass)
- `src/io.py` — read/merge CSV files
- `src/preprocessing.py` — date features + cleaning + encoding
- `src/features.py` — lag + rolling features (leakage-safe)
- `src/train_test_split.py` — time-based split
- `src/train_test.py` — train + evaluate functions
- `app.py` — training entry point
- `src/inference.py` — batch inference entry point
