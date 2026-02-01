## Walmart Sales End-to-End Project

This project forecasts **weekly Walmart sales at the Store–Department level** using an end-to-end machine learning pipeline. The forecasting model is built with **XGBoost** and achieves an **R² score of 0.973** on a time-based holdout set.

A strong emphasis is placed on **leakage-safe time-series modeling**. Lag and rolling features are computed strictly from historical data using shifted values to prevent future information leakage. Model evaluation uses a **date-based train/test split** rather than random sampling, ensuring realistic performance assessment for forecasting tasks.

The project also includes a **production-style batch inference pipeline**, allowing users to generate future sales predictions directly from the command line.

---

## Batch Inference (Forecasting)

Forecasting future sales is performed via a command-line inference script:

```bash
python -m src.inference \
  --artifact models/artifact.pkl \
  --history data/train.csv \
  --future data/test.csv \
  --output outputs/predictions.csv
```

The inference pipeline:

* Uses historical data to compute lag and rolling features
* Rebuilds the full preprocessing and feature-engineering pipeline
* Aligns inference features with the training schema
* Predicts weekly sales for future dates
* Exports results to the specified output file

---

## Project Workflow

* Merge raw datasets (sales, features, and store metadata)
* Perform preprocessing and feature engineering:

  * Cyclical date features (sine/cosine encoding)
  * Leakage-safe lag features and rolling statistics per (Store, Dept)
* Split data using a **time-based** train/test strategy
* Train an **XGBoost regressor**
* Evaluate performance using **MSE, MAE, and R²**
* Save a reusable model artifact (model + feature schema)
* Run batch inference from the command line

> **Note:** Retail sales are highly autocorrelated, making lag and rolling features strong predictors. Time-based splitting is used to avoid data leakage and ensure realistic evaluation.

---

## Project Structure

* `src/config.py` — Centralized configuration and paths (dataclass)
* `src/io.py` — Data loading and dataset merging
* `src/preprocessing.py` — Date features, cleaning, and encoding
* `src/features.py` — Leakage-safe lag and rolling feature engineering
* `src/train_test_split.py` — Time-based data splitting
* `src/train_test.py` — Model training and evaluation
* `app.py` — Training entry point
* `src/inference.py` — Batch inference entry point


* adapt it for **resume bullets**, or
* rewrite it with a **strong ML-engineer focus** instead of data-science tone.
