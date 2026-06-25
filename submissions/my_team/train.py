#!/usr/bin/env python3
"""
Training script for the bike-demand submission.

Run from this folder:

    cd submissions/YOUR_TEAM_NAME
    python train.py

Expected dataset:

    ../../dataset/train_set.csv

Output:

    weights.joblib

The evaluator will later load weights.joblib through predict.py.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
from model import create_features


DATA_ROOT = Path("../../dataset")
TRAIN_CSV = DATA_ROOT / "train_set.csv"
OUTPUT_WEIGHTS = "weights.joblib"


def main() -> None:

    train = pd.read_csv(TRAIN_CSV, low_memory=False)

    TARGET = "demand"

    y = train[TARGET]
    X = create_features(train.drop(columns=[TARGET]))

    feature_columns = list(X.columns)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = XGBRegressor(
        n_estimators=700,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    valid_preds = model.predict(X_valid)
    valid_preds = valid_preds.clip(min=0)

    rmse = root_mean_squared_error(y_valid, valid_preds)
    print(f"Validation RMSE: {rmse:.4f}")

    model.fit(X, y)

    artifacts = {
        "model": model,
        "feature_columns": feature_columns,
    }

    joblib.dump(artifacts, OUTPUT_WEIGHTS)

    print(f"Saved {OUTPUT_WEIGHTS}")

if __name__ == "__main__":
    main()