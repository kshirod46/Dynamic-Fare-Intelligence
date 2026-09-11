"""
Train the dynamic-pricing fare model and generate deployment artifacts.

Run from project root:
    python src/train_model.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATASET = "tharishreddy22/ride-hailing-cab-trip-dataset"

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "Pickup_City",
    "Vehicle_Type",
    "Distance_KM",
    "Duration_Min",
    "Time_of_Day",
    "Weather_Condition",
    "Traffic_Level",
    "Day_of_Week",
    "Is_Prime_Member",
    "Driver_Rating",
]
TARGET = "Total_Fare"

CATEGORICAL = [
    "Pickup_City",
    "Vehicle_Type",
    "Time_of_Day",
    "Weather_Condition",
    "Traffic_Level",
    "Day_of_Week",
]
NUMERIC = [
    "Distance_KM",
    "Duration_Min",
    "Is_Prime_Member",
    "Driver_Rating",
]


def load_dataset() -> pd.DataFrame:
    path = Path(kagglehub.dataset_download(DATASET))
    csvs = sorted(path.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV file found in {path}")
    train_csvs = [p for p in csvs if "train" in p.name.lower()]
    source = train_csvs[0] if train_csvs else csvs[0]
    df = pd.read_csv(source)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    required = set(FEATURES + [TARGET, "Is_Cancelled"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC),
            ("cat", categorical_pipe, CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "Ridge Regression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=80,
                        max_depth=16,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=160,
                        learning_rate=0.06,
                        max_depth=3,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    print("Downloading/loading dataset...")
    df = load_dataset()
    validate_columns(df)

    completed = df[df["Is_Cancelled"] == 0].copy()
    completed = completed.dropna(subset=[TARGET]).copy()
    if completed.empty:
        raise ValueError("No completed trips with Total_Fare were found.")

    print(f"Raw rows: {len(df):,}")
    print(f"Completed rows used for fare model: {len(completed):,}")

    X = completed[FEATURES].copy()
    y = completed[TARGET].astype(float).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    results = []
    fitted_models: dict[str, Pipeline] = {}

    for name, pipeline in make_models(build_preprocessor()).items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae = float(mean_absolute_error(y_test, pred))
        r2 = float(r2_score(y_test, pred))
        results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})
        fitted_models[name] = pipeline

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    best_name = str(results_df.iloc[0]["Model"])
    best_model = fitted_models[best_name]

    results_df.to_csv(MODEL_DIR / "model_comparison.csv", index=False)

    feature_meta = {
        "features": FEATURES,
        "categorical_features": CATEGORICAL,
        "numeric_features": NUMERIC,
        "target": TARGET,
        "best_model": best_name,
        "random_state": RANDOM_STATE,
        "categories": {c: sorted(completed[c].dropna().astype(str).unique().tolist()) for c in CATEGORICAL},
    }
    (MODEL_DIR / "feature_metadata.json").write_text(
        json.dumps(feature_meta, indent=2), encoding="utf-8"
    )

    dump(best_model, MODEL_DIR / "fare_model.joblib", compress=3)

    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print("Artifacts written to:", MODEL_DIR)


if __name__ == "__main__":
    main()
