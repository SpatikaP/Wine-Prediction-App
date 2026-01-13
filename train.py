from __future__ import annotations

import os
import argparse
import math
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import mlflow
import mlflow.sklearn


def parse_args():
    p = argparse.ArgumentParser("Simple MLflow demo (wine prediction)")
    p.add_argument("--csv", default="data/wine_sample.csv", help="Path to CSV")
    p.add_argument("--target", default="quality", help="Target column name")
    p.add_argument("--experiment", default="wine-prediction", help="MLflow experiment name")
    p.add_argument("--run", default="run-2", help="MLflow run name")
    p.add_argument("--n-estimators", type=int, default=50, help="RandomForest n_estimators")
    p.add_argument("--max-depth", type=int, default=5, help="RandomForest max_depth")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    p.add_argument("--random-state", type=int, default=42, help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()

    # ----------------------------
    # MLflow setup
    # ----------------------------
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:7006")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    # ----------------------------
    # Load data
    # ----------------------------
    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)

    if args.target not in df.columns:
        raise SystemExit(
            f"Target column '{args.target}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # ----------------------------
    # Data validation & cleaning
    # ----------------------------
    rows_before = len(df)
    df = df.dropna(subset=[args.target])
    rows_after = len(df)

    dropped_rows = rows_before - rows_after
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows with NaN target values")

    # Fail fast if dataset becomes empty
    if len(df) == 0:
        raise ValueError("No rows left after dropping NaN targets.")

    # ----------------------------
    # Split features / target
    # ----------------------------
    X = df.drop(columns=[args.target])
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # ----------------------------
    # Train & log with MLflow
    # ----------------------------
    with mlflow.start_run(run_name=args.run):
        # Log parameters
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)

        # Log data stats
        mlflow.log_param("rows_total", rows_after)
        mlflow.log_param("rows_dropped_nan_target", dropped_rows)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        # Safety checks (logged as metrics)
        mlflow.log_metric("y_train_nan_count", int(y_train.isna().sum()))
        mlflow.log_metric("y_test_nan_count", int(y_test.isna().sum()))

        # Train model
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
        )

        model.fit(X_train, y_train)

        # Evaluate
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_test, preds)

        # Log metrics
        mlflow.log_metric("mse", float(mse))
        mlflow.log_metric("rmse", float(rmse))
        mlflow.log_metric("r2", float(r2))

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")


if __name__ == "__main__":
    main()
