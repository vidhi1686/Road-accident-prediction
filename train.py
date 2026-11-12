#!/usr/bin/env python3
"""Train a road-accident severity classifier and track runs with MLflow.

Examples:
    python train.py --algorithm rf --n_estimators 100 --max_depth 6
    python train.py --algorithm logreg --C 0.5
    python train.py --algorithm gb --n_estimators 200 --max_depth 3 --learning_rate 0.05

Every run logs its hyperparameters and test metrics to the SQLite-backed
MLflow store (``mlflow.db``).  The trained pipeline is only promoted to
``model/model.joblib`` when its test F1 beats the currently deployed model, so
the artifact on disk is always the best run so far.
"""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from schema import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
MLFLOW_DB = BASE_DIR / "mlflow.db"

EXPERIMENT_NAME = "road-accident-severity"


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    if args.algorithm == "rf":
        estimator = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
            n_jobs=-1,
        )
    elif args.algorithm == "gb":
        estimator = GradientBoostingClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=args.random_state,
        )
    else:
        estimator = LogisticRegression(
            C=args.C,
            max_iter=2000,
            random_state=args.random_state,
        )

    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def collect_params(args: argparse.Namespace) -> dict:
    params = {
        "algorithm": args.algorithm,
        "test_size": args.test_size,
        "random_state": args.random_state,
    }
    if args.algorithm in {"rf", "gb"}:
        params["n_estimators"] = args.n_estimators
        params["max_depth"] = args.max_depth
    if args.algorithm == "rf":
        params["min_samples_leaf"] = args.min_samples_leaf
    if args.algorithm == "gb":
        params["learning_rate"] = args.learning_rate
    if args.algorithm == "logreg":
        params["C"] = args.C
    return params


def evaluate(pipeline: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict:
    probabilities = pipeline.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the accident severity model.")
    parser.add_argument("--algorithm", choices=["rf", "gb", "logreg"], default="rf")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data" / "accidents.csv")
    args = parser.parse_args()

    frame = pd.read_csv(args.data)
    missing = [column for column in FEATURES + [TARGET] if column not in frame.columns]
    if missing:
        raise SystemExit(f"Dataset is missing required columns: {missing}")

    features = frame[FEATURES]
    target = frame[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=target,
    )

    pipeline = build_pipeline(args)
    pipeline.fit(x_train, y_train)
    metrics = evaluate(pipeline, x_test, y_test)
    params = collect_params(args)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    candidate_path = MODEL_DIR / "candidate.joblib"
    joblib.dump(pipeline, candidate_path)

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    if args.algorithm == "logreg":
        run_name = f"logreg-C{args.C}"
    else:
        run_name = f"{args.algorithm}-n{args.n_estimators}-d{args.max_depth}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("n_rows", len(frame))
        mlflow.log_param("severe_rate", round(float(target.mean()), 4))
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(candidate_path), artifact_path="model")

        previous = None
        if METRICS_PATH.exists():
            previous = json.loads(METRICS_PATH.read_text())

        is_best = previous is None or metrics["f1"] > previous["metrics"]["f1"]
        mlflow.set_tag("promoted_to_model_joblib", str(is_best).lower())

        if is_best:
            shutil.move(candidate_path, MODEL_PATH)
            record = {
                "algorithm": args.algorithm,
                "params": params,
                "metrics": metrics,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "n_rows": int(len(frame)),
                "features": FEATURES,
            }
            METRICS_PATH.write_text(json.dumps(record, indent=2) + "\n")
        else:
            candidate_path.unlink()

    summary = {
        "algorithm": args.algorithm,
        "metrics": metrics,
        "promoted_to_best": is_best,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
