#!/usr/bin/env python3
"""Train and select a road-accident severity classifier on the real dataset.

Methodology
-----------
* One stratified train/validation/test split is created up front and reused for
  every candidate, so no model sees the test set during selection.
* Six candidates are compared on validation macro F1, severe-class recall,
  severe-class precision and ROC-AUC.
* The decision threshold is tuned on validation predictions only.
* The selected pipeline is refitted on train+validation, evaluated exactly once
  on the held-out test set, and only then promoted (when the promotion policy
  allows it).

Deliberate difference from the original notebook
------------------------------------------------
The notebook applied ``VarianceThreshold`` and ``SelectKBest(f_classif, k=50)``
to 183 one-hot columns.  With the curated 15-feature contract the encoded space
is only about 85 columns, so feature selection is not needed to control
dimensionality; it can also discard rare but valid categories and complicates
feature-importance reporting.  It is therefore not retained.  Numeric outliers
are kept (extreme vehicle/casualty counts are plausible accident reports) and
``RobustScaler`` limits their influence, matching the notebook's principle.

Usage:
    python train.py --all --promote always
    python train.py --algorithm logreg --C 0.1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.utils.class_weight import compute_sample_weight

from schema import (
    FEATURES,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_CATEGORIES,
    ORDINAL_FEATURES,
    TARGET,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
MLFLOW_DB = BASE_DIR / "mlflow.db"

EXPERIMENT_NAME = "road-accident-severity-real"
DATASET_NAME = "road_accident_real"
SCALAR_METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc"]

EXPERIMENTS = [
    {"name": "rf-n100-d6", "algorithm": "rf", "n_estimators": 100, "max_depth": 6},
    {"name": "rf-n200-d10", "algorithm": "rf", "n_estimators": 200, "max_depth": 10},
    {
        "name": "gb-n100-d3-lr0.1",
        "algorithm": "gb",
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
    },
    {
        "name": "gb-n200-d2-lr0.05",
        "algorithm": "gb",
        "n_estimators": 200,
        "max_depth": 2,
        "learning_rate": 0.05,
    },
    {"name": "logreg-C1.0", "algorithm": "logreg", "C": 1.0},
    {"name": "logreg-C0.1", "algorithm": "logreg", "C": 0.1},
]


def build_pipeline(config: dict) -> Pipeline:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    ordinal_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=ORDINAL_CATEGORIES,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    nominal_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("ordinal", ordinal_pipeline, ORDINAL_FEATURES),
            ("nominal", nominal_pipeline, NOMINAL_FEATURES),
        ]
    )

    if config["algorithm"] == "rf":
        estimator = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_leaf=config.get("min_samples_leaf", 1),
            class_weight="balanced_subsample",
            random_state=config.get("random_state", 42),
            n_jobs=-1,
        )
    elif config["algorithm"] == "gb":
        estimator = GradientBoostingClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            random_state=config.get("random_state", 42),
        )
    else:
        estimator = LogisticRegression(
            C=config["C"],
            max_iter=3000,
            class_weight="balanced",
            random_state=config.get("random_state", 42),
        )

    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def fit_pipeline(pipeline: Pipeline, features: pd.DataFrame, target: pd.Series, config: dict) -> None:
    if config["algorithm"] == "gb":
        weights = compute_sample_weight(class_weight="balanced", y=target)
        pipeline.fit(features, target, model__sample_weight=weights)
    else:
        pipeline.fit(features, target)


def split_dataset(
    frame: pd.DataFrame,
    test_size: float,
    validation_size: float,
    random_state: int,
):
    features = frame[FEATURES]
    target = frame[TARGET].astype(int)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_train_val,
    )
    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "x_train_val": x_train_val,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_train_val": y_train_val,
    }


def metrics_at_threshold(target: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    target_array = np.asarray(target).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "n": int(len(target_array)),
        "prevalence": float(target_array.mean()),
        "accuracy": float(accuracy_score(target_array, predictions)),
        "precision": float(precision_score(target_array, predictions, zero_division=0)),
        "recall": float(recall_score(target_array, predictions, zero_division=0)),
        "f1": float(f1_score(target_array, predictions, zero_division=0)),
        "macro_f1": float(
            f1_score(target_array, predictions, average="macro", zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(target_array, probabilities)),
        "confusion_matrix": confusion_matrix(
            target_array, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            target_array,
            predictions,
            labels=[0, 1],
            target_names=["Non-severe", "Severe"],
            output_dict=True,
            zero_division=0,
        ),
    }


def tune_threshold(target: pd.Series, probabilities: np.ndarray) -> float:
    target_array = np.asarray(target).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)

    best_threshold = 0.5
    best_key = (-1.0, -1.0, -1.0)
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        predictions = (probabilities >= threshold).astype(int)
        severe_f1 = f1_score(target_array, predictions, zero_division=0)
        severe_recall = recall_score(target_array, predictions, zero_division=0)
        key = (round(float(severe_f1), 6), round(float(severe_recall), 6), -float(threshold))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def select_best(results: list[dict]) -> dict:
    best_macro = max(result["validation_metrics"]["macro_f1"] for result in results)
    shortlist = [
        result
        for result in results
        if best_macro - result["validation_metrics"]["macro_f1"] <= 0.01
    ]
    return max(
        shortlist,
        key=lambda result: (
            result["validation_metrics"]["recall"],
            result["validation_metrics"]["precision"],
            result["validation_metrics"]["macro_f1"],
        ),
    )


def hyperparameters(config: dict) -> dict:
    params = {"algorithm": config["algorithm"]}
    if config["algorithm"] in {"rf", "gb"}:
        params["n_estimators"] = config["n_estimators"]
        params["max_depth"] = config["max_depth"]
    if config["algorithm"] == "rf":
        params["min_samples_leaf"] = config.get("min_samples_leaf", 1)
    if config["algorithm"] == "gb":
        params["learning_rate"] = config["learning_rate"]
    if config["algorithm"] == "logreg":
        params["C"] = config["C"]
    return params


def scalar_metrics(prefix: str, metrics: dict) -> dict:
    return {
        f"{prefix}_{key}": float(metrics[key])
        for key in SCALAR_METRIC_KEYS
        if key in metrics
    }


def config_from_args(args: argparse.Namespace) -> dict:
    config = {"algorithm": args.algorithm, "random_state": args.random_state}
    if args.algorithm == "rf":
        config.update(
            {
                "name": args.name or f"rf-n{args.n_estimators}-d{args.max_depth}",
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "min_samples_leaf": args.min_samples_leaf,
            }
        )
    elif args.algorithm == "gb":
        config.update(
            {
                "name": args.name
                or f"gb-n{args.n_estimators}-d{args.max_depth}-lr{args.learning_rate}",
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "learning_rate": args.learning_rate,
            }
        )
    else:
        config.update(
            {
                "name": args.name or f"logreg-C{args.C}",
                "C": args.C,
            }
        )
    return config


def should_promote(policy: str, validation_metrics: dict) -> bool:
    if policy == "always":
        return True
    if policy == "never":
        return False
    if not METRICS_PATH.exists():
        return True

    previous = json.loads(METRICS_PATH.read_text())
    if previous.get("dataset", {}).get("name") != DATASET_NAME:
        return True
    previous_score = previous.get("validation_metrics", {}).get("macro_f1")
    if previous_score is None:
        return True
    return validation_metrics["macro_f1"] > float(previous_score)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and select the accident severity model.")
    parser.add_argument("--all", action="store_true", help="Run all six experiments and select on validation.")
    parser.add_argument("--algorithm", choices=["rf", "gb", "logreg"], default="gb")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--validation_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data" / "accidents.csv")
    parser.add_argument(
        "--promote",
        choices=["auto", "always", "never"],
        default="auto",
        help="auto compares validation macro F1 against model/metrics.json.",
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Processed dataset not found: {args.data}. Run data/preprocess.py first.")

    frame = pd.read_csv(args.data)
    missing = [column for column in FEATURES + [TARGET] if column not in frame.columns]
    if missing:
        raise SystemExit(f"Dataset is missing required columns: {missing}")
    if frame[TARGET].isna().any():
        raise SystemExit("Target contains missing values.")
    if set(frame[TARGET].unique()) - {0, 1}:
        raise SystemExit("Target must be binary (0/1).")

    data_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()
    splits = split_dataset(
        frame,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )

    print(f"Dataset: {args.data} ({len(frame)} rows)")
    print(f"Target prevalence (severe): {frame[TARGET].mean():.4f}")
    print(
        "Split sizes: "
        f"train={len(splits['y_train'])}, "
        f"validation={len(splits['y_val'])}, "
        f"test={len(splits['y_test'])}"
    )

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    configs = EXPERIMENTS if args.all else [config_from_args(args)]
    results: list[dict] = []

    for config in configs:
        pipeline = build_pipeline(config)
        fit_pipeline(pipeline, splits["x_train"], splits["y_train"], config)
        validation_probabilities = pipeline.predict_proba(splits["x_val"])[:, 1]
        validation_metrics = metrics_at_threshold(
            splits["y_val"], validation_probabilities, 0.5
        )

        candidate_path = MODEL_DIR / f"candidate_{config['name']}.joblib"
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, candidate_path)

        with mlflow.start_run(run_name=config["name"]) as run:
            mlflow.log_params(hyperparameters(config))
            mlflow.log_params(
                {
                    "dataset_name": DATASET_NAME,
                    "dataset_rows": int(len(frame)),
                    "dataset_hash": data_hash,
                    "severe_prevalence": float(frame[TARGET].mean()),
                    "train_rows": int(len(splits["y_train"])),
                    "validation_rows": int(len(splits["y_val"])),
                    "test_rows": int(len(splits["y_test"])),
                    "random_state": int(args.random_state),
                    "test_size": float(args.test_size),
                    "validation_size": float(args.validation_size),
                }
            )
            mlflow.log_metrics(scalar_metrics("val", validation_metrics))
            mlflow.set_tag("selected", "false")
            mlflow.log_artifact(str(candidate_path), artifact_path="model")
            run_id = run.info.run_id

        candidate_path.unlink(missing_ok=True)
        results.append(
            {
                "config": config,
                "run_id": run_id,
                "validation_probabilities": validation_probabilities,
                "validation_metrics": validation_metrics,
            }
        )
        print(
            f"{config['name']:<24} "
            f"macro_f1={validation_metrics['macro_f1']:.4f} "
            f"severe_recall={validation_metrics['recall']:.4f} "
            f"precision={validation_metrics['precision']:.4f} "
            f"roc_auc={validation_metrics['roc_auc']:.4f}"
        )

    selected = select_best(results)
    selected_config = selected["config"]
    threshold = tune_threshold(splits["y_val"], selected["validation_probabilities"])
    selected_validation_tuned = metrics_at_threshold(
        splits["y_val"], selected["validation_probabilities"], threshold
    )

    final_pipeline = build_pipeline(selected_config)
    fit_pipeline(final_pipeline, splits["x_train_val"], splits["y_train_val"], selected_config)
    test_probabilities = final_pipeline.predict_proba(splits["x_test"])[:, 1]
    test_metrics = metrics_at_threshold(splits["y_test"], test_probabilities, threshold)

    promoted = should_promote(args.promote, selected_validation_tuned)

    final_candidate_path = MODEL_DIR / "candidate_selected.joblib"
    joblib.dump(final_pipeline, final_candidate_path)

    dataset_path = str(args.data.resolve())
    try:
        dataset_path = str(args.data.resolve().relative_to(BASE_DIR))
    except ValueError:
        pass

    record = {
        "algorithm": selected_config["algorithm"],
        "experiment_name": selected_config["name"],
        "hyperparameters": hyperparameters(selected_config),
        "features": FEATURES,
        "target": TARGET,
        "dataset": {
            "name": DATASET_NAME,
            "path": dataset_path,
            "source": "data/raw/Road.csv",
            "rows": int(len(frame)),
            "content_hash": data_hash,
            "target_prevalence": float(frame[TARGET].mean()),
            "target_distribution": {
                "non_severe": int((frame[TARGET] == 0).sum()),
                "severe": int((frame[TARGET] == 1).sum()),
            },
        },
        "split": {
            "random_state": int(args.random_state),
            "test_size": float(args.test_size),
            "validation_size": float(args.validation_size),
            "train_rows": int(len(splits["y_train"])),
            "validation_rows": int(len(splits["y_val"])),
            "test_rows": int(len(splits["y_test"])),
        },
        "selection_rule": (
            "Maximise validation macro F1; among candidates within 0.01 of the best, "
            "prefer higher severe-class recall, then precision. Test data is not used."
        ),
        "decision_threshold": threshold,
        "validation_metrics": selected_validation_tuned,
        "validation_metrics_at_0_5": selected["validation_metrics"],
        "test_metrics": test_metrics,
        "metrics": test_metrics,
        "candidate_validation_results": [
            {
                "name": result["config"]["name"],
                "algorithm": result["config"]["algorithm"],
                "hyperparameters": hyperparameters(result["config"]),
                "validation_metrics": result["validation_metrics"],
                "selected": result["config"]["name"] == selected_config["name"],
            }
            for result in results
        ],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "promoted_to_model_joblib": promoted,
    }

    with mlflow.start_run(run_id=selected["run_id"]):
        mlflow.set_tag("selected", "true")
        mlflow.log_metric("decision_threshold", threshold)
        mlflow.log_metrics(scalar_metrics("val_tuned", selected_validation_tuned))
        mlflow.log_metrics(scalar_metrics("test", test_metrics))
        mlflow.log_artifact(str(final_candidate_path), artifact_path="model")
        mlflow.set_tag("promoted_to_model_joblib", str(promoted).lower())

    if promoted:
        shutil.move(final_candidate_path, MODEL_PATH)
        METRICS_PATH.write_text(json.dumps(record, indent=2, default=float) + "\n")
        with mlflow.start_run(run_id=selected["run_id"]):
            mlflow.log_artifact(str(METRICS_PATH))
    else:
        final_candidate_path.unlink(missing_ok=True)

    print("\nSelected model:", selected_config["name"])
    print(f"Decision threshold: {threshold:.2f}")
    print(f"Promoted to model/model.joblib: {promoted}")
    print(json.dumps({"validation": selected_validation_tuned, "test": test_metrics}, indent=2, default=float))


if __name__ == "__main__":
    main()
