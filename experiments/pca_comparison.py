#!/usr/bin/env python3
"""Compare the production training pipeline with and without PCA.

The experiment reuses ``train.py`` for the dataset split, preprocessing
pipeline, validation threshold tuning and candidate selection.  It never
writes to ``model/``.  Results are written to ``experiments/`` only.

Usage:
    python experiments/pca_comparison.py
    python experiments/pca_comparison.py --json experiments/pca_results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from schema import FEATURES, TARGET  # noqa: E402
from train import (  # noqa: E402
    MAX_PREDICTED_SEVERE_RATE,
    build_pipeline,
    fit_pipeline,
    metrics_at_threshold,
    select_best,
    split_dataset,
    tune_threshold,
)

DEFAULT_DATA = REPO_ROOT / "data" / "accidents.csv"
DEFAULT_JSON = REPO_ROOT / "experiments" / "pca_results.json"
DEFAULT_REPORT = REPO_ROOT / "experiments" / "pca_results.md"
PRODUCTION_METRICS = REPO_ROOT / "model" / "metrics.json"

ESTIMATORS = [
    {
        "name": "baseline-rf",
        "algorithm": "rf",
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_leaf": 1,
        "balancing": "none",
    },
    {
        "name": "baseline-logreg",
        "algorithm": "logreg",
        "C": 0.1,
        "balancing": "none",
    },
]

PCA_COMPONENT_GRID = [2, 5, 10, 15, 20, 30, 40, 60]
REPORTED_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "roc_auc",
    "average_precision",
    "predicted_severe_rate",
]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in FEATURES + [TARGET] if column not in frame.columns]
    if missing:
        raise SystemExit(f"Dataset is missing required columns: {missing}")
    if frame[TARGET].isna().any():
        raise SystemExit("Target contains missing values.")
    if set(frame[TARGET].unique()) - {0, 1}:
        raise SystemExit("Target must be binary (0/1).")
    return frame


def build_pca_pipeline(config: dict, n_components: int) -> Pipeline:
    """Insert PCA between the production preprocessor and estimator."""
    base = build_pipeline(config)
    return Pipeline(
        [
            ("preprocessor", base.named_steps["preprocessor"]),
            (
                "pca",
                PCA(n_components=n_components, random_state=config.get("random_state", 42)),
            ),
            ("model", base.named_steps["model"]),
        ]
    )


def explained_variance(pipeline: Pipeline) -> tuple[float, list[float]]:
    ratios = np.asarray(pipeline.named_steps["pca"].explained_variance_ratio_, dtype=float)
    return float(ratios.sum()), [float(value) for value in ratios]


def fit_and_score(
    pipeline: Pipeline,
    train_features: pd.DataFrame,
    train_target: pd.Series,
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
    config: dict,
) -> tuple[float, dict]:
    fit_pipeline(pipeline, train_features, train_target, config)
    probabilities = pipeline.predict_proba(validation_features)[:, 1]
    threshold = tune_threshold(validation_target, probabilities)
    metrics = metrics_at_threshold(validation_target, probabilities, threshold)
    return threshold, metrics


def test_pipeline(
    pipeline: Pipeline,
    splits: dict,
    threshold: float,
    config: dict,
) -> dict:
    fit_pipeline(
        pipeline,
        splits["x_train_val"],
        splits["y_train_val"],
        config,
    )
    probabilities = pipeline.predict_proba(splits["x_test"])[:, 1]
    return metrics_at_threshold(splits["y_test"], probabilities, threshold)


def run(data_path: Path, random_state: int, test_size: float, validation_size: float) -> dict:
    frame = load_dataset(data_path)
    splits = split_dataset(
        frame,
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state,
    )
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()

    results: dict = {
        "experiment": "pca_comparison",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(data_path.relative_to(REPO_ROOT))
            if data_path.is_relative_to(REPO_ROOT)
            else str(data_path),
            "rows": int(len(frame)),
            "content_hash": data_hash,
            "target_prevalence": float(frame[TARGET].mean()),
        },
        "split": {
            "random_state": random_state,
            "test_size": test_size,
            "validation_size": validation_size,
            "train_rows": int(len(splits["y_train"])),
            "validation_rows": int(len(splits["y_val"])),
            "test_rows": int(len(splits["y_test"])),
        },
        "max_predicted_severe_rate": MAX_PREDICTED_SEVERE_RATE,
        "pca_component_grid": PCA_COMPONENT_GRID,
        "estimators": {},
        "comparison": {},
    }

    for base_config in ESTIMATORS:
        config = dict(base_config)
        config["random_state"] = random_state
        name = config["name"]

        no_pca_pipeline = build_pipeline(config)
        no_pca_threshold, no_pca_validation = fit_and_score(
            no_pca_pipeline,
            splits["x_train"],
            splits["y_train"],
            splits["x_val"],
            splits["y_val"],
            config,
        )

        pca_search = []
        for n_components in PCA_COMPONENT_GRID:
            pca_pipeline = build_pca_pipeline(config, n_components)
            threshold, validation = fit_and_score(
                pca_pipeline,
                splits["x_train"],
                splits["y_train"],
                splits["x_val"],
                splits["y_val"],
                config,
            )
            total_variance, ratios = explained_variance(pca_pipeline)
            pca_search.append(
                {
                    "config": {
                        "name": f"{name}-pca-{n_components}",
                        "algorithm": config["algorithm"],
                        "balancing": config["balancing"],
                    },
                    "n_components": n_components,
                    "threshold": threshold,
                    "explained_variance_total": total_variance,
                    "explained_variance_ratio": ratios,
                    "validation_tuned_metrics": validation,
                }
            )

        best_pca = select_best(pca_search)
        best_components = int(best_pca["n_components"])

        no_pca_test = test_pipeline(build_pipeline(config), splits, no_pca_threshold, config)
        pca_test = test_pipeline(
            build_pca_pipeline(config, best_components),
            splits,
            best_pca["threshold"],
            config,
        )
        no_pca_test["explained_variance_total"] = None
        pca_test["explained_variance_total"] = best_pca["explained_variance_total"]

        delta = {
            metric: float(pca_test[metric] - no_pca_test[metric])
            for metric in REPORTED_METRICS
        }

        if delta["macro_f1"] > 0 and delta["f1"] > 0:
            verdict = "PCA improves held-out performance"
        elif delta["macro_f1"] < 0 and delta["f1"] < 0:
            verdict = "PCA reduces held-out performance"
        else:
            verdict = "mixed held-out performance"

        results["estimators"][name] = {
            "config": config,
            "no_pca": {
                "threshold": no_pca_threshold,
                "validation_metrics": no_pca_validation,
                "test_metrics": no_pca_test,
            },
            "pca": {
                "best_n_components": best_components,
                "explained_variance_total": best_pca["explained_variance_total"],
                "explained_variance_ratio": best_pca["explained_variance_ratio"],
                "threshold": best_pca["threshold"],
                "validation_metrics": best_pca["validation_tuned_metrics"],
                "test_metrics": pca_test,
                "search": pca_search,
            },
            "delta_pca_minus_no_pca_test": delta,
            "verdict": verdict,
        }
        results["comparison"][name] = {
            "validation_macro_f1_no_pca": float(no_pca_validation["macro_f1"]),
            "validation_macro_f1_pca": float(best_pca["validation_tuned_metrics"]["macro_f1"]),
            "test_macro_f1_no_pca": float(no_pca_test["macro_f1"]),
            "test_macro_f1_pca": float(pca_test["macro_f1"]),
            "delta_test_macro_f1": delta["macro_f1"],
            "delta_test_severe_f1": delta["f1"],
            "verdict": verdict,
        }

    if PRODUCTION_METRICS.exists():
        production = json.loads(PRODUCTION_METRICS.read_text())
        baseline = results["estimators"].get("baseline-rf")
        if baseline is not None:
            production_test = production.get("test_metrics", {})
            mismatches = [
                metric
                for metric in REPORTED_METRICS
                if metric in production_test
                and not np.isclose(
                    baseline["no_pca"]["test_metrics"][metric],
                    production_test[metric],
                    rtol=0,
                    atol=1e-12,
                )
            ]
            results["production_reference_check"] = {
                "compared_to": "model/metrics.json",
                "baseline_rf_mismatches": mismatches,
                "reproduced": not mismatches,
            }

    return results


def format_value(value: float | int | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_report(results: dict, path: Path) -> None:
    lines: list[str] = []
    lines.append("# PCA vs No-PCA Comparison")
    lines.append("")
    lines.append(f"Generated: `{results['generated_at']}`")
    lines.append(
        f"Dataset: `{results['dataset']['rows']}` rows, "
        f"severe prevalence `{results['dataset']['target_prevalence']:.4%}`"
    )
    lines.append(
        f"Split: train `{results['split']['train_rows']}`, "
        f"validation `{results['split']['validation_rows']}`, "
        f"test `{results['split']['test_rows']}` (random_state "
        f"`{results['split']['random_state']}`)"
    )
    lines.append("")

    lines.append("## Validation comparison")
    lines.append("")
    lines.append(
        "| Estimator | Variant | n_components | Explained variance | Threshold | Accuracy | Balanced acc. | Severe precision | Severe recall | Severe F1 | Macro F1 | ROC-AUC | PR-AUC |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, item in results["estimators"].items():
        for variant, key in (("no PCA", "no_pca"), ("PCA", "pca")):
            bundle = item[key]
            metrics = bundle["validation_metrics"]
            components = bundle.get("best_n_components", "—") if key == "pca" else "—"
            variance = bundle.get("explained_variance_total") if key == "pca" else None
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        variant,
                        str(components),
                        format_value(variance),
                        format_value(bundle["threshold"]),
                        format_value(metrics["accuracy"]),
                        format_value(metrics["balanced_accuracy"]),
                        format_value(metrics["precision"]),
                        format_value(metrics["recall"]),
                        format_value(metrics["f1"]),
                        format_value(metrics["macro_f1"]),
                        format_value(metrics["roc_auc"]),
                        format_value(metrics["average_precision"]),
                    ]
                )
                + " |"
            )
    lines.append("")

    lines.append("## Held-out test comparison")
    lines.append("")
    lines.append(
        "| Estimator | Variant | n_components | Explained variance | Threshold | Accuracy | Balanced acc. | Severe precision | Severe recall | Severe F1 | Macro F1 | ROC-AUC | PR-AUC | Confusion matrix |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    )
    for name, item in results["estimators"].items():
        for variant, key in (("no PCA", "no_pca"), ("PCA", "pca")):
            bundle = item[key]
            metrics = bundle["test_metrics"]
            components = bundle.get("best_n_components", "—") if key == "pca" else "—"
            variance = bundle.get("explained_variance_total") if key == "pca" else None
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        variant,
                        str(components),
                        format_value(variance),
                        format_value(bundle["threshold"]),
                        format_value(metrics["accuracy"]),
                        format_value(metrics["balanced_accuracy"]),
                        format_value(metrics["precision"]),
                        format_value(metrics["recall"]),
                        format_value(metrics["f1"]),
                        format_value(metrics["macro_f1"]),
                        format_value(metrics["roc_auc"]),
                        format_value(metrics["average_precision"]),
                        str(metrics["confusion_matrix"]),
                    ]
                )
                + " |"
            )
    lines.append("")

    lines.append("## PCA component search (validation only)")
    lines.append("")
    for name, item in results["estimators"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(
            "| n_components | Explained variance | Threshold | Accuracy | Balanced acc. | Severe F1 | Macro F1 | Predicted severe rate |"
        )
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for candidate in item["pca"]["search"]:
            metrics = candidate["validation_tuned_metrics"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(candidate["n_components"]),
                        format_value(candidate["explained_variance_total"]),
                        format_value(candidate["threshold"]),
                        format_value(metrics["accuracy"]),
                        format_value(metrics["balanced_accuracy"]),
                        format_value(metrics["f1"]),
                        format_value(metrics["macro_f1"]),
                        format_value(metrics["predicted_severe_rate"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    for name, item in results["estimators"].items():
        comparison = results["comparison"][name]
        lines.append(
            f"- **{name}**: {item['verdict']}. "
            f"Test macro F1 {format_value(comparison['test_macro_f1_no_pca'])} → "
            f"{format_value(comparison['test_macro_f1_pca'])} "
            f"(Δ {comparison['delta_test_macro_f1']:+.4f}); "
            f"severe F1 Δ {comparison['delta_test_severe_f1']:+.4f}. "
            f"Best PCA size: {item['pca']['best_n_components']} components "
            f"({item['pca']['explained_variance_total']:.2%} variance)."
        )
    if "production_reference_check" in results:
        check = results["production_reference_check"]
        lines.append("")
        lines.append(
            "Production reference check (no-PCA Random Forest vs "
            f"`model/metrics.json`): {'reproduced exactly' if check['reproduced'] else 'mismatch: ' + ', '.join(check['baseline_rf_mismatches'])}."
        )
    lines.append("")

    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare no-PCA and PCA training pipelines.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    args = parser.parse_args()

    results = run(args.data, args.random_state, args.test_size, args.validation_size)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(results, indent=2, default=float) + "\n")
    write_report(results, args.report)

    print(f"Results written to {args.json}")
    print(f"Report written to {args.report}")
    for name, item in results["estimators"].items():
        comparison = results["comparison"][name]
        print(
            f"{name}: no-PCA test macro F1={comparison['test_macro_f1_no_pca']:.4f}, "
            f"PCA({item['pca']['best_n_components']}) test macro F1={comparison['test_macro_f1_pca']:.4f} "
            f"-> {item['verdict']}"
        )


if __name__ == "__main__":
    main()
