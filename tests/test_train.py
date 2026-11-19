"""Tests for the leakage-safe training utilities in ``train.py``."""

from itertools import combinations
from pathlib import Path

import pandas as pd
import pytest

from schema import FEATURES, TARGET
from train import resample_training, select_best, split_dataset, tune_threshold

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "accidents.csv"


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def test_split_is_stratified_and_disjoint(frame):
    splits = split_dataset(frame, test_size=0.2, validation_size=0.2, random_state=42)
    assert len(splits["x_train"]) == 7881
    assert len(splits["x_val"]) == 1971
    assert len(splits["x_test"]) == 2464

    names = ["x_train", "x_val", "x_test"]
    for left, right in combinations(names, 2):
        assert splits[left].index.intersection(splits[right].index).empty

    prevalences = [splits[f"y_{name.split('_')[1]}"].mean() for name in names]
    for prevalence in prevalences:
        assert prevalence == pytest.approx(frame[TARGET].mean(), abs=0.005)


def test_random_oversampling_only_touches_training_rows(frame):
    features = frame[FEATURES].head(400).copy()
    target = frame[TARGET].head(400).astype(int).copy()
    features_before = features.copy(deep=True)
    target_before = target.copy(deep=True)

    resampled_features, resampled_target, info = resample_training(
        features, target, {"balancing": "random_oversample", "random_state": 42}
    )

    counts = resampled_target.value_counts()
    assert counts[0] == counts[1] == int(target.value_counts().max())
    assert len(resampled_features) == len(resampled_target)
    assert info["method"] == "random_oversample"
    assert info["before"]["rows"] == 400
    assert info["after"]["rows"] == len(resampled_target)

    # The caller's training rows must not be modified in place.
    assert features.equals(features_before)
    assert target.equals(target_before)


def test_threshold_tuning_respects_predicted_severe_rate_cap():
    target = [0] * 80 + [1] * 20
    probabilities = [0.1] * 80 + [0.8] * 20

    threshold = tune_threshold(target, probabilities, max_predicted_severe_rate=0.25)
    predictions = (pd.Series(probabilities) >= threshold).astype(int)

    assert predictions.mean() <= 0.25
    assert predictions.sum() == 20


def test_model_selection_uses_validation_metrics_only():
    def result(name, macro_f1, f1, recall, precision, predicted_rate):
        return {
            "config": {"name": name},
            "validation_tuned_metrics": {
                "macro_f1": macro_f1,
                "f1": f1,
                "recall": recall,
                "precision": precision,
                "predicted_severe_rate": predicted_rate,
            },
        }

    best_macro = result("best-macro", 0.6200, 0.3000, 0.40, 0.35, 0.20)
    better_severe = result("better-severe", 0.6150, 0.3600, 0.50, 0.30, 0.25)
    degenerate = result("all-severe", 0.7000, 0.4200, 0.95, 0.16, 0.90)

    assert select_best([best_macro, better_severe, degenerate]) is better_severe
