#!/usr/bin/env python3
"""Drift check: compare logged prediction inputs against the training data.

Usage:
    python drift_check.py
    python drift_check.py --log logs/predictions.jsonl --threshold 0.5

For each numeric feature it reports how far the logged mean has moved, in units
of the training standard deviation.  For each categorical feature it reports
the total-variation distance between the logged and training distributions.
Exits with status 1 when any feature crosses the threshold (default 0.5 sigma
for numeric features, 0.2 TVD for categoricals).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from schema import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES

BASE_DIR = Path(__file__).resolve().parent


def load_log(path: Path) -> pd.DataFrame:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check prediction-input drift.")
    parser.add_argument("--train", type=Path, default=BASE_DIR / "data" / "accidents.csv")
    parser.add_argument("--log", type=Path, default=BASE_DIR / "logs" / "predictions.jsonl")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--categorical-threshold", type=float, default=0.2)
    args = parser.parse_args()

    if not args.log.exists():
        print(f"No prediction log at {args.log} - nothing to check.")
        return 0

    train = pd.read_csv(args.train)
    log = load_log(args.log)

    if log.empty:
        print(f"Prediction log at {args.log} is empty - nothing to check.")
        return 0
    if len(log) < 30:
        print(f"Warning: only {len(log)} logged predictions; drift estimates are noisy.\n")

    missing = [column for column in FEATURES if column not in log.columns]
    if missing:
        raise SystemExit(f"Log is missing columns: {missing}")

    drifted: list[str] = []

    print(f"{'feature':<18}{'train mean':>12}{'log mean':>12}{'shift (sd)':>12}{'flag':>8}")
    for feature in NUMERIC_FEATURES:
        train_mean = float(train[feature].mean())
        train_std = float(train[feature].std()) or 1.0
        log_mean = float(log[feature].mean())
        shift = (log_mean - train_mean) / train_std
        flagged = abs(shift) > args.threshold
        if flagged:
            drifted.append(feature)
        print(f"{feature:<18}{train_mean:>12.3f}{log_mean:>12.3f}{shift:>12.2f}{'DRIFT' if flagged else 'ok':>8}")

    print(f"\n{'feature':<18}{'train top':>18}{'log top':>18}{'TVD':>8}{'flag':>8}")
    for feature in CATEGORICAL_FEATURES:
        train_dist = train[feature].value_counts(normalize=True)
        log_dist = log[feature].value_counts(normalize=True)
        tvd = 0.5 * float(train_dist.sub(log_dist, fill_value=0).abs().sum())
        flagged = tvd > args.categorical_threshold
        if flagged:
            drifted.append(feature)
        train_top = train_dist.idxmax()
        log_top = log_dist.idxmax() if not log_dist.empty else "-"
        print(f"{feature:<18}{train_top:>18}{log_top:>18}{tvd:>8.2f}{'DRIFT' if flagged else 'ok':>8}")

    if drifted:
        print(f"\nDrift detected in: {', '.join(drifted)}")
        return 1
    print("\nNo meaningful drift detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
