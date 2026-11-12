"""Generate the synthetic road-accident severity dataset.

The dataset is fully reproducible: running ``python data/generate_data.py``
always produces the same 6,000 rows (seeded RNG).  Every feature is sampled
inside the ranges documented in the README data contract, and the label
(``severity``) is driven by a latent logit that encodes the kind of causal
structure a real accident-severity model would learn: higher speed limits,
more vehicles/casualties, bad weather, poor light, rural/highway roads,
novice/elderly drivers and night-time weekend travel all increase risk.

Usage:
    python data/generate_data.py --rows 6000 --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WEATHER = ["clear", "rain", "fog", "snow"]
ROAD_TYPES = ["urban", "rural", "highway", "intersection"]
LIGHT_CONDITIONS = ["daylight", "dark", "dusk"]

SPEED_LIMITS = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
SPEED_PROBS = [0.03, 0.05, 0.10, 0.12, 0.15, 0.14, 0.13, 0.10, 0.08, 0.06, 0.04]

WEATHER_EFFECT = {"clear": 0.0, "rain": 0.5, "fog": 1.1, "snow": 1.3}
ROAD_EFFECT = {"urban": 0.0, "rural": 0.9, "highway": 0.7, "intersection": 0.4}
LIGHT_EFFECT = {"daylight": 0.0, "dusk": 0.5, "dark": 1.0}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(rows: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    speed_limit = rng.choice(SPEED_LIMITS, size=rows, p=SPEED_PROBS)
    num_vehicles = rng.choice([1, 2, 3, 4, 5, 6], size=rows,
                              p=[0.15, 0.45, 0.25, 0.08, 0.05, 0.02])
    num_casualties = np.clip(rng.poisson(0.6, size=rows), 0, 10)
    driver_age = rng.integers(16, 86, size=rows)
    weather = rng.choice(WEATHER, size=rows, p=[0.58, 0.22, 0.08, 0.12])
    road_type = rng.choice(ROAD_TYPES, size=rows, p=[0.45, 0.25, 0.20, 0.10])
    light_condition = rng.choice(LIGHT_CONDITIONS, size=rows, p=[0.62, 0.28, 0.10])
    hour = rng.integers(0, 24, size=rows)
    is_weekend = rng.integers(0, 2, size=rows)

    weather_term = np.vectorize(WEATHER_EFFECT.get)(weather)
    road_term = np.vectorize(ROAD_EFFECT.get)(road_type)
    light_term = np.vectorize(LIGHT_EFFECT.get)(light_condition)

    young_driver = (driver_age < 25).astype(float)
    elderly_driver = (driver_age > 65).astype(float)
    night = ((hour < 6) | (hour >= 22)).astype(float)

    risk_score = (
        0.045 * (speed_limit - 30)
        + 0.90 * num_casualties
        + 0.35 * (num_vehicles - 1)
        + weather_term
        + road_term
        + light_term
        + 0.60 * young_driver
        + 0.40 * elderly_driver
        + 0.30 * is_weekend
        + 0.70 * night
    )
    logit = -6.6 + 1.8 * risk_score
    probability = _sigmoid(logit)
    severity = (rng.random(rows) < probability).astype(int)

    return pd.DataFrame(
        {
            "speed_limit": speed_limit.astype(int),
            "num_vehicles": num_vehicles.astype(int),
            "num_casualties": num_casualties.astype(int),
            "driver_age": driver_age.astype(int),
            "weather": weather,
            "road_type": road_type,
            "light_condition": light_condition,
            "hour": hour.astype(int),
            "is_weekend": is_weekend.astype(int),
            "severity": severity,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/accidents.csv"))
    args = parser.parse_args()

    frame = generate(rows=args.rows, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    print(f"Wrote {len(frame)} rows to {args.out}")
    print(f"Severity prevalence: {frame['severity'].mean():.3f}")
    print(frame.head().to_string(index=False))


if __name__ == "__main__":
    main()
