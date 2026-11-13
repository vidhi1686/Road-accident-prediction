#!/usr/bin/env python3
"""Build the cleaned, leak-safe accident dataset from the original Kaggle CSV.

This script only performs deterministic, row-preserving transformations:

* derive ``hour`` from ``Time`` and ``is_weekend`` from ``Day_of_week``;
* map ``Accident_severity`` to the binary target ``severe``
  (Serious Injury / Fatal injury -> 1, Slight Injury -> 0);
* keep the 15 approved report-time features and drop every casualty-outcome
  column that would leak the label;
* normalise whitespace and inconsistent categorical spellings;
* convert empty/``na``/``nan`` tokens to real missing values while preserving
  genuine ``Unknown`` categories.

No imputer, scaler, encoder or feature selector is fitted here.  All learned
preprocessing lives inside the scikit-learn training pipeline and is fitted
after the train/validation/test split.

Usage:
    python data/preprocess.py
    python data/preprocess.py --input data/raw/Road.csv --output data/accidents.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "raw" / "Road.csv"
DEFAULT_OUTPUT = BASE_DIR / "accidents.csv"

NUMERIC_FEATURES = [
    "hour",
    "is_weekend",
    "num_vehicles",
    "num_casualties",
]

ORDINAL_FEATURES = [
    "age_band_of_driver",
    "driving_experience",
]

NOMINAL_FEATURES = [
    "sex_of_driver",
    "type_of_vehicle",
    "area_accident_occurred",
    "types_of_junction",
    "road_surface_type",
    "road_surface_conditions",
    "light_conditions",
    "weather_conditions",
    "type_of_collision",
]

FEATURES = NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES
TARGET = "severe"

# Source column -> output column.  ``hour``, ``is_weekend`` and ``severe`` are
# derived separately.
COLUMN_RENAMES = {
    "Number_of_vehicles_involved": "num_vehicles",
    "Number_of_casualties": "num_casualties",
    "Age_band_of_driver": "age_band_of_driver",
    "Driving_experience": "driving_experience",
    "Sex_of_driver": "sex_of_driver",
    "Type_of_vehicle": "type_of_vehicle",
    "Area_accident_occured": "area_accident_occurred",
    "Types_of_Junction": "types_of_junction",
    "Road_surface_type": "road_surface_type",
    "Road_surface_conditions": "road_surface_conditions",
    "Light_conditions": "light_conditions",
    "Weather_conditions": "weather_conditions",
    "Type_of_collision": "type_of_collision",
}

REQUIRED_SOURCE_COLUMNS = [
    "Time",
    "Day_of_week",
    "Accident_severity",
    *COLUMN_RENAMES.keys(),
]

TARGET_VALUES = {"Slight Injury", "Serious Injury", "Fatal injury"}
SEVERE_TARGET_VALUES = {"Serious Injury", "Fatal injury"}

# Tokens that represent missing data in the raw CSV.  Empty strings are also
# converted to NaN after whitespace stripping.
NA_TOKENS = ["", "na", "nan", "NaN", "N/A", "NA", "None", "null", "NULL", "n/a"]

# Casualty/outcome columns that must never be used as input features.
LEAKAGE_COLUMNS = [
    "Casualty_class",
    "Sex_of_casualty",
    "Age_band_of_casualty",
    "Casualty_severity",
    "Work_of_casuality",
    "Fitness_of_casuality",
    "Pedestrian_movement",
]


def _normalize_categorical(series: pd.Series) -> pd.Series:
    """Trim whitespace, unify ``unknown`` spellings and mark missing tokens."""
    values = series.astype("string").str.strip()
    values = values.mask(values.str.len() == 0)
    values = values.mask(values.str.lower().isin(["na", "nan", "n/a", "none", "null"]))
    values = values.replace({"unknown": "Unknown"})
    return values


def build(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    """Transform the raw Kaggle CSV into the cleaned modelling dataset."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    raw = pd.read_csv(input_path, keep_default_na=True, na_values=NA_TOKENS)

    missing_columns = [column for column in REQUIRED_SOURCE_COLUMNS if column not in raw.columns]
    if missing_columns:
        raise SystemExit(f"Raw dataset is missing required columns: {missing_columns}")

    if raw["Accident_severity"].isna().any():
        raise SystemExit("Raw dataset contains missing Accident_severity values.")

    unexpected_targets = set(raw["Accident_severity"].unique()) - TARGET_VALUES
    if unexpected_targets:
        raise SystemExit(f"Unexpected Accident_severity values: {sorted(unexpected_targets)}")

    times = pd.to_datetime(
        raw["Time"].astype("string").str.strip(),
        format="%H:%M:%S",
        errors="coerce",
    )
    if times.isna().any():
        bad = raw.loc[times.isna(), "Time"].head(5).tolist()
        raise SystemExit(f"Could not parse Time values, examples: {bad}")

    days = raw["Day_of_week"].astype("string").str.strip()

    cleaned = pd.DataFrame(index=raw.index)
    cleaned["hour"] = times.dt.hour.astype("Int64")
    cleaned["is_weekend"] = days.isin(["Saturday", "Sunday"]).astype("Int64")

    for source, target in COLUMN_RENAMES.items():
        if target in NUMERIC_FEATURES:
            cleaned[target] = pd.to_numeric(raw[source], errors="coerce")
        else:
            cleaned[target] = _normalize_categorical(raw[source])

    cleaned[TARGET] = raw["Accident_severity"].isin(SEVERE_TARGET_VALUES).astype("int64")

    cleaned = cleaned[FEATURES + [TARGET]]

    # Never silently drop rows: the cleaned file must preserve every source row.
    if len(cleaned) != len(raw):
        raise SystemExit(
            f"Row count changed during preprocessing: {len(raw)} -> {len(cleaned)}"
        )

    present_leakage = [column for column in LEAKAGE_COLUMNS if column in cleaned.columns]
    if present_leakage:
        raise SystemExit(f"Leakage columns present in output: {present_leakage}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    print(f"Input:  {input_path} ({raw.shape[0]} rows x {raw.shape[1]} columns)")
    print(f"Output: {output_path} ({cleaned.shape[0]} rows x {cleaned.shape[1]} columns)")
    print("\nOriginal target distribution:")
    print(raw["Accident_severity"].value_counts().to_string())
    print("\nBinary target distribution:")
    target_counts = cleaned[TARGET].value_counts().sort_index()
    for label, count in target_counts.items():
        name = "Severe" if label == 1 else "Non-severe"
        print(f"  {label} ({name}): {count} ({count / len(cleaned):.4%})")
    print(f"  Severe prevalence: {cleaned[TARGET].mean():.4f}")
    print("\nMissing values in output:")
    missing = cleaned.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  None")
    else:
        for column, count in missing.items():
            print(f"  {column}: {count} ({count / len(cleaned):.2%})")
    print("\nCategorical value counts:")
    for column in ORDINAL_FEATURES + NOMINAL_FEATURES:
        print(f"\n  {column}:")
        print(cleaned[column].value_counts(dropna=False).to_string().replace("\n", "\n  "))

    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the cleaned accident dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Raw Kaggle CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Cleaned output CSV.")
    args = parser.parse_args()
    build(args.input, args.output)


if __name__ == "__main__":
    main()
