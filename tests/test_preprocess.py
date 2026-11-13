"""Tests for the deterministic preprocessing step and its schema contract."""

from pathlib import Path

import pandas as pd
import pytest

from data.preprocess import (
    FEATURES as PREPROCESS_FEATURES,
    LEAKAGE_COLUMNS,
    NOMINAL_FEATURES as PREPROCESS_NOMINAL,
    NUMERIC_FEATURES as PREPROCESS_NUMERIC,
    ORDINAL_FEATURES as PREPROCESS_ORDINAL,
    TARGET as PREPROCESS_TARGET,
    build,
)
from schema import (
    CATEGORY_VALUES,
    FEATURES,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_CATEGORIES,
    ORDINAL_FEATURES,
    TARGET,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = REPO_ROOT / "data" / "raw" / "Road.csv"
EXPECTED_ROWS = 12316
EXPECTED_NON_SEVERE = 10415
EXPECTED_SEVERE = 1901


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH)


@pytest.fixture(scope="module")
def cleaned(tmp_path_factory) -> pd.DataFrame:
    output = tmp_path_factory.mktemp("processed") / "accidents.csv"
    return build(RAW_PATH, output)


def test_output_shape_and_row_preservation(cleaned, raw):
    assert cleaned.shape == (EXPECTED_ROWS, 16)
    assert len(cleaned) == len(raw)


def test_output_columns(cleaned):
    assert list(cleaned.columns) == FEATURES + [TARGET]


def test_target_mapping(cleaned, raw):
    assert set(cleaned[TARGET].unique()) == {0, 1}
    assert int((cleaned[TARGET] == 0).sum()) == EXPECTED_NON_SEVERE
    assert int((cleaned[TARGET] == 1).sum()) == EXPECTED_SEVERE

    raw_slight = int((raw["Accident_severity"] == "Slight Injury").sum())
    raw_serious = int((raw["Accident_severity"] == "Serious Injury").sum())
    raw_fatal = int((raw["Accident_severity"] == "Fatal injury").sum())
    assert raw_slight == EXPECTED_NON_SEVERE
    assert raw_serious + raw_fatal == EXPECTED_SEVERE


def test_severe_prevalence(cleaned):
    expected = EXPECTED_SEVERE / EXPECTED_ROWS
    assert cleaned[TARGET].mean() == pytest.approx(expected, abs=1e-12)


def test_leakage_columns_absent(cleaned):
    source_leakage = [
        "Casualty_class",
        "Sex_of_casualty",
        "Age_band_of_casualty",
        "Casualty_severity",
        "Work_of_casuality",
        "Fitness_of_casuality",
        "Pedestrian_movement",
    ]
    for column in LEAKAGE_COLUMNS + source_leakage:
        assert column not in cleaned.columns


def test_schema_lists_match_preprocessing():
    assert NUMERIC_FEATURES == PREPROCESS_NUMERIC
    assert ORDINAL_FEATURES == PREPROCESS_ORDINAL
    assert NOMINAL_FEATURES == PREPROCESS_NOMINAL
    assert FEATURES == PREPROCESS_FEATURES
    assert TARGET == PREPROCESS_TARGET


def test_category_values_match_cleaned_data(cleaned):
    for column in ORDINAL_FEATURES + NOMINAL_FEATURES:
        observed = set(cleaned[column].dropna().astype(str).unique())
        assert set(CATEGORY_VALUES[column]) == observed

    for column, categories in zip(ORDINAL_FEATURES, ORDINAL_CATEGORIES):
        observed = set(cleaned[column].dropna().astype(str).unique())
        assert set(categories) == observed


def test_missing_value_handling(cleaned):
    assert cleaned["driving_experience"].isna().sum() == 829
    assert cleaned["type_of_vehicle"].isna().sum() == 950
    assert cleaned["area_accident_occurred"].isna().sum() == 239
    assert cleaned["types_of_junction"].isna().sum() == 887
    assert cleaned["road_surface_type"].isna().sum() == 172
    assert cleaned["type_of_collision"].isna().sum() == 155

    assert cleaned["age_band_of_driver"].isna().sum() == 0
    assert cleaned["sex_of_driver"].isna().sum() == 0
    assert cleaned["road_surface_conditions"].isna().sum() == 0
    assert cleaned["light_conditions"].isna().sum() == 0
    assert cleaned["weather_conditions"].isna().sum() == 0

    for column in ORDINAL_FEATURES + NOMINAL_FEATURES:
        values = cleaned[column].dropna().astype(str).str.lower()
        assert not values.isin(["na", "nan", "n/a", "none", "null"]).any()


def test_whitespace_and_unknown_normalisation(cleaned):
    assert "Church areas" in set(cleaned["area_accident_occurred"].dropna())
    assert "Recreational areas" in set(cleaned["area_accident_occurred"].dropna())
    assert int((cleaned["driving_experience"] == "Unknown").sum()) == 33
    assert int((cleaned["age_band_of_driver"] == "Unknown").sum()) == 1548


def test_numeric_domains(cleaned):
    assert cleaned["hour"].between(0, 23).all()
    assert set(cleaned["is_weekend"].unique()) == {0, 1}
    assert cleaned["num_vehicles"].between(1, 7).all()
    assert cleaned["num_casualties"].between(1, 8).all()
