"""Pytest suite for the accident-severity API."""

import json

import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import app
from schema import FEATURES

client = TestClient(app)

VALID_PAYLOAD = {
    "hour": 18,
    "is_weekend": 0,
    "num_vehicles": 2,
    "num_casualties": 1,
    "age_band_of_driver": "31-50",
    "driving_experience": "5-10yr",
    "sex_of_driver": "Male",
    "type_of_vehicle": "Automobile",
    "area_accident_occurred": "Office areas",
    "types_of_junction": "No junction",
    "road_surface_type": "Asphalt roads",
    "road_surface_conditions": "Dry",
    "light_conditions": "Daylight",
    "weather_conditions": "Normal",
    "type_of_collision": "Vehicle with vehicle collision",
}


@pytest.fixture(autouse=True)
def isolate_prediction_log(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "LOG_PATH", tmp_path / "predictions.jsonl")


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_ui_loads():
    response = client.get("/")
    assert response.status_code == 200
    assert "Accident Severity Classifier" in response.text


def test_ui_contains_all_feature_fields():
    html = client.get("/").text
    for feature in FEATURES:
        assert f'name="{feature}"' in html


def test_categories_endpoint():
    response = client.get("/categories")
    assert response.status_code == 200
    categories = response.json()
    assert categories["weather_conditions"] == [
        "Cloudy",
        "Fog or mist",
        "Normal",
        "Other",
        "Raining",
        "Raining and Windy",
        "Snow",
        "Unknown",
        "Windy",
    ]
    assert "age_band_of_driver" in categories


def test_prediction_contract():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in {"Severe", "Non-severe"}
    assert 0.0 <= body["severity_probability"] <= 1.0
    assert 2 <= len(body["tips"]) <= 3


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["algorithm"] in {"rf", "gb", "logreg"}
    assert "test_metrics" in body
    assert "metrics" in body
    assert 0.0 <= body["test_metrics"]["roc_auc"] <= 1.0
    assert body["top_features"]


def test_prediction_logging():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    lines = app_module.LOG_PATH.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    for key in VALID_PAYLOAD:
        assert record[key] == VALID_PAYLOAD[key]
    assert record["prediction"] == response.json()["prediction"]
    assert record["timestamp"]


def test_missing_field_returns_422():
    payload = dict(VALID_PAYLOAD)
    payload.pop("weather_conditions")
    assert client.post("/predict", json=payload).status_code == 422


def test_extra_field_returns_422():
    payload = dict(VALID_PAYLOAD)
    payload["unexpected"] = 1
    assert client.post("/predict", json=payload).status_code == 422


def test_invalid_category_returns_422():
    payload = dict(VALID_PAYLOAD)
    payload["weather_conditions"] = "Storm"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "weather_conditions" in response.text


@pytest.mark.parametrize(
    "field,value",
    [
        ("hour", 24),
        ("hour", -1),
        ("is_weekend", 2),
        ("num_vehicles", 0),
        ("num_casualties", -1),
    ],
)
def test_invalid_numeric_values_return_422(field, value):
    payload = dict(VALID_PAYLOAD)
    payload[field] = value
    assert client.post("/predict", json=payload).status_code == 422


def test_boundary_values_are_accepted():
    payload = dict(VALID_PAYLOAD)
    payload.update({"hour": 0, "is_weekend": 1, "num_vehicles": 1, "num_casualties": 0})
    assert client.post("/predict", json=payload).status_code == 200


def test_unknown_ordinal_category_returns_422():
    payload = dict(VALID_PAYLOAD)
    payload["age_band_of_driver"] = "65-80"
    assert client.post("/predict", json=payload).status_code == 422
