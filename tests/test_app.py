"""Pytest suite for the road-accident severity API.

Covers the rubric requirements: health check, UI load, domain logic
(high-risk input must score above low-risk input) and boundary validation
(out-of-range values must return HTTP 422).
"""

import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

LOW_RISK = {
    "speed_limit": 25,
    "num_vehicles": 1,
    "num_casualties": 0,
    "driver_age": 45,
    "weather": "clear",
    "road_type": "urban",
    "light_condition": "daylight",
    "hour": 10,
    "is_weekend": 0,
}

HIGH_RISK = {
    "speed_limit": 70,
    "num_vehicles": 4,
    "num_casualties": 3,
    "driver_age": 19,
    "weather": "snow",
    "road_type": "highway",
    "light_condition": "dark",
    "hour": 2,
    "is_weekend": 1,
}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_ui_loads():
    response = client.get("/")
    assert response.status_code == 200
    assert "Road Accident Severity" in response.text


def test_prediction_contract():
    response = client.post("/predict", json=LOW_RISK)
    assert response.status_code == 200

    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in {"Severe", "Non-severe"}
    assert 0.0 <= body["severity_probability"] <= 1.0
    assert 2 <= len(body["tips"]) <= 3


def test_high_risk_case_scores_above_low_risk_case():
    low = client.post("/predict", json=LOW_RISK).json()
    high = client.post("/predict", json=HIGH_RISK).json()

    assert low["prediction"] == 0
    assert high["prediction"] == 1
    assert high["severity_probability"] > low["severity_probability"] + 0.3


@pytest.mark.parametrize(
    "field,value",
    [
        ("speed_limit", 999),
        ("driver_age", 15),
        ("num_vehicles", 0),
        ("hour", 24),
        ("weather", "storm"),
        ("road_type", "motorway"),
        ("is_weekend", 2),
    ],
)
def test_out_of_contract_input_returns_422(field, value):
    payload = dict(LOW_RISK)
    payload[field] = value
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_boundary_values_are_accepted():
    payload = dict(LOW_RISK)
    payload.update({"speed_limit": 120, "driver_age": 16, "hour": 0})
    assert client.post("/predict", json=payload).status_code == 200


def test_missing_field_returns_422():
    payload = dict(LOW_RISK)
    payload.pop("weather")
    assert client.post("/predict", json=payload).status_code == 422
