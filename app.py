"""FastAPI service for accident-severity classification.

The model classifies the severity of a reported road accident from
accident-report information.  It is not a pre-accident crash predictor.

The complete preprocessing-and-model pipeline is deserialized once at process
start, never inside a request handler.  Every prediction is appended to
``logs/predictions.jsonl`` for later drift analysis.
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from schema import CATEGORICAL_FEATURES, CATEGORY_VALUES, FEATURES

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.joblib"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"
LOG_PATH = BASE_DIR / "logs" / "predictions.jsonl"
STATIC_DIR = BASE_DIR / "static"

MODEL = joblib.load(MODEL_PATH)

_log_lock = threading.Lock()


class AccidentInput(BaseModel):
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    is_weekend: int = Field(..., ge=0, le=1, description="1 if the accident was on a weekend")
    num_vehicles: int = Field(..., ge=1, le=50, description="Vehicles involved")
    num_casualties: int = Field(..., ge=0, le=100, description="Reported casualties")
    age_band_of_driver: str
    driving_experience: str
    sex_of_driver: str
    type_of_vehicle: str
    area_accident_occurred: str
    types_of_junction: str
    road_surface_type: str
    road_surface_conditions: str
    light_conditions: str
    weather_conditions: str
    type_of_collision: str

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
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
        },
    )

    @field_validator(*CATEGORICAL_FEATURES, mode="before")
    @classmethod
    def validate_category(cls, value: object, info) -> object:
        allowed = CATEGORY_VALUES[info.field_name]
        if value not in allowed:
            raise ValueError(f"must be one of: {', '.join(allowed)}")
        return value


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    severity_probability: float
    tips: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


MODEL_INFO: dict = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
DECISION_THRESHOLD = float(MODEL_INFO.get("decision_threshold", 0.5))


def top_feature_importance(limit: int = 8) -> list[dict]:
    preprocessor = MODEL.named_steps["preprocessor"]
    estimator = MODEL.named_steps["model"]
    names = preprocessor.get_feature_names_out()

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float)
        values = np.abs(coefficients).mean(axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
    else:
        return []

    if len(values) != len(names):
        return []

    ranked = sorted(zip(names, values), key=lambda item: float(item[1]), reverse=True)
    return [
        {"feature": str(name).split("__", 1)[-1], "importance": round(float(value), 6)}
        for name, value in ranked[:limit]
    ]


MODEL_INFO["top_features"] = top_feature_importance()


def base_tips(payload: AccidentInput) -> list[str]:
    tips: list[str] = []
    if payload.weather_conditions in {"Raining", "Raining and Windy", "Snow", "Fog or mist"}:
        tips.append("Adverse weather: reduce speed and increase your following distance.")
    if payload.road_surface_conditions in {"Wet or damp", "Snow", "Flood over 3cm. deep"}:
        tips.append("Slippery or flooded road: brake gently and avoid sudden steering.")
    if payload.light_conditions.startswith("Darkness"):
        tips.append("Low visibility: use headlights and watch for pedestrians and cyclists.")
    if payload.num_casualties >= 2:
        tips.append("Multiple casualties reported: prioritise emergency response and secure the scene.")
    if payload.is_weekend == 1 and (payload.hour >= 22 or payload.hour < 6):
        tips.append("Late-night weekend travel is high risk: avoid fatigue and alcohol.")
    if payload.type_of_collision in {
        "Collision with pedestrians",
        "Rollover",
        "Collision with animals",
        "Fall from vehicles",
    }:
        tips.append("This collision type carries higher injury risk: request a medical assessment.")
    return tips


def build_tips(payload: AccidentInput, probability: float) -> list[str]:
    tips = base_tips(payload)
    if probability >= DECISION_THRESHOLD:
        tips.append("Model flags this report as high severity: prioritise emergency response.")
    else:
        tips.append("Model flags this report as lower severity: document the scene and exchange details.")
    if len(tips) < 2:
        tips.append("Keep your speed appropriate for the posted limit and road conditions.")
    return tips[:3]


def log_prediction(payload: AccidentInput, result: PredictionResponse) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload.model_dump(),
        **result.model_dump(),
    }
    with _log_lock, LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


app = FastAPI(
    title="Accident Severity Classifier",
    description=(
        "Classifies the severity of a reported road accident from accident-report "
        "information. This is not a pre-accident crash predictor."
    ),
    version="2.0.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=MODEL is not None)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/categories")
def categories() -> dict:
    return CATEGORY_VALUES


@app.get("/model-info")
def model_info() -> dict:
    return MODEL_INFO


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: AccidentInput) -> PredictionResponse:
    frame = pd.DataFrame([payload.model_dump()])[FEATURES]
    probability = float(MODEL.predict_proba(frame)[0, 1])
    prediction = int(probability >= DECISION_THRESHOLD)
    result = PredictionResponse(
        prediction=prediction,
        label="Severe" if prediction == 1 else "Non-severe",
        severity_probability=round(probability, 4),
        tips=build_tips(payload, probability),
    )
    log_prediction(payload, result)
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
