"""FastAPI service for road-accident severity prediction.

The model is deserialized once at process start (module import), never inside a
request handler.  Every prediction is appended to ``logs/predictions.jsonl`` so
input drift can be audited later with ``drift_check.py``.
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from schema import FEATURES

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.joblib"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"
LOG_PATH = BASE_DIR / "logs" / "predictions.jsonl"
STATIC_DIR = BASE_DIR / "static"

MODEL = joblib.load(MODEL_PATH)

_log_lock = threading.Lock()


class AccidentInput(BaseModel):
    speed_limit: int = Field(..., ge=20, le=120, description="Posted speed limit (mph)")
    num_vehicles: int = Field(..., ge=1, le=10, description="Vehicles involved")
    num_casualties: int = Field(..., ge=0, le=10, description="Reported casualties")
    driver_age: int = Field(..., ge=16, le=90, description="Age of the at-fault driver")
    weather: Literal["clear", "rain", "fog", "snow"]
    road_type: Literal["urban", "rural", "highway", "intersection"]
    light_condition: Literal["daylight", "dark", "dusk"]
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    is_weekend: int = Field(..., ge=0, le=1, description="1 if weekend, else 0")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "speed_limit": 60,
                "num_vehicles": 2,
                "num_casualties": 1,
                "driver_age": 24,
                "weather": "rain",
                "road_type": "highway",
                "light_condition": "dark",
                "hour": 22,
                "is_weekend": 1,
            }
        },
    )


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    severity_probability: float
    tips: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def base_tips(payload: AccidentInput) -> list[str]:
    """Rule-based guidance derived from the request (independent of the model)."""
    tips: list[str] = []
    if payload.weather in {"rain", "fog", "snow"}:
        tips.append("Adverse weather: slow down and increase your following distance.")
    if payload.speed_limit >= 60:
        tips.append("High-speed road: keep extra braking distance and avoid risky overtaking.")
    if payload.light_condition == "dark":
        tips.append("Low visibility: use headlights and watch for pedestrians and cyclists.")
    if payload.num_casualties >= 1:
        tips.append("Injuries reported: call emergency services and secure the scene.")
    if payload.is_weekend == 1 and (payload.hour >= 22 or payload.hour < 6):
        tips.append("Late-night weekend travel is high risk: avoid fatigue and alcohol.")
    return tips


def build_tips(payload: AccidentInput, probability: float) -> list[str]:
    tips = base_tips(payload)
    if probability >= 0.5:
        tips.append("Model flags this as high severity: prioritise emergency response.")
    else:
        tips.append("Model flags this as lower severity: document the scene and exchange details.")
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


def top_feature_importance(limit: int = 8) -> list[dict]:
    preprocessor = MODEL.named_steps["preprocessor"]
    estimator = MODEL.named_steps["model"]
    names = preprocessor.get_feature_names_out()

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = abs(estimator.coef_[0])
    else:
        return []

    ranked = sorted(zip(names, values), key=lambda item: float(item[1]), reverse=True)
    return [
        {"feature": str(name).split("__", 1)[-1], "importance": round(float(value), 4)}
        for name, value in ranked[:limit]
    ]


app = FastAPI(
    title="Road Accident Severity Predictor",
    description="Predicts whether a road accident will be severe, with rule-based safety tips.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

MODEL_INFO: dict = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
MODEL_INFO["top_features"] = top_feature_importance()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=MODEL is not None)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/model-info")
def model_info() -> dict:
    return MODEL_INFO


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: AccidentInput) -> PredictionResponse:
    frame = pd.DataFrame([payload.model_dump()])[FEATURES]
    probability = float(MODEL.predict_proba(frame)[0, 1])
    prediction = int(probability >= 0.5)
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

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
