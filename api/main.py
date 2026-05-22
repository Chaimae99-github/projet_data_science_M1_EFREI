import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.feature_engineering import build_features
from src.models.save_model import load_artifact


MODEL_PATH = PROJECT_ROOT / "models" / "best_failure_classifier.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

app = FastAPI(
    title="API Maintenance Predictive",
    version="1.0.0",
    description="Prediction du risque de panne machine dans les 24 prochaines heures.",
)


class MachineObservation(BaseModel):
    timestamp: str | None = None
    machine_id: int
    machine_type: str
    vibration_rms: float | None = None
    temperature_motor: float | None = None
    current_phase_avg: float | None = None
    pressure_level: float | None = None
    rpm: float | None = None
    operating_mode: str
    hours_since_maintenance: float | None = None
    ambient_temp: float | None = None


class PredictionResponse(BaseModel):
    failure_probability: float = Field(..., ge=0, le=1)
    failure_within_24h_prediction: int
    threshold: float
    model: str | None = None


class ModelInfoResponse(BaseModel):
    target: str | None = None
    production_model: str | None = None
    best_overall_by_f1: str | None = None
    feature_count: int
    decision_threshold: float | None = None
    model_available: bool


def get_model():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Modele absent. Lancez python src/models/train.py.")
    return load_artifact(MODEL_PATH)


def get_metadata() -> dict[str, Any]:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_available": MODEL_PATH.exists()}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    metadata = get_metadata()
    return ModelInfoResponse(
        target=metadata.get("target"),
        production_model=metadata.get("production_model"),
        best_overall_by_f1=metadata.get("best_overall_by_f1"),
        feature_count=len(metadata.get("feature_columns", [])),
        decision_threshold=metadata.get("decision_threshold"),
        model_available=MODEL_PATH.exists(),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    observation: MachineObservation,
    threshold: float = Query(0.5, ge=0.0, le=1.0),
) -> PredictionResponse:
    model = get_model()
    metadata = get_metadata()
    feature_columns = metadata.get("feature_columns")
    if not feature_columns:
        raise HTTPException(status_code=503, detail="Metadata features absentes. Relancez l'entrainement.")

    row = pd.DataFrame([observation.model_dump()])
    features = build_features(row)
    X = features.reindex(columns=feature_columns)
    probability = float(model.predict_proba(X)[:, 1][0])
    return PredictionResponse(
        failure_probability=probability,
        failure_within_24h_prediction=int(probability >= threshold),
        threshold=threshold,
        model=metadata.get("production_model"),
    )
