from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from noshow_iq.database import (
    get_last_predictions,
    get_stats,
    insert_prediction,
    now_utc,
)
from noshow_iq.model import MODEL_PATH, load_model, predict
from noshow_iq.preprocess import clean_single_record
from noshow_iq.schemas import AppointmentRequest, PredictionResponse


app_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        app_state["model"] = load_model(MODEL_PATH)
    else:
        app_state["model"] = None
    yield


app = FastAPI(
    title="NoShowIQ",
    description="Appointment no-show prediction API.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": app_state.get("model") is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_appointment(payload: AppointmentRequest):
    model = app_state.get("model")

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Train the model before prediction.",
        )

    raw_input = payload.model_dump()

    try:
        cleaned_features = clean_single_record(raw_input)
        prediction = predict(model, cleaned_features)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    document = {
        "timestamp": now_utc(),
        "raw_input": raw_input,
        "cleaned_features": cleaned_features.iloc[0].to_dict(),
        "risk_level": prediction["risk_level"],
        "probability": prediction["probability"],
        "recommendation": prediction["recommendation"],
    }

    try:
        insert_prediction(document)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction made but MongoDB logging failed: {exc}",
        ) from exc

    return prediction


@app.get("/history")
def history():
    try:
        return {"predictions": get_last_predictions(limit=20)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/stats")
def stats():
    try:
        return get_stats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
