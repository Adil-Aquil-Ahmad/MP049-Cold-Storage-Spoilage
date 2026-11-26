"""Prediction service for MP049: POST /predict, GET /health."""
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from pythonjsonlogger.json import JsonFormatter

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.joblib"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "champion")


def make_logger() -> logging.Logger:
    logger = logging.getLogger("mp049")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


log = make_logger()


class Lot(BaseModel):
    """One stored lot. The client's two rules are in the Field limits."""
    model_config = ConfigDict(extra="forbid")

    storage_days: int = Field(..., ge=0, le=350, description="days in storage; above 350 is impossible")
    temp_set_c: float = Field(..., description="chamber temperature setting, deg C")
    humidity_pct: float = Field(..., ge=0, le=100, description="chamber humidity, percent")
    power_cuts_hours: float = Field(..., ge=0, description="power cut hours in storage")
    bag_weight_kg: float = Field(..., ge=20, description="weight per bag; below 20 is a typing mistake")
    door_openings_per_day: float = Field(..., ge=0, description="average door openings per day")
    crop: Literal["potato", "onion", "apple"]
    chamber: Literal["Chamber 1", "Chamber 2", "Chamber 3"]


class Prediction(BaseModel):
    spoilage_pct: float
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "model", None) is None and MODEL_PATH.exists():
        app.state.model = joblib.load(MODEL_PATH)
        log.info("model loaded", extra={"model_path": str(MODEL_PATH)})
    yield


app = FastAPI(title="MP049 cold storage spoilage", lifespan=lifespan)
app.state.model = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": app.state.model is not None}


@app.post("/predict", response_model=Prediction)
def predict(lot: Lot) -> Prediction:
    model = app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    start = time.perf_counter()
    value = float(model.predict(pd.DataFrame([lot.model_dump()]))[0])
    log.info("prediction", extra={
        "crop": lot.crop,
        "chamber": lot.chamber,
        "storage_days": lot.storage_days,
        "prediction": round(value, 4),
        "model_version": MODEL_VERSION,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    })
    return Prediction(spoilage_pct=value, model_version=MODEL_VERSION)
