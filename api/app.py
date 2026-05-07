"""
api/app.py
----------
FastAPI REST API for the Time Series Forecasting System.

Endpoints:
  GET  /                  — Health check
  GET  /states            — List all available states
  GET  /forecast/{state}  — 8-week forecast for a given state
  GET  /models/{state}    — Model evaluation metrics for a given state
  GET  /forecast/all      — Forecasts for all states
  POST /retrain           — Trigger retraining pipeline (background task)

Run with:
    uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT                = Path(__file__).resolve().parents[1]
ARTIFACTS_FORECASTS = ROOT / "artifacts" / "forecasts"
ARTIFACTS_RESULTS   = ROOT / "artifacts" / "results"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("api")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Time Series Forecasting API",
    description=(
        "Production-style REST API for per-state sales forecasting. "
        "Models: SARIMA, Prophet, XGBoost, LSTM. "
        "Forecast horizon: 8 weeks."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ForecastPoint(BaseModel):
    date: str
    sales: float


class MetricsSchema(BaseModel):
    rmse: float
    mae: float
    mape: float


class ForecastResponse(BaseModel):
    state: str
    best_model: str
    forecast: List[ForecastPoint]
    metrics: MetricsSchema


class StatesResponse(BaseModel):
    states: List[str]
    count: int


class HealthResponse(BaseModel):
    status: str
    version: str
    forecasts_available: int


class RetrainResponse(BaseModel):
    status: str
    message: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_all_forecasts() -> Dict[str, Dict]:
    """Load all per-state JSON forecast files from artifacts/forecasts/."""
    results = {}
    if not ARTIFACTS_FORECASTS.exists():
        return results
    for fp in ARTIFACTS_FORECASTS.glob("*.json"):
        state = fp.stem
        with open(fp) as f:
            results[state] = json.load(f)
    return results


def _load_forecast(state: str) -> Dict:
    """Load a single state's forecast JSON. Raises 404 if not found."""
    # Normalise: try exact match first, then case-insensitive
    fp = ARTIFACTS_FORECASTS / f"{state}.json"
    if not fp.exists():
        # Case-insensitive search
        candidates = list(ARTIFACTS_FORECASTS.glob("*.json"))
        for c in candidates:
            if c.stem.lower() == state.lower():
                fp = c
                break
        else:
            available = [c.stem for c in candidates]
            raise HTTPException(
                status_code=404,
                detail=f"No forecast found for state '{state}'. "
                       f"Available states: {available}",
            )
    with open(fp) as f:
        return json.load(f)


def _retrain_background():
    """Run the training pipeline as a subprocess."""
    logger.info("[Retrain] Starting background retraining ...")
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "src" / "train_all.py")],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        if result.returncode == 0:
            logger.info("[Retrain] Completed successfully.")
        else:
            logger.error(f"[Retrain] Failed:\n{result.stderr}")
    except Exception as e:
        logger.exception(f"[Retrain] Exception: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["System"])
def health_check():
    """Health check — confirms the API is running and shows artifact count."""
    n = len(list(ARTIFACTS_FORECASTS.glob("*.json"))) if ARTIFACTS_FORECASTS.exists() else 0
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        forecasts_available=n,
    )


@app.get("/states", response_model=StatesResponse, tags=["Data"])
def list_states():
    """Return a list of all states for which forecasts are available."""
    if not ARTIFACTS_FORECASTS.exists():
        return StatesResponse(states=[], count=0)
    states = sorted([fp.stem for fp in ARTIFACTS_FORECASTS.glob("*.json")])
    return StatesResponse(states=states, count=len(states))


@app.get("/forecast/all", tags=["Forecast"])
def forecast_all():
    """Return 8-week forecasts for ALL states."""
    all_data = _load_all_forecasts()
    if not all_data:
        raise HTTPException(
            status_code=503,
            detail="No forecasts available. Run `python src/train_all.py` first.",
        )
    return JSONResponse(content=all_data)


@app.get("/forecast/{state}", response_model=ForecastResponse, tags=["Forecast"])
def forecast_state(state: str):
    """
    Return the 8-week sales forecast for a given state.

    Response includes: best model name, forecast dates/values, and RMSE/MAE/MAPE.
    """
    data = _load_forecast(state)
    return ForecastResponse(
        state=data["state"],
        best_model=data["best_model"],
        forecast=[ForecastPoint(**p) for p in data["forecast"]],
        metrics=MetricsSchema(**data["metrics"]),
    )


@app.get("/models/{state}", tags=["Evaluation"])
def model_metrics(state: str):
    """
    Return evaluation metrics (RMSE, MAE, MAPE) for all models trained on a state.
    Shows the best model and the full comparison.
    """
    data = _load_forecast(state)
    return JSONResponse(content={
        "state": data["state"],
        "best_model": data["best_model"],
        "metrics": data["metrics"],
        "all_metrics": data.get("all_metrics", {}),
    })


@app.post("/retrain", response_model=RetrainResponse, tags=["System"])
def retrain(background_tasks: BackgroundTasks):
    """
    Trigger a full retraining of all models for all states.
    Training runs in the background — check server logs for progress.
    """
    background_tasks.add_task(_retrain_background)
    logger.info("[API] Retrain task queued.")
    return RetrainResponse(
        status="accepted",
        message="Retraining started in background. Monitor logs for progress.",
    )


# ── Dev server entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
