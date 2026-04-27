"""FastAPI serving app with Prometheus metrics for MLOps monitoring."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

try:
    from src.monitor import load_monitor_state
except ImportError:  # pragma: no cover
    from monitor import load_monitor_state


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVING_MODEL_PATH = PROJECT_ROOT / "mlflow" / "serving_model"

app = FastAPI(title="Adult Income MLOps API", version="1.0.0")

REQUEST_COUNTER = Counter("api_requests_total", "Total API requests", ["endpoint", "status"])
PREDICT_LATENCY = Histogram("predict_latency_seconds", "Prediction endpoint latency")
DRIFT_SCORE_GAUGE = Gauge("drift_score", "Latest drift score by detector", ["detector"])
MODEL_ACCURACY_GAUGE = Gauge("model_accuracy", "Latest model accuracy")
ALERT_COUNT_GAUGE = Gauge("alert_count", "Total drift alerts observed")
DETECTOR_TIME_GAUGE = Gauge("detector_time_seconds", "Latest detector execution time", ["detector"])
RETRAIN_TRIGGER_GAUGE = Gauge(
    "retraining_trigger_time_seconds", "Retraining trigger time by detector", ["detector"]
)

_model: Any = None
_monitoring_task: asyncio.Task | None = None


def _load_model() -> Any:
    if not SERVING_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Serving model not found. Run `python src/train.py` or `python src/retrain.py` first."
        )
    return mlflow.pyfunc.load_model(str(SERVING_MODEL_PATH))


def _refresh_gauges_from_state() -> None:
    state = load_monitor_state()
    MODEL_ACCURACY_GAUGE.set(float(state.get("accuracy", 0.0)))
    ALERT_COUNT_GAUGE.set(float(state.get("alert_count", 0)))

    drift_scores = state.get("drift_scores", {})
    detector_times = state.get("detector_time_seconds", {})
    retrain_times = state.get("retraining_trigger_time_seconds", {})

    for detector in ["ks", "psi", "js"]:
        DRIFT_SCORE_GAUGE.labels(detector=detector).set(float(drift_scores.get(detector, 0.0)))
        DETECTOR_TIME_GAUGE.labels(detector=detector).set(float(detector_times.get(detector, 0.0)))
        RETRAIN_TRIGGER_GAUGE.labels(detector=detector).set(float(retrain_times.get(detector, 0.0)))


async def _monitor_state_loop() -> None:
    while True:
        _refresh_gauges_from_state()
        await asyncio.sleep(15)


@app.on_event("startup")
async def startup_event() -> None:
    global _model
    global _monitoring_task

    _model = _load_model()
    _refresh_gauges_from_state()
    _monitoring_task = asyncio.create_task(_monitor_state_loop())


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global _monitoring_task
    if _monitoring_task is not None:
        _monitoring_task.cancel()
        _monitoring_task = None


@app.get("/health")
def health() -> dict[str, str]:
    REQUEST_COUNTER.labels(endpoint="/health", status="ok").inc()
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: list[dict[str, Any]]) -> dict[str, Any]:
    start = time.perf_counter()
    if _model is None:
        REQUEST_COUNTER.labels(endpoint="/predict", status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not payload:
        REQUEST_COUNTER.labels(endpoint="/predict", status="error").inc()
        raise HTTPException(status_code=400, detail="Payload must be a non-empty list of records")

    try:
        df = pd.DataFrame(payload)
        predictions = _model.predict(df)
    except Exception as exc:  # noqa: BLE001
        REQUEST_COUNTER.labels(endpoint="/predict", status="error").inc()
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    elapsed = time.perf_counter() - start
    PREDICT_LATENCY.observe(elapsed)
    REQUEST_COUNTER.labels(endpoint="/predict", status="ok").inc()

    return {
        "predictions": [int(p) if str(p).isdigit() else str(p) for p in predictions],
        "model_path": str(SERVING_MODEL_PATH),
        "latency_seconds": elapsed,
    }


@app.get("/metrics")
def metrics() -> Response:
    REQUEST_COUNTER.labels(endpoint="/metrics", status="ok").inc()
    _refresh_gauges_from_state()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
