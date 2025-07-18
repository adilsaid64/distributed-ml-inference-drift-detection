from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from contextlib import asynccontextmanager
import numpy as np
import httpx
import logging
import pandas as pd
logger = logging.getLogger("uvicorn")

REQUEST_COUNT: Counter = Counter("prediction_requests_total", "Total number of prediction requests")
PREDICT_LATENCY: Histogram = Histogram("prediction_latency_seconds", "Prediction latency in seconds")

METRIC_SERVER_URL: str = "http://metric-server/datadrift"

model: RandomForestClassifier | None = None


class GetPredictionRequest(BaseModel):
    """Get Prediction Payload"""
    features: list[list[float]] = Field(description="List of 30 numeric feature values")
    feature_names: list[str] = Field(description="List of 30 feature names")

class GetPredictionResponse(BaseModel):
    """Get Prediction Response"""
    prediction: list[int]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle handler"""
    global model
    logger.info("Loading Data and Training Model")
    X, y = load_breast_cancer(as_frame=True, return_X_y=True)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    logger.info("Model Trianed and Set")
    yield

app = FastAPI(title="Model Server", description="Breast Cancer Prediction API", lifespan=lifespan)

async def send_to_metric_monitoring(payload: dict[str, list[float] | list[str]]) -> None:
    """Send features to external metric monitoring service"""
    
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(METRIC_SERVER_URL, json=payload)
    except httpx.RequestError:
        pass

@app.post("/get-prediction", response_model=GetPredictionResponse)
@PREDICT_LATENCY.time()
async def get_prediction(request: GetPredictionRequest, background_tasks: BackgroundTasks) -> GetPredictionResponse:
    """Return prediction for input features"""
    REQUEST_COUNT.inc()

    background_tasks.add_task(send_to_metric_monitoring, {"features": request.features, "feature_names": request.feature_names})
    
    X: pd.DataFrame = pd.DataFrame(request.features, columns=request.feature_names)

    prediction = model.predict(X)

    return GetPredictionResponse(prediction=prediction.tolist())

@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics"""
    return Response(generate_latest(), media_type="text/plain")
