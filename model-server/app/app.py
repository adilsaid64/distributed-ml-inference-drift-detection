import time
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from contextlib import asynccontextmanager
import httpx
import logging
from fastapi.responses import Response
import pandas as pd
from prometheus_client import generate_latest, Counter, Histogram


PREDICTION_REQUESTS = Counter("prediction_requests_total", "Total number of prediction requests")
PREDICTION_ERRORS = Counter("prediction_errors_total", "Total number of failed prediction requests")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency in seconds")


logger = logging.getLogger("uvicorn")

METRIC_SERVER_URL: str = "http://load-balancer/metric/datadrift"

MODEL: RandomForestClassifier | None = None

class FeaturePayload(BaseModel):
    columns: list[str]
    data: list[list[float]]

class GetPredictionRequest(BaseModel):
    """Get Prediction Payload"""
    features: FeaturePayload


class GetPredictionResponse(BaseModel):
    """Get Prediction Response"""
    prediction: list[int]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle handler"""
    global MODEL
    logger.info("Loading Data and Training Model")
    X, y = load_breast_cancer(as_frame=True, return_X_y=True)
    MODEL = RandomForestClassifier(n_estimators=100, random_state=42)
    MODEL.fit(X, y)
    logger.info("Model Trianed and Set")
    yield

app = FastAPI(title="Model Server", description="Breast Cancer Prediction API", lifespan=lifespan)

async def send_to_metric_monitoring(payload: dict[str, dict[str, list[float] | list[str]]]) -> None:
    """Send features to external metric monitoring service"""
    
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(METRIC_SERVER_URL, json=payload)
    except httpx.RequestError:
        pass

@app.post("/get-prediction")
async def get_prediction(payload: GetPredictionRequest, background_tasks: BackgroundTasks) -> GetPredictionResponse:
    """Return prediction for input features"""
    start_time = time.time()
    PREDICTION_REQUESTS.inc()

    try:
        feature_dict = payload.features
        X = pd.DataFrame(data=feature_dict.data, columns=feature_dict.columns)
        background_tasks.add_task(send_to_metric_monitoring, {"features":  {"columns": X.columns.tolist(), "data": X.values.tolist()}})
        prediction = MODEL.predict(X)
        return GetPredictionResponse(prediction=prediction.tolist())
    
    except Exception as e:
        PREDICTION_ERRORS.inc()
        raise e

    finally:
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")