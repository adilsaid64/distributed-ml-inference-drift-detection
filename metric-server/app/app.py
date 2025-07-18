from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import generate_latest, Counter, Gauge, Histogram
from sklearn.datasets import load_breast_cancer
from evidently.report import Report
from evidently.metrics import DataDriftPreset
from contextlib import asynccontextmanager
from typing import Literal
import pandas as pd
import numpy as np
import time
from collections import deque

# --- Prometheus Metrics ---
DRIFTED_WINDOWS = Counter("drifted_windows_total", "Number of windows where dataset drift was detected")
DRIFT_SHARE = Gauge("drift_share", "Drift share for the most recent window")
DRIFT_LATENCY = Histogram("drift_check_latency_seconds", "Latency for drift check")

FEATURE_DRIFT_DETECTED = Gauge(
    "feature_drift_detected",
    "Whether drift was detected for a feature (1=yes, 0=no)",
    labelnames=["feature"]
)

FEATURE_DRIFT_SCORE = Gauge(
    "feature_drift_score",
    "Drift score for a feature (e.g., JS distance or similar)",
    labelnames=["feature"]
)

FEATURE_DRIFT_PVALUE = Gauge(
    "feature_drift_pvalue",
    "P-value for a feature drift test",
    labelnames=["feature"]
)

class FeaturePayload(BaseModel):
    columns: list[str]
    data: list[list[float]]
    
class DataDriftRequest(BaseModel):
    features: FeaturePayload

class DataDriftResponse(BaseModel):
    status: Literal["accumulating", "drift_checked"]
    drift_share: float
    samples_in_window: int

# --- Drift Monitor ---
class DriftMonitor:
    def __init__(self, reference_df: pd.DataFrame, window_size: int = 100):
        self.reference_df = reference_df
        self.window_size = window_size
        self.buffer: deque[pd.DataFrame] = deque(maxlen=window_size)

    def add_sample(self, row_df: pd.DataFrame) -> tuple[bool, float]:
        self.buffer.append(row_df)

        if len(self.buffer) < self.window_size:
            return False, 0.0

        current_df = pd.concat(self.buffer, ignore_index=True)

        start = time.time()
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=self.reference_df, current_data=current_df)
        elapsed = time.time() - start
        DRIFT_LATENCY.observe(elapsed)

        result = report.as_dict()
        dataset_drift = result["metrics"][0]["result"]["dataset_drift"]
        drift_share = result["metrics"][0]["result"]["drift_share"]
        feature_results = result["metrics"][0]["result"]["drift_by_columns"]

        DRIFT_SHARE.set(drift_share)
        if dataset_drift:
            DRIFTED_WINDOWS.inc()

        for feature, data in feature_results.items():
            FEATURE_DRIFT_DETECTED.labels(feature=feature).set(float(data["drift_detected"]))
            FEATURE_DRIFT_SCORE.labels(feature=feature).set(data.get("drift_score", 0.0))
            FEATURE_DRIFT_PVALUE.labels(feature=feature).set(data.get("p_value", 1.0))

        return True, drift_share

# --- App Initialization ---
monitor: DriftMonitor

@asynccontextmanager
async def lifespan(app: FastAPI):
    global monitor
    dataset = load_breast_cancer()
    reference_df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    monitor = DriftMonitor(reference_df=reference_df, window_size=100)
    yield

app = FastAPI(
    title="Drift Monitoring Server",
    description="Monitors Feature Drift with Evidently",
    lifespan=lifespan
)

@app.post("/datadrift", response_model=DataDriftResponse)
async def datadrift(payload: DataDriftRequest) -> DataDriftResponse:

    feature_dict = payload.features
    X = pd.DataFrame(data=feature_dict.data, columns=feature_dict.columns)
    ready, drift_share = monitor.add_sample(X)

    return DataDriftResponse(
        status="drift_checked" if ready else "accumulating",
        drift_share=drift_share,
        samples_in_window=len(monitor.buffer)
    )

@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type="text/plain")
