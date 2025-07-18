from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import generate_latest, Counter, Gauge, Histogram
from sklearn.datasets import load_breast_cancer
from contextlib import asynccontextmanager
from typing import Literal
from scipy.stats import ks_2samp
import pandas as pd
import time
from collections import deque

DRIFTED_WINDOWS = Counter("drifted_windows_total", "Number of windows where dataset drift was detected")
DRIFT_SHARE = Gauge("drift_share", "Share of features with drift in the last window")
DRIFT_LATENCY = Histogram("drift_check_latency_seconds", "Latency for drift check")
FEATURE_PVALUE = Gauge(
    "feature_drift_pvalue",
    "P-value from KS test per feature",
    labelnames=["feature"]
)

FEATURE_DRIFTED = Gauge(
    "feature_drift_detected",
    "Whether drift was detected for a feature (1=yes, 0=no)",
    labelnames=["feature"]
)

FEATURE_MAGNITUDE = Gauge(
    "feature_drift_magnitude",
    "Estimated magnitude of feature drift",
    labelnames=["feature"]
)


FEATURE_DRIFT_EVENTS = Counter(
    "feature_drift_events_total",
    "Cumulative count of drift detections per feature",
    labelnames=["feature"]
)

DRIFTED_WINDOWS_LAST_TIMESTAMP = Gauge(
    "drifted_windows_last_timestamp",
    "Unix timestamp when the last drifted window was detected"
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
    feature_pvalues: dict[str, float] | None = None

class KSTestDriftDetector:
    def __init__(self, reference_df: pd.DataFrame, window_size: int, p_threshold: float):
        self.reference_df = reference_df
        self.window_size = window_size
        self.p_threshold = p_threshold
        self.buffer: deque[pd.DataFrame] = deque(maxlen=window_size)

    def add_sample(self, row_df: pd.DataFrame) -> tuple[bool, float, dict[str, float]]:
        self.buffer.append(row_df)

        if len(self.buffer) < self.window_size:
            return False, 0.0, {}

        current_df = pd.concat(self.buffer, ignore_index=True)

        start = time.time()

        drifted = 0
        feature_pvalues: dict[str, float] = {}

        for col in self.reference_df.columns:
            ref_vals = self.reference_df[col].values
            curr_vals = current_df[col].values

            mean_diff = abs(ref_vals.mean() - curr_vals.mean()) 
            FEATURE_MAGNITUDE.labels(feature=col).set(mean_diff)

            _, p_value = ks_2samp(ref_vals, curr_vals)
            feature_pvalues[col] = p_value
            
            FEATURE_PVALUE.labels(feature=col).set(p_value)
            drift_flag = 1 if p_value < self.p_threshold else 0
            FEATURE_DRIFTED.labels(feature=col).set(drift_flag)

            if drift_flag:
                drifted += 1
                FEATURE_DRIFT_EVENTS.labels(feature=col).inc()

        drift_share = drifted / len(self.reference_df.columns)
        elapsed = time.time() - start

        DRIFT_LATENCY.observe(elapsed)
        DRIFT_SHARE.set(drift_share)
        if drift_share > 0:
            DRIFTED_WINDOWS.inc()
            DRIFTED_WINDOWS_LAST_TIMESTAMP.set(time.time())

        return True, drift_share, feature_pvalues

monitor: KSTestDriftDetector | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global monitor
    dataset = load_breast_cancer()
    reference_df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    monitor = KSTestDriftDetector(reference_df=reference_df, window_size=10, p_threshold=0.05)
    yield

app = FastAPI(
    title="KS-Test Drift Monitor",
    description="Performs simple dataset drift detection using KS test per feature",
    lifespan=lifespan
)

@app.post("/datadrift", response_model=DataDriftResponse)
async def datadrift(payload: DataDriftRequest) -> DataDriftResponse:
    feature_dict = payload.features
    X = pd.DataFrame(data=feature_dict.data, columns=feature_dict.columns)
    ready, drift_share, pvalues = monitor.add_sample(X)

    return DataDriftResponse(
        status="drift_checked" if ready else "accumulating",
        drift_share=drift_share,
        samples_in_window=len(monitor.buffer),
        feature_pvalues=pvalues if ready else None
    )
@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type="text/plain")
