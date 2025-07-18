import requests
from sklearn.datasets import load_breast_cancer
import pandas as pd
import time

MODEL_SERVER_URL = "http://localhost:8001/get-prediction"
REQUEST_INTERVAL = 1.0

X, y = load_breast_cancer(as_frame=True, return_X_y=True)

def stream_predictions():
    """Continuously send prediction requests to the model server"""
    request_count = 1
    while True:
        sample = X.sample(1)

        payload = {
            "features": {
                "columns": sample.columns.tolist(),
                "data": sample.values.tolist()
            }
        }
        try:
            response = requests.post(MODEL_SERVER_URL, json=payload)
            if response.ok:
                prediction = response.json()["prediction"]
                print(f"[{request_count}] Prediction: {prediction}")
            else:
                print(f"[{request_count}] Error {response.status_code}: {response.text}")
        except requests.RequestException as e:
            print(f"[{request_count}] Request failed: {e}")

        request_count += 1
        time.sleep(REQUEST_INTERVAL)

if __name__ == "__main__":
    stream_predictions()
