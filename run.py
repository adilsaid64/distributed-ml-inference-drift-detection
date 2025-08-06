import requests
import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
import time
import click

MODEL_SERVER_URL = "http://localhost:8002/model/get-prediction"
REQUEST_INTERVAL = 1.0

X, y = load_breast_cancer(as_frame=True, return_X_y=True)


@click.command()
@click.option('--drift', help='Boolean Toggle To Simulate Drift', type = bool)
def stream_predictions(drift: bool):
    """Continuously send prediction requests to the model server"""

    print("DRIFT SETTING: ", drift)
    request_count = 1
    while True:
        sample = X.sample(1)

        if drift:
            noise_std = 100
            sample += noise_std * np.random.randn(*sample.shape)

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
