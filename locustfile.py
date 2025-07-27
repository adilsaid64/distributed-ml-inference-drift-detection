import numpy as np
from sklearn.datasets import load_breast_cancer
from locust import HttpUser, task, between

MODEL_SERVER_URL = "http://localhost:8002/get-prediction"
REQUEST_INTERVAL = 1.0

X, y = load_breast_cancer(as_frame=True, return_X_y=True)

class ClientTestUser(HttpUser):

    @task
    def get_prediction(self):
        sample = X.sample(1)

        if np.random.random() < 0.1:
            noise_std = 100
            sample += noise_std * np.random.randn(*sample.shape)

        payload = {
            "features": {
                "columns": sample.columns.tolist(),
                "data": sample.values.tolist()
            }
        }
        self.client.post(MODEL_SERVER_URL, json=payload)
