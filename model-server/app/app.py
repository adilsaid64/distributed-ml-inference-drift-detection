from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, Summary, generate_latest
from pydantic import BaseModel

app = FastAPI(title="Model-Server", description="Model Server Endpoint")


class GetPredictionRequest(BaseModel):
    """Get Prediction Request Schema"""
    ...


class GetPredictionResponse(BaseModel):
    """Get Prediction Response Schema"""
    ...


def send_data_to_metric_server(data: ...) -> None:
   """Sends data to the metric server"""
   ...

@app.post("/get-prediction/")
async def get_prediction(request: GetPredictionRequest, background_tasks: BackgroundTasks) -> GetPredictionResponse:
    """Returns back a prediction"""
    
    # send data as a background task to metric server 
    background_tasks.add_task(send_data_to_metric_server, ...)

    # do predictions
    model = ...
    pred = ...

    return pred



@app.get("/metrics")
def metrics():
    """..."""
    return Response(generate_latest(), media_type="text/plain")