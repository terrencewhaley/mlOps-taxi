import sys
import os
import boto3
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training'))

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from features import build_features
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

Instrumentator().instrument(app).expose(app)

def load_model():
    s3 = boto3.client('s3', region_name='us-east-1')
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
        s3.download_fileobj('mlops-taxi-models-665012226357', 'models/model.joblib', f)
        return joblib.load(f.name)

model = load_model()

class TripRequest(BaseModel):
    tpep_pickup_datetime: str
    Pickup_longitude: float
    Pickup_latitude: float
    Dropoff_longitude: float
    Dropoff_latitude: float
    Passenger_count: int
    Trip_distance: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(trip: TripRequest):
    df = pd.DataFrame([trip.model_dump()])
    features = build_features(df)
    prediction = model.predict(features)
    return {"predicted_fare": round(float(prediction[0]), 2)}
