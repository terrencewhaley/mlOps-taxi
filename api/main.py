import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training'))

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from features import build_features

app = FastAPI()

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'd17020d3-286b-47c9-b896-b8794b4c318c', 'model.joblib'))
model = joblib.load(MODEL_PATH)

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