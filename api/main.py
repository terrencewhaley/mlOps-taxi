import sys
import os
import boto3
import tempfile
import hashlib
import json
import redis

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training'))

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from features import build_features
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

Instrumentator().instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_BUCKET = 'mlops-taxi-models-665012226357'
MODEL_KEY = 'models/model.joblib'
CACHE_TTL_SECONDS = 900  # 15 min — fare predictions don't need to be fresher than this

redis_client = redis.Redis(
    host=os.environ.get("REDIS_HOST", "localhost"),
    port=6379,
    decode_responses=True,
    socket_connect_timeout=2,  # fail fast rather than hang the request if Redis is unreachable
)

def load_model():
    s3 = boto3.client('s3', region_name='us-east-1')
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
        s3.download_fileobj(MODEL_BUCKET, MODEL_KEY, f)
        return joblib.load(f.name)

model = load_model()

def get_model_version():
    # S3's ETag changes whenever model.joblib's content changes, so this
    # doubles as a free "model version" tag with no manual bookkeeping —
    # redeploying a retrained model automatically invalidates old cache entries.
    s3 = boto3.client('s3', region_name='us-east-1')
    obj = s3.head_object(Bucket=MODEL_BUCKET, Key=MODEL_KEY)
    return obj['ETag']

MODEL_VERSION = get_model_version()

def build_cache_key(trip: "TripRequest") -> str:
    payload = {**trip.model_dump(), "model_version": MODEL_VERSION}
    raw = json.dumps(payload, sort_keys=True)
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return f"fare_pred:{digest}"

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
    cache_key = build_cache_key(trip)

    try:
        cached = redis_client.get(cache_key)
        if cached is not None:
            return {"predicted_fare": float(cached), "source": "cache"}
    except redis.exceptions.RedisError:
        pass  # Redis being down should never take /predict down with it

    df = pd.DataFrame([trip.model_dump()])
    features = build_features(df)
    prediction = round(float(model.predict(features)[0]), 2)

    try:
        redis_client.setex(cache_key, CACHE_TTL_SECONDS, prediction)
    except redis.exceptions.RedisError:
        pass

    return {"predicted_fare": prediction, "source": "model"}