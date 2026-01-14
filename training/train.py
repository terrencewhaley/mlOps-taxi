import argparse
import json
import os
import uuid
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from features import build_features

COLUMN_MAP = {
    "pickup_longitude": "Pickup_longitude",
    "pickup_latitude": "Pickup_latitude",
    "dropoff_longitude": "Dropoff_longitude",
    "dropoff_latitude": "Dropoff_latitude",
    "passenger_count": "Passenger_count",
    "trip_distance": "Trip_distance",
    "fare_amount": "Fare_amount",
}



# Required raw columns (exact casing from your dataset)
RAW_REQUIRED = [
    "tpep_pickup_datetime",
    "Pickup_longitude",
    "Pickup_latitude",
    "Dropoff_longitude",
    "Dropoff_latitude",
    "Passenger_count",
    "Trip_distance",
    "Fare_amount",  # target
]


FEATURE_SCHEMA = {
    "type": "object",
    "required": [
        "tpep_pickup_datetime",
        "Pickup_longitude",
        "Pickup_latitude",
        "Dropoff_longitude",
        "Dropoff_latitude",
        "Passenger_count",
        "Trip_distance",
    ],
    "properties": {
        "tpep_pickup_datetime": {"type": "string", "description": "ISO-8601 timestamp"},
        "Pickup_longitude": {"type": "number"},
        "Pickup_latitude": {"type": "number"},
        "Dropoff_longitude": {"type": "number"},
        "Dropoff_latitude": {"type": "number"},
        "Passenger_count": {"type": "integer", "minimum": 1, "maximum": 6},
        "Trip_distance": {"type": "number", "minimum": 0},
    },
}


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=RAW_REQUIRED).copy()

    numeric_cols = [
        "Pickup_longitude",
        "Pickup_latitude",
        "Dropoff_longitude",
        "Dropoff_latitude",
        "Passenger_count",
        "Trip_distance",
        "Fare_amount",
    ]

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=numeric_cols)

    # Stability filters (keeps training sane & reproducible)
    df = df[df["Passenger_count"].between(1, 6)]
    df = df[df["Trip_distance"].between(0.0, 100.0)]
    df = df[df["Fare_amount"].between(2.5, 200.0)]

    # Loose geo bounds (avoid garbage coordinates)
    df = df[df["Pickup_latitude"].between(35.0, 45.0)]
    df = df[df["Dropoff_latitude"].between(35.0, 45.0)]
    df = df[df["Pickup_longitude"].between(-80.0, -70.0)]
    df = df[df["Dropoff_longitude"].between(-80.0, -70.0)]

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV from Kaggle dataset")
    parser.add_argument("--out", default="../artifacts", help="Artifacts output folder")
    parser.add_argument("--run-id", default=None, help="Optional UUID. Auto-generated if omitted.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_id = args.run_id or str(uuid.uuid4())
    out_dir = os.path.join(args.out, run_id)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(args.data)
    df = df.rename(columns=COLUMN_MAP)


    missing = [c for c in RAW_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    df = clean_df(df)

    X_raw = df.drop(columns=["Fare_amount"])
    y = df["Fare_amount"].astype(float)

    X = build_features(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    cat_cols = ["pickup_hour", "pickup_dayofweek", "pickup_month"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.1,
        max_iter=200,
        random_state=args.seed,
    )

    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    # Artifacts
    joblib.dump(pipe, os.path.join(out_dir, "model.joblib"))

    metrics = {
        "run_id": run_id,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "rows_total": int(len(df)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(out_dir, "feature_schema.json"), "w") as f:
        json.dump(FEATURE_SCHEMA, f, indent=2)

    run_config = {
        "run_id": run_id,
        "data_file": os.path.basename(args.data),
        "target": "Fare_amount",
        "test_size": args.test_size,
        "seed": args.seed,
        "model": "HistGradientBoostingRegressor",
        "features_module": "training/features.py",
        "notes": "Step 1 baseline local training; intended to be reused inside SageMaker training job.",
    }
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"✅ Run complete: {run_id}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
