import math
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Distance in kilometers via Haversine formula.
    Accepts pandas Series.
    """
    r = 6371.0

    lat1 = lat1.astype(float) * math.pi / 180.0
    lon1 = lon1.astype(float) * math.pi / 180.0
    lat2 = lat2.astype(float) * math.pi / 180.0
    lon2 = lon2.astype(float) * math.pi / 180.0

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (dlat / 2.0).apply(math.sin) ** 2 + lat1.apply(math.cos) * lat2.apply(math.cos) * (dlon / 2.0).apply(math.sin) ** 2
    c = a.apply(lambda x: 2.0 * math.asin(min(1.0, math.sqrt(x))))
    return r * c


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build model-ready features from the Kaggle dataset columns:
      - tpep_pickup_datetime
      - Passenger_count
      - Trip_distance
      - Pickup_longitude, Pickup_latitude
      - Dropoff_longitude, Dropoff_latitude
    """
    dt = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce", utc=True)

    out = pd.DataFrame(index=df.index)

    out["passenger_count"] = (
        pd.to_numeric(df["Passenger_count"], errors="coerce")
        .fillna(1)
        .clip(1, 6)
    )

    # As reported by the meter (miles)
    out["trip_distance_miles"] = (
        pd.to_numeric(df["Trip_distance"], errors="coerce")
        .fillna(0.0)
        .clip(0.0, 100.0)
    )

    # Geo distance (km) as a robustness feature
    out["geo_distance_km"] = (
        haversine_km(
            df["Pickup_latitude"], df["Pickup_longitude"],
            df["Dropoff_latitude"], df["Dropoff_longitude"],
        )
        .fillna(0.0)
        .clip(0.0, 200.0)
    )

    out["pickup_hour"] = dt.dt.hour
    out["pickup_dayofweek"] = dt.dt.dayofweek
    out["pickup_month"] = dt.dt.month

    out["pickup_in_nyc_bbox"] = (
        df["Pickup_latitude"].between(40.4, 41.0)
        & df["Pickup_longitude"].between(-74.3, -73.6)
    ).astype(int)

    out["dropoff_in_nyc_bbox"] = (
        df["Dropoff_latitude"].between(40.4, 41.0)
        & df["Dropoff_longitude"].between(-74.3, -73.6)
    ).astype(int)

    return out
