from pathlib import Path

import numpy as np
import pandas as pd

WEIGHT_CAP = 47_500.0
EPOCH = pd.Timestamp("2025-01-01")
EQUIPMENT_TYPES = ["Dry Van", "Flatbed", "Reefer"]
TARGET = "posted_rate"


def read_loads(path):
    frame = pd.read_csv(Path(path))
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def city_table(*frames):
    parts = []
    for frame in frames:
        parts.append(frame[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1))
        parts.append(frame[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).drop_duplicates("city").set_index("city")


def daily_market_index(frame):
    daily = frame.groupby("date")["market_index"].mean()
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(full).interpolate(limit_direction="both")


def engineer(frame, market_by_date=None):
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])

    out["weight_missing"] = out["weight"].isna().astype(int)
    out["weight_abs"] = out["weight"].abs()
    out["weight_capped"] = (out["weight_abs"] >= WEIGHT_CAP).astype(int)

    if market_by_date is None:
        market_by_date = daily_market_index(out)
    out["market_day"] = out["date"].map(market_by_date)
    out["market_filled"] = out["market_index"].fillna(out["market_day"])

    out["t"] = (out["date"] - EPOCH).dt.days.astype(float)
    out["dow"] = out["date"].dt.dayofweek

    out["log_distance"] = np.log(out["distance"])
    for name in EQUIPMENT_TYPES:
        out["eq_" + name.replace(" ", "_").lower()] = (out["equipment"] == name).astype(int)

    out["haversine"] = haversine_miles(out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"])
    out["road_ratio"] = out["distance"] / out["haversine"].clip(lower=1.0)
    out["dlat"] = out["delivery_lat"] - out["pickup_lat"]
    out["dlon"] = out["delivery_lon"] - out["pickup_lon"]
    out["mid_lat"] = (out["pickup_lat"] + out["delivery_lat"]) / 2
    out["mid_lon"] = (out["pickup_lon"] + out["delivery_lon"]) / 2
    return out
