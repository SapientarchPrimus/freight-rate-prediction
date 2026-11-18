import numpy as np
import pandas as pd

from .data import city_table, daily_market_index, engineer


def december_frame(chart_inputs, cities, validation, quote_signal):
    frame = chart_inputs.drop(columns=["predicted_rate"]).copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["pickup_lat"] = frame["pickup"].map(cities["lat"])
    frame["pickup_lon"] = frame["pickup"].map(cities["lon"])
    frame["delivery_lat"] = frame["delivery"].map(cities["lat"])
    frame["delivery_lon"] = frame["delivery"].map(cities["lon"])
    frame["market_index"] = np.nan
    frame["quote_signal"] = quote_signal
    return engineer(frame, market_by_date=daily_market_index(validation))
