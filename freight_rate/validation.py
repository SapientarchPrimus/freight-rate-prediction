import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from .data import TARGET
from .metrics import regression_report
from .model import BASE_FEATURES, DEFAULT_PARAMS, GEO_FEATURES, HybridFreightModel

TEMPORAL_FOLDS = [
    ("F1 (test May-Jun)", "2025-04-30", "2025-05-01", "2025-06-30"),
    ("F2 (test Jul-Aug)", "2025-06-30", "2025-07-01", "2025-08-31"),
    ("F3 (test Sep-Oct)", "2025-08-31", "2025-09-01", "2025-10-31"),
]
PRIMARY_FOLD = TEMPORAL_FOLDS[-1][0]


class MedianRatePerMile:
    def fit(self, frame):
        self.rate_ = (frame[TARGET] / frame["distance"]).groupby(frame["equipment"]).median()
        return self

    def predict(self, frame):
        return (frame["distance"] * frame["equipment"].map(self.rate_)).to_numpy()


class PlainBoosting:
    def __init__(self, features):
        self.features = features

    def fit(self, frame):
        self.model_ = HistGradientBoostingRegressor(**DEFAULT_PARAMS)
        self.model_.fit(frame[self.features], np.log(frame[TARGET]))
        return self

    def predict(self, frame):
        return np.exp(self.model_.predict(frame[self.features]))


def final_model():
    return HybridFreightModel(drift="linear", market_beta="within")


def model_zoo():
    no_quote = [f for f in BASE_FEATURES if f != "quote_signal"]
    return {
        "Naive: median $/mile by equipment": lambda: MedianRatePerMile(),
        "Boosting, load features only": lambda: PlainBoosting(BASE_FEATURES + GEO_FEATURES),
        "Boosting + market_index + day index as tree features": lambda: PlainBoosting(
            BASE_FEATURES + GEO_FEATURES + ["market_filled", "t"]
        ),
        "Hybrid, pooled market coefficient + linear drift": lambda: HybridFreightModel(
            drift="linear", market_beta="pooled"
        ),
        "Hybrid, within-day market coefficient, no drift": lambda: HybridFreightModel(
            drift="none", market_beta="within"
        ),
        "Hybrid, within-day coefficient + capped drift": lambda: HybridFreightModel(
            drift="capped", market_beta="within"
        ),
        "FINAL: hybrid, within-day coefficient + linear drift": final_model,
        "Final without geography features": lambda: HybridFreightModel(features=BASE_FEATURES),
        "Final without quote_signal": lambda: HybridFreightModel(features=no_quote + GEO_FEATURES),
        "Final without outlier filtering": lambda: HybridFreightModel(outlier_floor=9.0),
        "Final with a single fitting round": lambda: HybridFreightModel(rounds=1),
    }


def temporal_split(frame, train_end, test_start, test_end):
    train = frame[frame["date"] <= train_end]
    test = frame[(frame["date"] >= test_start) & (frame["date"] <= test_end)]
    return train, test


def run_temporal(frame, zoo):
    rows = []
    for name, make in zoo.items():
        for fold, train_end, test_start, test_end in TEMPORAL_FOLDS:
            train, test = temporal_split(frame, train_end, test_start, test_end)
            model = make().fit(train)
            report = regression_report(test[TARGET], model.predict(test))
            report.update(model=name, fold=fold)
            rows.append(report)
    return pd.DataFrame(rows)


def run_random_kfold(frame, splits=5, seed=0):
    candidates = {
        "Final model, random K-fold": final_model,
        "Boosting + market_index + day index, random K-fold": lambda: PlainBoosting(
            BASE_FEATURES + GEO_FEATURES + ["market_filled", "t"]
        ),
    }
    rows = []
    for name, make in candidates.items():
        for number, (train_idx, test_idx) in enumerate(KFold(splits, shuffle=True, random_state=seed).split(frame), 1):
            train, test = frame.iloc[train_idx], frame.iloc[test_idx]
            model = make().fit(train)
            report = regression_report(test[TARGET], model.predict(test))
            report.update(model=name, fold=f"K{number}")
            rows.append(report)
    return pd.DataFrame(rows)


def run_unseen_cities(frame, seeds=(1, 2, 3), held_out=8):
    train_end, test_start, test_end = TEMPORAL_FOLDS[-1][1:]
    cities = np.array(sorted(set(frame["pickup"]) | set(frame["delivery"])))
    variants = {
        "With coordinate features": lambda: HybridFreightModel(features=BASE_FEATURES + GEO_FEATURES),
        "Without geography features": lambda: HybridFreightModel(features=BASE_FEATURES),
    }
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        held = set(rng.choice(cities, held_out, replace=False))
        train, test = temporal_split(frame, train_end, test_start, test_end)
        train = train[~(train["pickup"].isin(held) | train["delivery"].isin(held))]
        touches = test["pickup"].isin(held) | test["delivery"].isin(held)
        for name, make in variants.items():
            model = make().fit(train)
            for label, part in (("loads touching unseen cities", test[touches]), ("loads on seen cities", test[~touches])):
                report = regression_report(part[TARGET], model.predict(part))
                report.update(model=name, seed=seed, subset=label)
                rows.append(report)
    return pd.DataFrame(rows)
