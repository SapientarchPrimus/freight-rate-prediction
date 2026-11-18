import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .data import TARGET

BASE_FEATURES = [
    "log_distance",
    "eq_dry_van",
    "eq_flatbed",
    "eq_reefer",
    "weight_abs",
    "weight_capped",
    "quote_signal",
]
GEO_FEATURES = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "mid_lat",
    "mid_lon",
    "dlat",
    "dlon",
    "road_ratio",
]
DEFAULT_PARAMS = dict(
    loss="squared_error",
    learning_rate=0.05,
    max_iter=500,
    max_leaf_nodes=31,
    min_samples_leaf=40,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=0,
)


class HybridFreightModel:
    def __init__(
        self,
        features=None,
        drift="linear",
        market_beta="within",
        use_market=True,
        rounds=3,
        outlier_floor=0.15,
        outlier_sigmas=5.0,
        gbm_params=None,
    ):
        self.features = list(features) if features is not None else BASE_FEATURES + GEO_FEATURES
        self.drift = drift
        self.market_beta = market_beta
        self.use_market = use_market
        self.rounds = rounds
        self.outlier_floor = outlier_floor
        self.outlier_sigmas = outlier_sigmas
        self.gbm_params = {**DEFAULT_PARAMS, **(gbm_params or {})}

    def _day_columns(self):
        columns = []
        if self.use_market:
            columns.append("market_filled")
        if self.drift != "none":
            columns.append("t")
        return columns

    def _day_matrix(self, frame, for_predict=False):
        columns = []
        for name in self._day_columns():
            values = frame[name].to_numpy(dtype=float)
            if name == "t" and for_predict and self.drift == "capped":
                values = np.minimum(values, self.t_max_)
            columns.append(values)
        if not columns:
            return np.zeros((len(frame), 0))
        return np.column_stack(columns)

    def fit(self, frame):
        y = np.log(frame[TARGET].to_numpy(dtype=float))
        self.t_max_ = float(frame["t"].max())
        day = self._day_matrix(frame)
        coef = np.zeros(day.shape[1])
        intercept = 0.0
        keep = np.ones(len(frame), dtype=bool)

        for _ in range(self.rounds):
            offset = intercept + day @ coef
            self.gbm_ = HistGradientBoostingRegressor(**self.gbm_params)
            self.gbm_.fit(frame.loc[keep, self.features], (y - offset)[keep])
            load_part = self.gbm_.predict(frame[self.features])
            target = y - load_part
            design = np.column_stack([np.ones(len(frame)), day])
            beta = self._solve_day_terms(frame, design, target, keep)
            intercept, coef = beta[0], beta[1:]
            residual = y - (load_part + design @ beta)
            center = np.median(residual)
            sigma = 1.4826 * np.median(np.abs(residual - center))
            threshold = max(self.outlier_floor, self.outlier_sigmas * sigma)
            keep = np.abs(residual - center) < threshold

        self.intercept_ = float(intercept)
        self.coef_ = coef
        self.keep_ = keep
        self.sigma_ = float(sigma)
        return self

    def _within_day_beta(self, frame, target, keep):
        usable = keep & frame["market_index"].notna().to_numpy()
        sub = pd.DataFrame(
            {
                "date": frame["date"].to_numpy()[usable],
                "mi": frame["market_index"].to_numpy()[usable],
                "y": target[usable],
            }
        )
        mi_dev = sub["mi"] - sub.groupby("date")["mi"].transform("mean")
        y_dev = sub["y"] - sub.groupby("date")["y"].transform("mean")
        return float((mi_dev * y_dev).sum() / (mi_dev**2).sum())

    def _solve_day_terms(self, frame, design, target, keep):
        if not (self.use_market and self.market_beta == "within"):
            return np.linalg.lstsq(design[keep], target[keep], rcond=None)[0]
        slope = self._within_day_beta(frame, target, keep)
        remainder = target - slope * frame["market_filled"].to_numpy(dtype=float)
        columns = [design[:, 0]] + [design[:, j] for j in range(2, design.shape[1])]
        reduced = np.column_stack(columns)
        solved = np.linalg.lstsq(reduced[keep], remainder[keep], rcond=None)[0]
        return np.concatenate([[solved[0], slope], solved[1:]])

    def predict_log(self, frame):
        day = self._day_matrix(frame, for_predict=True)
        return self.gbm_.predict(frame[self.features]) + self.intercept_ + day @ self.coef_

    def predict(self, frame):
        return np.exp(self.predict_log(frame))

    def day_coefficients(self):
        return dict(zip(self._day_columns(), self.coef_.tolist()))
