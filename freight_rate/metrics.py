import numpy as np


def regression_report(y_true, y_pred, outlier_log_gap=0.4):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    pct = np.abs(error) / y_true
    clean = np.abs(np.log(y_true / y_pred)) < outlier_log_gap
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape_pct": float(100 * pct.mean()),
        "median_ape_pct": float(100 * np.median(pct)),
        "r2": 1 - ss_res / ss_tot,
        "outlier_share_pct": float(100 * (~clean).mean()),
        "clean_mae": float(np.mean(np.abs(error[clean]))),
        "clean_mape_pct": float(100 * pct[clean].mean()),
    }
