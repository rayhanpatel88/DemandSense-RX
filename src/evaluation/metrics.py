"""Forecasting evaluation metrics."""

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + eps))) * 100)


def wape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Weighted Absolute Percentage Error (more robust than MAPE for zero-heavy series)."""
    total = np.sum(np.abs(y_true))
    if total < eps:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / total * 100)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": round(mae(y_true, y_pred), 3),
        "RMSE": round(rmse(y_true, y_pred), 3),
        "MAPE": round(mape(y_true, y_pred), 3),
        "WAPE": round(wape(y_true, y_pred), 3),
    }


def compute_metrics_by_sku(df: pd.DataFrame, pred_col: str = "forecast") -> pd.DataFrame:
    """Compute metrics per SKU."""
    rows = []
    for sku, group in df.groupby("sku"):
        m = compute_all_metrics(group["demand"].values, group[pred_col].values)
        m["sku"] = sku
        rows.append(m)
    return pd.DataFrame(rows).set_index("sku")
