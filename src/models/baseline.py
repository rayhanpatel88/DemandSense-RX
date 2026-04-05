"""Baseline forecasting models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MovingAverageForecaster(BaseForecaster):
    """Rolling moving-average baseline with recursive multi-step support."""

    def __init__(self, window: int = 7):
        self.window = int(window)
        self._history: dict[str, list[float]] = {}
        self._sku_stds: dict[str, float] = {}

    def fit(self, df: pd.DataFrame, feature_cols: list, target_col: str = "demand"):
        self._history = {}
        self._sku_stds = {}
        for sku, group in df.groupby("sku"):
            values = group.sort_values("date")[target_col].astype(float).tolist()
            self._history[sku] = values
            tail = values[-self.window:] if len(values) >= self.window else values
            self._sku_stds[sku] = float(np.std(tail)) if tail else 0.0
        return self

    def predict(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        return np.clip(df["sku"].map(self._point_prediction).fillna(0.0).values, 0, None)

    def predict_with_intervals(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        point = self.predict(df, feature_cols)
        std = df["sku"].map(self._sku_stds).fillna(0.0).values
        lower = np.clip(point - 1.28 * std, 0, None)
        upper = point + 1.28 * std
        return pd.DataFrame({"forecast": point, "lower": lower, "upper": upper}, index=df.index)

    def recursive_forecast(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        histories = {sku: values.copy() for sku, values in self._history.items()}
        rows: list[dict] = []
        for _, row in future_df.sort_values(["sku", "date"]).iterrows():
            sku = row["sku"]
            history = histories.setdefault(sku, [])
            window_values = history[-self.window:] if history else [0.0]
            forecast = float(np.mean(window_values)) if window_values else 0.0
            std = float(np.std(window_values)) if len(window_values) > 1 else self._sku_stds.get(sku, 0.0)
            rows.append(
                {
                    "date": row["date"],
                    "sku": sku,
                    "category": row.get("category", "Unknown"),
                    "family": row.get("family", "Unknown"),
                    "promotion": row.get("promotion", 0),
                    "price": row.get("price", 0.0),
                    "forecast": forecast,
                    "lower": max(forecast - 1.28 * std, 0.0),
                    "upper": forecast + 1.28 * std,
                }
            )
            history.append(forecast)
        return pd.DataFrame(rows)

    def _point_prediction(self, sku: str) -> float:
        history = self._history.get(sku, [])
        tail = history[-self.window:] if history else [0.0]
        return float(np.mean(tail)) if tail else 0.0

    @property
    def name(self) -> str:
        return "MovingAverage"


class SeasonalNaiveForecaster(BaseForecaster):
    """Seasonal naive baseline with recursive multi-step support."""

    def __init__(self, season: int = 7):
        self.season = int(season)
        self._history: dict[str, list[float]] = {}

    def fit(self, df: pd.DataFrame, feature_cols: list, target_col: str = "demand"):
        self._history = {
            sku: group.sort_values("date")[target_col].astype(float).tolist()
            for sku, group in df.groupby("sku")
        }
        return self

    def predict(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        return np.clip(df["sku"].map(self._point_prediction).fillna(0.0).values, 0, None)

    def recursive_forecast(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        histories = {sku: values.copy() for sku, values in self._history.items()}
        rows: list[dict] = []
        for _, row in future_df.sort_values(["sku", "date"]).iterrows():
            sku = row["sku"]
            history = histories.setdefault(sku, [])
            forecast = history[-self.season] if len(history) >= self.season else (float(np.mean(history)) if history else 0.0)
            rows.append(
                {
                    "date": row["date"],
                    "sku": sku,
                    "category": row.get("category", "Unknown"),
                    "family": row.get("family", "Unknown"),
                    "promotion": row.get("promotion", 0),
                    "price": row.get("price", 0.0),
                    "forecast": forecast,
                    "lower": max(forecast * 0.85, 0.0),
                    "upper": forecast * 1.15,
                }
            )
            history.append(float(forecast))
        return pd.DataFrame(rows)

    def _point_prediction(self, sku: str) -> float:
        history = self._history.get(sku, [])
        if len(history) >= self.season:
            return float(history[-self.season])
        return float(np.mean(history)) if history else 0.0

    @property
    def name(self) -> str:
        return "SeasonalNaive"
