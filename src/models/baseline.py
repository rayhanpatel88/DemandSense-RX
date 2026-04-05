"""Baseline forecasting models."""

import pandas as pd
import numpy as np
from src.models.base import BaseForecaster
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MovingAverageForecaster(BaseForecaster):
    """Rolling window moving average baseline."""

    def __init__(self, window: int = 7):
        self.window = window
        self._sku_means: dict = {}
        self._sku_stds: dict = {}

    def fit(self, df: pd.DataFrame, feature_cols: list, target_col: str = "demand"):
        for sku, group in df.groupby("sku"):
            vals = group.sort_values("date")[target_col].values
            if len(vals) >= self.window:
                self._sku_means[sku] = float(np.mean(vals[-self.window:]))
                self._sku_stds[sku] = float(np.std(vals[-self.window:]))
            else:
                self._sku_means[sku] = float(np.mean(vals))
                self._sku_stds[sku] = float(np.std(vals))
        logger.info(f"MovingAverageForecaster fitted on {len(self._sku_means)} SKUs")
        return self

    def predict(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        preds = df["sku"].map(self._sku_means).fillna(0).values
        return np.clip(preds, 0, None)

    def predict_with_intervals(
        self, df: pd.DataFrame, feature_cols: list
    ) -> pd.DataFrame:
        point = self.predict(df, feature_cols)
        stds = df["sku"].map(self._sku_stds).fillna(0).values
        lower = np.clip(point - 1.645 * stds, 0, None)
        upper = point + 1.645 * stds
        return pd.DataFrame({"forecast": point, "lower": lower, "upper": upper},
                            index=df.index)

    @property
    def name(self) -> str:
        return f"MovingAverage(w={self.window})"


class SeasonalNaiveForecaster(BaseForecaster):
    """Seasonal naive: use same day from last week."""

    def __init__(self, season: int = 7):
        self.season = season
        self._history: dict = {}

    def fit(self, df: pd.DataFrame, feature_cols: list, target_col: str = "demand"):
        for sku, group in df.groupby("sku"):
            vals = group.sort_values("date")[target_col].values
            self._history[sku] = vals
        logger.info(f"SeasonalNaiveForecaster fitted on {len(self._history)} SKUs")
        return self

    def predict(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        """Use lag_{season} column if available, else fallback to mean."""
        lag_col = f"lag_{self.season}"
        if lag_col in df.columns:
            return np.clip(df[lag_col].fillna(0).values, 0, None)
        # Fallback: use mean from history
        means = {s: float(np.mean(v)) for s, v in self._history.items()}
        return np.clip(df["sku"].map(means).fillna(0).values, 0, None)

    @property
    def name(self) -> str:
        return f"SeasonalNaive(s={self.season})"
