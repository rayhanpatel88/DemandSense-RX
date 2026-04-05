"""Abstract base class for all forecasters."""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseForecaster(ABC):
    """Base interface for demand forecasting models."""

    @abstractmethod
    def fit(self, df: pd.DataFrame, feature_cols: list, target_col: str = "demand"):
        """Fit the model on training data."""
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        """Generate point forecasts."""
        ...

    def predict_with_intervals(
        self, df: pd.DataFrame, feature_cols: list
    ) -> pd.DataFrame:
        """Return DataFrame with columns: forecast, lower, upper."""
        point = self.predict(df, feature_cols)
        return pd.DataFrame({"forecast": point, "lower": point, "upper": point},
                            index=df.index)

    @property
    def name(self) -> str:
        return self.__class__.__name__
