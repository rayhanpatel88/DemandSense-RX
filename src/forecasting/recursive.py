"""Recursive forecasting service used for inference and backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.features.engineer import RecursiveFeatureGenerator, get_feature_cols


@dataclass
class RecursiveForecastResult:
    forecast_frame: pd.DataFrame
    feature_frame: pd.DataFrame


class RecursiveForecaster:
    """Generate multi-step forecasts where each prediction updates future lags."""

    def __init__(self, model):
        self.model = model

    def forecast(
        self,
        history_features: pd.DataFrame,
        future_exogenous: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
    ) -> RecursiveForecastResult:
        if future_exogenous.empty:
            empty = future_exogenous.copy()
            return RecursiveForecastResult(empty, empty)

        feature_cols = feature_cols or getattr(self.model, "feature_cols_", None) or get_feature_cols(history_features)
        feature_rows: list[dict] = []
        output_rows: list[dict] = []

        for sku, future_sku in future_exogenous.sort_values(["sku", "date"]).groupby("sku"):
            history_sku = history_features[history_features["sku"] == sku].sort_values("date")
            generator = RecursiveFeatureGenerator(history_sku, {"forecasting": {
                "lag_features": [int(col.split("_")[1]) for col in feature_cols if col.startswith("lag_") and col[4:].isdigit()],
                "rolling_windows": [int(col.split("_")[2]) for col in feature_cols if col.startswith("rolling_mean_")],
                "seasonal_period": 7,
            }})

            for _, exog in future_sku.iterrows():
                feature_row = generator.build_feature_row(
                    pd.Timestamp(exog["date"]),
                    float(exog.get("price", history_sku["price"].iloc[-1])),
                    float(exog.get("promotion", 0.0)),
                )
                feature_frame = pd.DataFrame([feature_row])
                interval = self.model.predict_with_intervals(feature_frame, feature_cols).iloc[0]
                forecast = float(np.clip(interval["forecast"], 0, None))
                lower = float(np.clip(interval["lower"], 0, None))
                upper = float(np.clip(interval["upper"], 0, None))
                generator.update_history(forecast, feature_row["promotion"], feature_row["price"])

                record = {
                    **feature_row,
                    "demand": forecast,
                    "forecast": forecast,
                    "lower": lower,
                    "upper": upper,
                }
                feature_rows.append(record)
                output_rows.append(
                    {
                        "date": feature_row["date"],
                        "sku": feature_row["sku"],
                        "category": feature_row["category"],
                        "family": feature_row["family"],
                        "promotion": feature_row["promotion"],
                        "price": feature_row["price"],
                        "forecast": forecast,
                        "lower": lower,
                        "upper": upper,
                    }
                )

        forecast_df = pd.DataFrame(output_rows).sort_values(["sku", "date"]).reset_index(drop=True)
        feature_df = pd.DataFrame(feature_rows).sort_values(["sku", "date"]).reset_index(drop=True)
        return RecursiveForecastResult(forecast_frame=forecast_df, feature_frame=feature_df)
