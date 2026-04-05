"""LightGBM demand forecasting model with quantile regression for intervals."""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import lightgbm as lgb

from src.models.base import BaseForecaster
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LGBMForecaster(BaseForecaster):
    """LightGBM forecaster with 80% prediction intervals via quantile regression."""

    def __init__(self, config: dict):
        cfg = config["models"]["lgbm"]
        self.params_base = {
            "n_estimators": cfg.get("n_estimators", 300),
            "learning_rate": cfg.get("learning_rate", 0.05),
            "max_depth": cfg.get("max_depth", 6),
            "num_leaves": cfg.get("num_leaves", 31),
            "min_child_samples": cfg.get("min_child_samples", 20),
            "subsample": cfg.get("subsample", 0.8),
            "colsample_bytree": cfg.get("colsample_bytree", 0.8),
            "reg_alpha": cfg.get("reg_alpha", 0.1),
            "reg_lambda": cfg.get("reg_lambda", 0.1),
            "verbose": -1,
            "n_jobs": -1,
        }
        self.quantiles = config["forecasting"].get("quantiles", [0.1, 0.5, 0.9])
        self._models: dict = {}  # keyed by quantile or "point"
        self.feature_cols_: list = []

    def fit(self, df: pd.DataFrame, feature_cols: list, target_col: str = "demand"):
        self.feature_cols_ = feature_cols
        X = df[feature_cols]
        y = df[target_col].values

        # Point forecast model (RMSE objective)
        logger.info("Training LightGBM point forecast model")
        point_params = {**self.params_base, "objective": "regression", "metric": "rmse"}
        self._models["point"] = lgb.LGBMRegressor(**point_params)
        self._models["point"].fit(X, y)

        # Quantile models for prediction intervals
        for q in self.quantiles:
            logger.info(f"Training quantile model q={q}")
            q_params = {
                **self.params_base,
                "objective": "quantile",
                "alpha": q,
                "metric": "quantile",
            }
            model = lgb.LGBMRegressor(**q_params)
            model.fit(X, y)
            self._models[q] = model

        logger.info("LGBMForecaster training complete")
        return self

    def predict(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        X = df[feature_cols]
        preds = self._models["point"].predict(X)
        return np.clip(preds, 0, None)

    def predict_with_intervals(
        self, df: pd.DataFrame, feature_cols: list
    ) -> pd.DataFrame:
        X = df[feature_cols]
        point = np.clip(self._models["point"].predict(X), 0, None)

        q_low = min(self.quantiles)
        q_high = max(self.quantiles)
        lower = np.clip(self._models[q_low].predict(X), 0, None)
        upper = np.clip(self._models[q_high].predict(X), 0, None)

        # Ensure lower <= point <= upper
        lower = np.minimum(lower, point)
        upper = np.maximum(upper, point)

        return pd.DataFrame({"forecast": point, "lower": lower, "upper": upper},
                            index=df.index)

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importances from the point forecast model."""
        model = self._models.get("point")
        if model is None:
            return pd.DataFrame()
        imp = model.feature_importances_
        cols = self.feature_cols_
        return (
            pd.DataFrame({"feature": cols, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._models, path)
        logger.info(f"Saved LGBMForecaster models to {path}")

    def load(self, path: str):
        self._models = joblib.load(path)
        logger.info(f"Loaded LGBMForecaster models from {path}")
        return self

    @property
    def name(self) -> str:
        return "LightGBM"
