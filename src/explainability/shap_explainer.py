"""SHAP-based explainability for tree models."""

import numpy as np
import pandas as pd
import shap
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SHAPExplainer:
    """Wraps SHAP TreeExplainer for LightGBM models."""

    def __init__(self, model, feature_cols: list):
        self.feature_cols = feature_cols
        logger.info("Initialising SHAP TreeExplainer")
        self.explainer = shap.TreeExplainer(model)
        self.shap_values_: np.ndarray | None = None
        self.sample_df_: pd.DataFrame | None = None

    def compute(self, df: pd.DataFrame, max_samples: int = 500) -> "SHAPExplainer":
        """Compute SHAP values on a sample of df."""
        sample = df.dropna(subset=self.feature_cols)
        if len(sample) > max_samples:
            sample = sample.sample(max_samples, random_state=42)
        self.sample_df_ = sample.reset_index(drop=True)
        X = self.sample_df_[self.feature_cols]
        logger.info(f"Computing SHAP values for {len(X)} samples")
        self.shap_values_ = self.explainer.shap_values(X)
        return self

    def global_importance(self) -> pd.DataFrame:
        """Mean absolute SHAP value per feature."""
        if self.shap_values_ is None:
            raise RuntimeError("Call compute() first")
        mean_abs = np.abs(self.shap_values_).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.feature_cols, "importance": mean_abs})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def local_explanation(self, row_idx: int) -> pd.DataFrame:
        """SHAP values for a single prediction (row index into sample_df_)."""
        if self.shap_values_ is None:
            raise RuntimeError("Call compute() first")
        if row_idx >= len(self.shap_values_):
            row_idx = len(self.shap_values_) - 1
        sv = self.shap_values_[row_idx]
        feat_vals = self.sample_df_[self.feature_cols].iloc[row_idx]
        return (
            pd.DataFrame({
                "feature": self.feature_cols,
                "shap_value": sv,
                "feature_value": feat_vals.values,
            })
            .sort_values("shap_value", key=abs, ascending=False)
            .reset_index(drop=True)
        )

    def get_shap_df(self) -> pd.DataFrame:
        """Return SHAP values as a DataFrame for visualisation."""
        if self.shap_values_ is None:
            raise RuntimeError("Call compute() first")
        return pd.DataFrame(self.shap_values_, columns=self.feature_cols)
