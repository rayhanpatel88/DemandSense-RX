"""Feature engineering for demand forecasting."""

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Create all ML features from raw demand data."""
    cfg = config["forecasting"]
    lag_features = cfg.get("lag_features", [1, 7, 14, 28])
    rolling_windows = cfg.get("rolling_windows", [7, 14, 28])

    df = df.copy().sort_values(["sku", "date"])

    # Calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_month"] = df["date"].dt.day
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["year"] = df["date"].dt.year

    # Lag features per SKU
    for lag in lag_features:
        df[f"lag_{lag}"] = df.groupby("sku")["demand"].shift(lag)

    # Rolling statistics per SKU
    for window in rolling_windows:
        df[f"rolling_mean_{window}"] = (
            df.groupby("sku")["demand"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"rolling_std_{window}"] = (
            df.groupby("sku")["demand"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=2).std().fillna(0))
        )

    # Rolling max and min
    df["rolling_max_7"] = (
        df.groupby("sku")["demand"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).max())
    )
    df["rolling_min_7"] = (
        df.groupby("sku")["demand"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).min())
    )

    # Promotion lag (days since last promotion)
    df["promo_lag_1"] = df.groupby("sku")["promotion"].shift(1).fillna(0)
    df["promo_lag_7"] = df.groupby("sku")["promotion"].shift(7).fillna(0)

    # SKU encoding
    sku_codes = {s: i for i, s in enumerate(sorted(df["sku"].unique()))}
    df["sku_encoded"] = df["sku"].map(sku_codes)

    # Category encoding
    cat_codes = {c: i for i, c in enumerate(sorted(df["category"].unique()))}
    df["category_encoded"] = df["category"].map(cat_codes)

    feature_cols = _get_feature_cols(df)
    logger.info(f"Created {len(feature_cols)} features")
    return df


def _get_feature_cols(df: pd.DataFrame) -> list:
    """Return list of feature column names (excludes meta cols)."""
    exclude = {"date", "sku", "category", "demand"}
    return [c for c in df.columns if c not in exclude]


def get_feature_cols(df: pd.DataFrame) -> list:
    """Public accessor for feature column names."""
    return _get_feature_cols(df)
