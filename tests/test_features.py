"""Unit tests for feature engineering."""

import pandas as pd
import numpy as np
import pytest
from src.features.engineer import create_features, get_feature_cols


@pytest.fixture
def sample_df():
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    records = []
    for sku in ["SKU_001", "SKU_002"]:
        for date in dates:
            records.append({
                "date": date, "sku": sku,
                "demand": int(np.random.default_rng(42).integers(50, 150)),
                "price": 29.99, "promotion": 0, "category": "Electronics",
            })
    return pd.DataFrame(records)


@pytest.fixture
def config():
    return {
        "forecasting": {
            "lag_features": [1, 7, 14, 28],
            "rolling_windows": [7, 14, 28],
        }
    }


def test_feature_count(sample_df, config):
    featured = create_features(sample_df, config)
    feat_cols = get_feature_cols(featured)
    assert len(feat_cols) > 10, "Expected more than 10 feature columns"


def test_calendar_features_created(sample_df, config):
    featured = create_features(sample_df, config)
    for col in ["day_of_week", "month", "week_of_year", "is_weekend"]:
        assert col in featured.columns, f"Missing calendar feature: {col}"


def test_lag_features_created(sample_df, config):
    featured = create_features(sample_df, config)
    for lag in config["forecasting"]["lag_features"]:
        assert f"lag_{lag}" in featured.columns, f"Missing lag feature: lag_{lag}"


def test_rolling_features_created(sample_df, config):
    featured = create_features(sample_df, config)
    for w in config["forecasting"]["rolling_windows"]:
        assert f"rolling_mean_{w}" in featured.columns


def test_sku_encoded(sample_df, config):
    featured = create_features(sample_df, config)
    assert "sku_encoded" in featured.columns
    assert featured["sku_encoded"].nunique() == 2


def test_no_negative_lag(sample_df, config):
    featured = create_features(sample_df, config)
    # Lags of demand should be NaN or non-negative (demand is always >= 0)
    lag1 = featured["lag_1"].dropna()
    assert (lag1 >= 0).all(), "Lag features should be non-negative"


def test_row_count_preserved(sample_df, config):
    featured = create_features(sample_df, config)
    assert len(featured) == len(sample_df), "Row count should be unchanged"
