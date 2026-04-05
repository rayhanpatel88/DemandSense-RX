"""Unit tests for inventory intelligence engine."""

import pandas as pd
import numpy as np
import pytest
from src.recommendations.inventory import InventoryEngine


@pytest.fixture
def config():
    return {
        "inventory": {
            "default_lead_time_days": 7,
            "default_service_level": 0.95,
            "default_reorder_quantity_multiplier": 2.0,
        }
    }


@pytest.fixture
def historical_df():
    dates = pd.date_range("2023-01-01", periods=90, freq="D")
    records = []
    for sku in ["SKU_001", "SKU_002", "SKU_003"]:
        rng = np.random.default_rng(hash(sku) % 2**32)
        for date in dates:
            records.append({
                "date": date, "sku": sku,
                "demand": int(rng.integers(80, 120)),
            })
    return pd.DataFrame(records)


@pytest.fixture
def forecast_df(historical_df):
    future = pd.date_range("2023-04-01", periods=30, freq="D")
    records = []
    for sku in historical_df["sku"].unique():
        for date in future:
            records.append({
                "date": date, "sku": sku,
                "forecast": 100.0, "lower": 80.0, "upper": 120.0,
            })
    return pd.DataFrame(records)


def test_output_columns(config, historical_df, forecast_df):
    engine = InventoryEngine(config)
    result = engine.compute(historical_df, forecast_df)
    required = {"sku", "safety_stock", "reorder_point", "reorder_qty",
                "days_to_stockout", "stockout_risk", "reorder_needed"}
    assert required.issubset(set(result.columns))


def test_one_row_per_sku(config, historical_df, forecast_df):
    engine = InventoryEngine(config)
    result = engine.compute(historical_df, forecast_df)
    assert len(result) == historical_df["sku"].nunique()


def test_safety_stock_positive(config, historical_df, forecast_df):
    engine = InventoryEngine(config)
    result = engine.compute(historical_df, forecast_df)
    assert (result["safety_stock"] >= 0).all()


def test_reorder_point_exceeds_safety_stock(config, historical_df, forecast_df):
    engine = InventoryEngine(config)
    result = engine.compute(historical_df, forecast_df)
    assert (result["reorder_point"] >= result["safety_stock"]).all()


def test_days_to_stockout_positive(config, historical_df, forecast_df):
    engine = InventoryEngine(config)
    result = engine.compute(historical_df, forecast_df)
    assert (result["days_to_stockout"] >= 0).all()


def test_risk_flags_valid(config, historical_df, forecast_df):
    engine = InventoryEngine(config)
    result = engine.compute(historical_df, forecast_df)
    valid_risks = {"critical", "high", "medium", "low"}
    assert set(result["stockout_risk"].unique()).issubset(valid_risks)


def test_higher_lead_time_increases_safety_stock(config, historical_df, forecast_df):
    engine_short = InventoryEngine({**config, "inventory": {
        **config["inventory"], "default_lead_time_days": 3
    }})
    engine_long = InventoryEngine({**config, "inventory": {
        **config["inventory"], "default_lead_time_days": 14
    }})
    r_short = engine_short.compute(historical_df, forecast_df)
    r_long = engine_long.compute(historical_df, forecast_df)
    assert (r_long["safety_stock"] >= r_short["safety_stock"]).all()


def test_stockout_timeline_columns(config, historical_df, forecast_df):
    engine = InventoryEngine(config)
    timeline = engine.compute_stockout_timeline(historical_df, forecast_df)
    assert {"date", "sku", "projected_stock", "reorder_point"}.issubset(set(timeline.columns))
