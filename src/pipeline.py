"""
Main pipeline orchestrator.
Loads data → engineers features → trains models → generates forecasts
→ computes inventory → runs backtesting → computes SHAP.
Results are returned as a single dict used by all Streamlit pages.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.data.loader import load_or_generate
from src.features.engineer import create_features, get_feature_cols
from src.models.baseline import MovingAverageForecaster, SeasonalNaiveForecaster
from src.models.lgbm_model import LGBMForecaster
from src.evaluation.backtesting import RollingBacktester
from src.evaluation.metrics import compute_metrics_by_sku
from src.recommendations.inventory import InventoryEngine

logger = get_logger(__name__)


def run_pipeline(config: dict = None) -> dict:
    """Execute the full DemandSense-RX pipeline and return all artefacts."""
    if config is None:
        config = load_config()

    logger.info("=== DemandSense-RX Pipeline Start ===")

    # 1. Load / generate data
    raw_df = load_or_generate(config)

    # 2. Feature engineering
    featured_df = create_features(raw_df, config)
    feature_cols = get_feature_cols(featured_df)

    # 3. Train / test split (chronological)
    train_ratio = config["forecasting"].get("train_ratio", 0.8)
    dates = sorted(featured_df["date"].unique())
    split_idx = int(len(dates) * train_ratio)
    train_cutoff = dates[split_idx]

    train_df = featured_df[featured_df["date"] < train_cutoff].dropna(subset=feature_cols)
    test_df = featured_df[featured_df["date"] >= train_cutoff].dropna(subset=feature_cols)

    logger.info(f"Train: {len(train_df)} rows ({train_df['date'].min().date()} – "
                f"{train_df['date'].max().date()})")
    logger.info(f"Test: {len(test_df)} rows ({test_df['date'].min().date()} – "
                f"{test_df['date'].max().date()})")

    # 4. Train models
    lgbm = LGBMForecaster(config)
    lgbm.fit(train_df, feature_cols)

    ma = MovingAverageForecaster(window=config["models"]["baseline"].get("window", 7))
    ma.fit(train_df, feature_cols)

    sn = SeasonalNaiveForecaster()
    sn.fit(train_df, feature_cols)

    # 5. Generate forecasts on test set with prediction intervals
    forecast_intervals = lgbm.predict_with_intervals(test_df, feature_cols)
    test_df = test_df.copy()
    test_df["forecast"] = forecast_intervals["forecast"].values
    test_df["lower"] = forecast_intervals["lower"].values
    test_df["upper"] = forecast_intervals["upper"].values

    # Forecast uncertainty / reliability score
    interval_width = test_df["upper"] - test_df["lower"]
    max_width = interval_width.max() if interval_width.max() > 0 else 1
    rel_cfg = config["reliability"]
    test_df["uncertainty_score"] = (interval_width / max_width)
    test_df["reliability_score"] = 1.0 - test_df["uncertainty_score"]

    def _flag(s):
        hi = rel_cfg.get("high_threshold", 0.75)
        lo = rel_cfg.get("low_threshold", 0.45)
        if s >= hi:
            return "high"
        elif s >= lo:
            return "medium"
        return "low"

    test_df["confidence"] = test_df["reliability_score"].map(_flag)

    # 6. Future forecast (next 30 days beyond training data)
    future_df = _build_future_features(featured_df, feature_cols, config)
    future_intervals = lgbm.predict_with_intervals(future_df, feature_cols)
    future_df = future_df.copy()
    future_df["forecast"] = np.clip(future_intervals["forecast"].values, 0, None)
    future_df["lower"] = np.clip(future_intervals["lower"].values, 0, None)
    future_df["upper"] = np.clip(future_intervals["upper"].values, 0, None)
    future_df["is_future"] = True

    # 7. Backtesting
    logger.info("Running backtesting")
    backtester = RollingBacktester(n_splits=4, test_size=30)
    models = {
        lgbm.name: lgbm,
        ma.name: ma,
        sn.name: sn,
    }
    backtest_results = backtester.run(
        featured_df.dropna(subset=feature_cols),
        models,
        feature_cols,
        config,
    )

    # 8. Inventory recommendations
    inv_engine = InventoryEngine(config)
    # Combine test + future for inventory calculation
    all_forecast_df = pd.concat([
        test_df[["date", "sku", "demand", "forecast", "lower", "upper"]],
        future_df[["date", "sku", "demand", "forecast", "lower", "upper"]],
    ], ignore_index=True)

    inventory_df = inv_engine.compute(raw_df, future_df[["date", "sku", "forecast", "lower", "upper"]])
    stockout_timeline = inv_engine.compute_stockout_timeline(
        raw_df, future_df[["date", "sku", "forecast", "lower", "upper"]]
    )

    # 9. SHAP explainability
    logger.info("Computing SHAP values")
    shap_data = _compute_shap(lgbm, train_df, feature_cols)

    # 10. Feature importance from LightGBM
    feature_importance = lgbm.get_feature_importance()

    logger.info("=== Pipeline Complete ===")
    return {
        "config": config,
        "raw_df": raw_df,
        "featured_df": featured_df,
        "train_df": train_df,
        "test_df": test_df,
        "future_df": future_df,
        "feature_cols": feature_cols,
        "lgbm_model": lgbm,
        "ma_model": ma,
        "inventory_df": inventory_df,
        "stockout_timeline": stockout_timeline,
        "backtest_results": backtest_results,
        "shap_data": shap_data,
        "feature_importance": feature_importance,
        "train_cutoff": train_cutoff,
    }


def _build_future_features(featured_df: pd.DataFrame,
                            feature_cols: list,
                            config: dict) -> pd.DataFrame:
    """Build feature rows for future horizon by propagating last known values."""
    horizon = config["forecasting"].get("default_horizon", 30)
    last_date = featured_df["date"].max()
    skus = featured_df["sku"].unique()

    rows = []
    for sku in skus:
        sku_df = featured_df[featured_df["sku"] == sku].sort_values("date")
        last_row = sku_df.iloc[-1].copy()
        for d in range(1, horizon + 1):
            future_date = last_date + pd.Timedelta(days=d)
            row = last_row.copy()
            row["date"] = future_date
            row["demand"] = 0
            # Update calendar features
            row["day_of_week"] = future_date.dayofweek
            row["month"] = future_date.month
            row["week_of_year"] = future_date.isocalendar()[1]
            row["day_of_month"] = future_date.day
            row["quarter"] = future_date.quarter
            row["is_weekend"] = int(future_date.dayofweek >= 5)
            row["is_month_start"] = int(future_date.is_month_start)
            row["is_month_end"] = int(future_date.is_month_end)
            row["year"] = future_date.year
            rows.append(row)

    future_df = pd.DataFrame(rows).reset_index(drop=True)
    # Fill NaN features
    for col in feature_cols:
        if col in future_df.columns:
            future_df[col] = future_df[col].fillna(0)
    return future_df


def _compute_shap(lgbm: LGBMForecaster, train_df: pd.DataFrame,
                  feature_cols: list) -> dict:
    """Safely compute SHAP values, returning empty dict on failure."""
    try:
        from src.explainability.shap_explainer import SHAPExplainer
        explainer = SHAPExplainer(lgbm._models["point"], feature_cols)
        explainer.compute(train_df, max_samples=400)
        return {
            "explainer": explainer,
            "global_importance": explainer.global_importance(),
            "shap_df": explainer.get_shap_df(),
            "sample_df": explainer.sample_df_,
        }
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return {}
