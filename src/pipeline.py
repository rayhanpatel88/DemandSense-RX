"""DemandSense-RX pipeline orchestrator."""

from __future__ import annotations

import pandas as pd
from typing import Optional

from src.data.loader import load_or_generate
from src.evaluation.backtesting import RollingBacktester
from src.evaluation.metrics import compute_all_metrics
from src.features.engineer import build_future_exogenous_frame, create_features, get_feature_cols
from src.forecasting.recursive import RecursiveForecaster
from src.forecasting.reliability import ReliabilityScorer
from src.models.baseline import MovingAverageForecaster, SeasonalNaiveForecaster
from src.models.lgbm_model import LGBMForecaster
from src.recommendations.inventory import InventoryEngine
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_pipeline(
    config: Optional[dict] = None,
    include_backtesting: bool = False,
    include_shap: bool = False,
) -> dict:
    config = config or load_config()
    raw_df = load_or_generate(config)
    featured_df = create_features(raw_df, config)
    feature_cols = get_feature_cols(featured_df)

    dates = sorted(featured_df["date"].unique())
    split_idx = int(len(dates) * float(config["forecasting"].get("train_ratio", 0.8)))
    train_cutoff = pd.Timestamp(dates[split_idx])

    train_raw = raw_df[raw_df["date"] < train_cutoff].copy()
    test_raw = raw_df[raw_df["date"] >= train_cutoff].copy()
    train_df = featured_df[featured_df["date"] < train_cutoff].dropna(subset=feature_cols + ["demand"]).copy()
    test_exogenous = test_raw[["date", "sku", "category", "family", "promotion", "price", "lead_time_days"]].copy()

    lgbm = LGBMForecaster(config).fit(train_df, feature_cols)
    ma = MovingAverageForecaster(window=config["models"]["baseline"].get("window", 7)).fit(train_df, feature_cols)
    sn = SeasonalNaiveForecaster().fit(train_df, feature_cols)

    recursive = RecursiveForecaster(lgbm)
    test_recursive = recursive.forecast(featured_df[featured_df["date"] < train_cutoff].copy(), test_exogenous, feature_cols)
    test_df = test_recursive.forecast_frame.merge(
        test_raw[["date", "sku", "demand"]],
        on=["date", "sku"],
        how="left",
    )

    horizon = int(config["forecasting"].get("default_horizon", 30))
    future_exogenous = build_future_exogenous_frame(raw_df, horizon=horizon, config=config)
    future_recursive = recursive.forecast(featured_df.copy(), future_exogenous, feature_cols)
    future_df = future_recursive.forecast_frame.copy()
    future_df["is_future"] = True
    future_df["horizon_day"] = future_df.groupby("sku").cumcount() + 1

    reliability = ReliabilityScorer(config).score(test_df)
    test_df = test_df.merge(reliability, on="sku", how="left")
    future_df = future_df.merge(reliability, on="sku", how="left")

    inventory_engine = InventoryEngine(config)
    inventory_df = inventory_engine.compute(raw_df, future_df)
    stockout_timeline = inventory_engine.compute_stockout_timeline(raw_df, future_df, inventory_df=inventory_df)
    slotting_df = _build_slotting_plan(future_df, inventory_df)

    shap_data = _compute_shap(lgbm, train_df, feature_cols) if include_shap else {}
    feature_importance = lgbm.get_feature_importance()
    holdout_metrics = compute_all_metrics(test_df["demand"].values, test_df["forecast"].values) if not test_df.empty else {}
    backtest_results = {"summary": pd.DataFrame(), "by_sku": pd.DataFrame(), "predictions": pd.DataFrame()}
    if include_backtesting:
        backtester = RollingBacktester(n_splits=2, test_size=21)
        backtest_results = backtester.run(raw_df, {lgbm.name: lgbm, ma.name: ma, sn.name: sn}, config)

    return {
        "config": config,
        "raw_df": raw_df,
        "featured_df": featured_df,
        "train_df": train_df,
        "test_df": test_df,
        "future_df": future_df,
        "future_7d_df": future_df[future_df["horizon_day"] <= 7].copy(),
        "future_30d_df": future_df[future_df["horizon_day"] <= 30].copy(),
        "feature_cols": feature_cols,
        "lgbm_model": lgbm,
        "ma_model": ma,
        "inventory_df": inventory_df,
        "stockout_timeline": stockout_timeline,
        "backtest_results": backtest_results,
        "holdout_metrics": holdout_metrics,
        "reliability_df": reliability,
        "slotting_df": slotting_df,
        "shap_data": shap_data,
        "feature_importance": feature_importance,
        "train_cutoff": train_cutoff,
    }


def _build_slotting_plan(future_df: pd.DataFrame, inventory_df: pd.DataFrame) -> pd.DataFrame:
    volume = future_df.groupby("sku")["forecast"].sum().rename("forecast_30d_total")
    risk = inventory_df.set_index("sku")[["stockout_risk", "days_to_stockout", "current_stock", "reorder_point"]]
    slotting = volume.to_frame().join(risk, how="left").sort_values("forecast_30d_total", ascending=False).reset_index()
    slotting["slot_rank"] = slotting.index + 1
    return slotting


def _compute_shap(lgbm: LGBMForecaster, train_df: pd.DataFrame, feature_cols: list[str]) -> dict:
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
    except Exception as exc:
        logger.warning("SHAP computation failed: %s", exc)
        return {}
