"""Feature engineering utilities for leakage-safe demand forecasting."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class FeatureSpec:
    """Configuration for demand features used in training and inference."""

    lag_features: list[int]
    rolling_windows: list[int]
    seasonal_period: int


CALENDAR_FEATURES = [
    "day_of_week",
    "week_of_year",
    "month",
    "quarter",
    "day_of_month",
    "day_of_year",
    "is_weekend",
    "is_month_start",
    "is_month_end",
]

STATIC_FEATURES = [
    "promotion",
    "price",
    "price_index",
    "category_encoded",
    "family_encoded",
    "sku_encoded",
    "lead_time_days",
]


def build_feature_spec(config: dict) -> FeatureSpec:
    cfg = config["forecasting"]
    lags = sorted({int(v) for v in cfg.get("lag_features", [1, 7, 14, 28])})
    windows = sorted({int(v) for v in cfg.get("rolling_windows", [7, 14, 28])})
    seasonal_period = int(cfg.get("seasonal_period", 7))
    return FeatureSpec(lag_features=lags, rolling_windows=windows, seasonal_period=seasonal_period)


def create_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Create training features from historical data only."""
    spec = build_feature_spec(config)
    frame = df.copy().sort_values(["sku", "date"]).reset_index(drop=True)
    _ensure_optional_columns(frame)
    frame = _add_calendar_features(frame)
    frame = _add_encoded_features(frame)

    grouped = frame.groupby("sku", group_keys=False)
    for lag in spec.lag_features:
        frame[f"lag_{lag}"] = grouped["demand"].shift(lag)

    for window in spec.rolling_windows:
        shifted = grouped["demand"].shift(1)
        frame[f"rolling_mean_{window}"] = shifted.groupby(frame["sku"]).transform(
            lambda series: series.rolling(window, min_periods=1).mean()
        )
        frame[f"rolling_std_{window}"] = shifted.groupby(frame["sku"]).transform(
            lambda series: series.rolling(window, min_periods=2).std()
        ).fillna(0.0)

    seasonal = spec.seasonal_period
    frame[f"lag_{seasonal}_promo"] = grouped["promotion"].shift(seasonal).fillna(0)
    frame["promo_lag_1"] = grouped["promotion"].shift(1).fillna(0)
    frame["promo_rate_28"] = grouped["promotion"].shift(1).groupby(frame["sku"]).transform(
        lambda series: series.rolling(28, min_periods=1).mean()
    ).fillna(0.0)
    frame["price_ma_28"] = grouped["price"].shift(1).groupby(frame["sku"]).transform(
        lambda series: series.rolling(28, min_periods=1).mean()
    )
    frame["price_gap"] = (frame["price"] - frame["price_ma_28"]).fillna(0.0)
    frame["demand_cv_28"] = (
        frame["rolling_std_28"] / frame["rolling_mean_28"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    frame["trend_7_vs_28"] = (
        frame["rolling_mean_7"] - frame["rolling_mean_28"]
    ).fillna(0.0)

    logger.info("Created %s forecasting features", len(get_feature_cols(frame)))
    return frame


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Public accessor for feature columns."""
    exclude = {
        "date",
        "sku",
        "category",
        "family",
        "demand",
        "inventory_position",
        "received_inventory",
        "unconstrained_demand",
        "lost_sales",
        "unit_cost",
        "service_level_target",
    }
    return [col for col in df.columns if col not in exclude]


class RecursiveFeatureGenerator:
    """Stateful feature generator used during recursive multi-step inference."""

    def __init__(self, history_df: pd.DataFrame, config: dict):
        self.spec = build_feature_spec(config)
        self.max_history = max(max(self.spec.lag_features), max(self.spec.rolling_windows), 35)
        ordered = history_df.sort_values("date").copy()
        _ensure_optional_columns(ordered)
        if ordered.empty:
            raise ValueError("history_df must contain at least one observation for recursive inference")

        self.sku = str(ordered["sku"].iloc[0])
        self.category = str(ordered["category"].iloc[-1])
        self.family = str(ordered["family"].iloc[-1]) if "family" in ordered.columns else self.category
        self.sku_encoded = int(ordered["sku_encoded"].iloc[-1]) if "sku_encoded" in ordered.columns else 0
        self.category_encoded = (
            int(ordered["category_encoded"].iloc[-1]) if "category_encoded" in ordered.columns else 0
        )
        self.family_encoded = (
            int(ordered["family_encoded"].iloc[-1]) if "family_encoded" in ordered.columns else 0
        )
        self.lead_time_days = float(ordered["lead_time_days"].iloc[-1]) if "lead_time_days" in ordered.columns else 7.0
        self.price_anchor = float(ordered["price"].tail(28).median())
        self.demand_history: Deque[float] = deque(ordered["demand"].tail(self.max_history).astype(float), maxlen=self.max_history)
        self.promo_history: Deque[float] = deque(ordered["promotion"].tail(self.max_history).astype(float), maxlen=self.max_history)
        self.price_history: Deque[float] = deque(ordered["price"].tail(self.max_history).astype(float), maxlen=self.max_history)

    def build_feature_row(self, date: pd.Timestamp, price: float, promotion: float) -> dict:
        history = list(self.demand_history)
        promo_history = list(self.promo_history)
        price_history = list(self.price_history)
        row = {
            "date": pd.Timestamp(date),
            "sku": self.sku,
            "category": self.category,
            "family": self.family,
            "promotion": float(promotion),
            "price": float(price),
            "price_index": float(price) / max(self.price_anchor, 1e-6),
            "sku_encoded": self.sku_encoded,
            "category_encoded": self.category_encoded,
            "family_encoded": self.family_encoded,
            "lead_time_days": self.lead_time_days,
        }
        row.update(_calendar_row(pd.Timestamp(date)))

        for lag in self.spec.lag_features:
            row[f"lag_{lag}"] = _safe_lag(history, lag)

        for window in self.spec.rolling_windows:
            window_values = history[-window:] if history else []
            row[f"rolling_mean_{window}"] = float(np.mean(window_values)) if window_values else 0.0
            row[f"rolling_std_{window}"] = float(np.std(window_values, ddof=1)) if len(window_values) > 1 else 0.0

        seasonal = self.spec.seasonal_period
        row[f"lag_{seasonal}_promo"] = _safe_lag(promo_history, seasonal)
        row["promo_lag_1"] = _safe_lag(promo_history, 1)
        row["promo_rate_28"] = float(np.mean(promo_history[-28:])) if promo_history else 0.0
        price_ma_28 = float(np.mean(price_history[-28:])) if price_history else float(price)
        row["price_ma_28"] = price_ma_28
        row["price_gap"] = float(price) - price_ma_28
        rolling_mean_28 = row.get("rolling_mean_28", row.get("rolling_mean_7", 0.0))
        rolling_std_28 = row.get("rolling_std_28", 0.0)
        row["demand_cv_28"] = rolling_std_28 / rolling_mean_28 if rolling_mean_28 > 0 else 0.0
        row["trend_7_vs_28"] = row.get("rolling_mean_7", 0.0) - rolling_mean_28
        return row

    def update_history(self, demand: float, promotion: float, price: float) -> None:
        self.demand_history.append(float(demand))
        self.promo_history.append(float(promotion))
        self.price_history.append(float(price))


def build_future_exogenous_frame(history_df: pd.DataFrame, horizon: int, config: dict) -> pd.DataFrame:
    """Construct plausible future exogenous inputs without using future demand."""
    _ = config
    rows: list[dict] = []
    for sku, group in history_df.sort_values("date").groupby("sku"):
        group = group.sort_values("date").copy()
        last_date = pd.Timestamp(group["date"].max())
        recent = group.tail(56)
        base_price = float(recent["price"].tail(28).median())
        promo_rate_by_dow = recent.groupby(recent["date"].dt.dayofweek)["promotion"].mean().to_dict()
        seasonal_price_by_dow = recent.groupby(recent["date"].dt.dayofweek)["price"].median().to_dict()
        category = str(group["category"].iloc[-1])
        family = str(group["family"].iloc[-1]) if "family" in group.columns else category
        lead_time_days = float(group["lead_time_days"].iloc[-1]) if "lead_time_days" in group.columns else 7.0

        for step in range(1, horizon + 1):
            future_date = last_date + pd.Timedelta(days=step)
            dow = future_date.dayofweek
            promo_probability = float(promo_rate_by_dow.get(dow, recent["promotion"].mean()))
            promotion = 1.0 if promo_probability >= 0.4 else 0.0
            price = float(seasonal_price_by_dow.get(dow, base_price))
            if promotion:
                price *= 0.9
            rows.append(
                {
                    "date": future_date,
                    "sku": sku,
                    "category": category,
                    "family": family,
                    "promotion": promotion,
                    "price": round(price, 2),
                    "lead_time_days": lead_time_days,
                }
            )

    future = pd.DataFrame(rows).sort_values(["sku", "date"]).reset_index(drop=True)
    _ensure_optional_columns(future)
    return future


def _ensure_optional_columns(df: pd.DataFrame) -> None:
    if "promotion" not in df.columns:
        df["promotion"] = 0.0
    if "price" not in df.columns:
        df["price"] = 1.0
    if "category" not in df.columns:
        df["category"] = "Unknown"
    if "family" not in df.columns:
        df["family"] = df["category"]
    if "lead_time_days" not in df.columns:
        df["lead_time_days"] = 7


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["week_of_year"] = frame["date"].dt.isocalendar().week.astype(int)
    frame["month"] = frame["date"].dt.month
    frame["quarter"] = frame["date"].dt.quarter
    frame["day_of_month"] = frame["date"].dt.day
    frame["day_of_year"] = frame["date"].dt.dayofyear
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(int)
    frame["is_month_start"] = frame["date"].dt.is_month_start.astype(int)
    frame["is_month_end"] = frame["date"].dt.is_month_end.astype(int)
    return frame


def _add_encoded_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["sku_encoded"] = pd.factorize(frame["sku"], sort=True)[0]
    frame["category_encoded"] = pd.factorize(frame["category"], sort=True)[0]
    frame["family_encoded"] = pd.factorize(frame["family"], sort=True)[0]
    price_anchor = frame.groupby("sku")["price"].transform(lambda values: values.rolling(28, min_periods=1).median())
    frame["price_index"] = frame["price"] / price_anchor.replace(0, np.nan)
    frame["price_index"] = frame["price_index"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return frame


def _calendar_row(date: pd.Timestamp) -> dict:
    return {
        "day_of_week": date.dayofweek,
        "week_of_year": int(date.isocalendar().week),
        "month": date.month,
        "quarter": date.quarter,
        "day_of_month": date.day,
        "day_of_year": date.dayofyear,
        "is_weekend": int(date.dayofweek >= 5),
        "is_month_start": int(date.is_month_start),
        "is_month_end": int(date.is_month_end),
    }


def _safe_lag(values: list[float], lag: int) -> float:
    if not values or lag <= 0 or len(values) < lag:
        return 0.0
    return float(values[-lag])
