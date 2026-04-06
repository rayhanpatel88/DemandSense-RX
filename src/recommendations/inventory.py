"""Inventory policy engine with service-level based recommendations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class InventoryEngine:
    """Compute credible inventory recommendations per SKU."""

    def __init__(self, config: dict):
        inv_cfg = config["inventory"]
        self.lead_time = int(inv_cfg.get("default_lead_time_days", 7))
        self.service_level = float(inv_cfg.get("default_service_level", 0.95))
        self.review_period = int(inv_cfg.get("review_period_days", 7))
        self.order_cycle = int(inv_cfg.get("order_cycle_days", 14))
        self.max_cover_days = int(inv_cfg.get("max_cover_days", 35))
        self.reorder_qty_multiplier = float(inv_cfg.get("default_reorder_quantity_multiplier", 1.25))

    def compute(
        self,
        historical_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        current_stock: Optional[dict] = None,
        lead_time: Optional[int] = None,
        service_level: Optional[float] = None,
    ) -> pd.DataFrame:
        lt = int(lead_time if lead_time is not None else self.lead_time)
        sl = float(service_level if service_level is not None else self.service_level)
        z_value = float(stats.norm.ppf(sl))

        rows: list[dict] = []
        ordered_history = historical_df.sort_values(["sku", "date"]).copy()

        for sku, hist_group in ordered_history.groupby("sku"):
            hist_group = hist_group.sort_values("date")
            demand = hist_group["demand"].astype(float)
            latest_stock = (
                float(current_stock[sku]) if current_stock and sku in current_stock
                else float(hist_group["inventory_position"].iloc[-1]) if "inventory_position" in hist_group.columns
                else float(max(demand.tail(30).mean() * 18, demand.tail(30).sum() * 0.7))
            )
            sku_forecast = forecast_df[forecast_df["sku"] == sku].sort_values("date").copy()
            if sku_forecast.empty:
                continue

            lead_window = sku_forecast.head(lt)
            cycle_window = sku_forecast.head(min(len(sku_forecast), lt + self.review_period))
            mean_daily = float(demand.tail(90).mean())
            demand_std = float(demand.tail(90).std(ddof=1)) if len(demand.tail(90)) > 1 else max(mean_daily * 0.25, 1.0)
            mean_lead_time_demand = float(lead_window["forecast"].sum())
            lead_time_std = demand_std * np.sqrt(max(lt, 1))
            safety_stock = max(z_value * lead_time_std, 0.0)
            reorder_point = mean_lead_time_demand + safety_stock
            target_stock = float(cycle_window["forecast"].sum() + safety_stock)
            target_stock = min(target_stock, mean_daily * self.max_cover_days)
            reorder_gap = max(target_stock - latest_stock, 0.0)
            reorder_qty = max(reorder_gap, mean_daily * lt * 0.35) if latest_stock <= reorder_point else 0.0

            cumulative_forecast = sku_forecast["forecast"].cumsum()
            stock_position = latest_stock - cumulative_forecast
            stockout_idx = stock_position.le(0).idxmax() if (stock_position <= 0).any() else None
            days_to_stockout = (
                int((pd.Timestamp(sku_forecast.loc[stockout_idx, "date"]) - pd.Timestamp(sku_forecast["date"].min())).days + 1)
                if stockout_idx is not None
                else int(self.max_cover_days)
            )

            shortage_units = max(mean_lead_time_demand - latest_stock, 0.0)
            coverage_days = latest_stock / max(mean_daily, 1.0)
            risk = self._risk_category(latest_stock, reorder_point, days_to_stockout, lt)
            explanation = self._build_explanation(
                latest_stock=latest_stock,
                reorder_point=reorder_point,
                safety_stock=safety_stock,
                coverage_days=coverage_days,
                shortage_units=shortage_units,
                reorder_qty=reorder_qty,
                risk=risk,
            )

            rows.append(
                {
                    "sku": sku,
                    "category": hist_group["category"].iloc[-1] if "category" in hist_group.columns else "Unknown",
                    "family": hist_group["family"].iloc[-1] if "family" in hist_group.columns else "Unknown",
                    "mean_daily_demand": round(mean_daily, 1),
                    "demand_std": round(demand_std, 1),
                    "lead_time_days": lt,
                    "service_level": round(sl, 3),
                    "safety_stock": round(safety_stock, 1),
                    "reorder_point": round(reorder_point, 1),
                    "target_stock": round(target_stock, 1),
                    "reorder_qty": round(reorder_qty, 1),
                    "current_stock": round(latest_stock, 1),
                    "coverage_days": round(coverage_days, 1),
                    "days_to_stockout": round(float(days_to_stockout), 1),
                    "forecast_7d_total": round(float(sku_forecast.head(7)["forecast"].sum()), 1),
                    "forecast_30d_total": round(float(sku_forecast.head(30)["forecast"].sum()), 1),
                    "forecast_daily_avg": round(float(sku_forecast["forecast"].mean()), 1),
                    "shortage_units": round(shortage_units, 1),
                    "reorder_needed": int(latest_stock <= reorder_point),
                    "stockout_risk": risk,
                    "explanation": explanation,
                }
            )

        result = pd.DataFrame(rows).sort_values(["reorder_needed", "days_to_stockout"], ascending=[False, True])
        logger.info("Computed inventory recommendations for %s SKUs", len(result))
        return result.reset_index(drop=True)

    def compute_stockout_timeline(
        self,
        historical_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        inventory_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        inventory = inventory_df if inventory_df is not None else self.compute(historical_df, forecast_df)
        rows: list[dict] = []
        for _, inv_row in inventory.iterrows():
            stock = float(inv_row["current_stock"])
            sku_forecast = forecast_df[forecast_df["sku"] == inv_row["sku"]].sort_values("date")
            for _, forecast_row in sku_forecast.iterrows():
                stock = max(stock - float(forecast_row["forecast"]), 0.0)
                rows.append(
                    {
                        "date": forecast_row["date"],
                        "sku": inv_row["sku"],
                        "projected_stock": round(stock, 1),
                        "reorder_point": inv_row["reorder_point"],
                        "risk": inv_row["stockout_risk"],
                    }
                )
        return pd.DataFrame(rows)

    def _risk_category(self, current_stock: float, reorder_point: float, days_to_stockout: float, lead_time: int) -> str:
        if current_stock <= 0 or days_to_stockout <= max(lead_time * 0.5, 2):
            return "critical"
        if current_stock <= reorder_point or days_to_stockout <= lead_time:
            return "high"
        if days_to_stockout <= lead_time * 2:
            return "medium"
        return "low"

    def _build_explanation(
        self,
        latest_stock: float,
        reorder_point: float,
        safety_stock: float,
        coverage_days: float,
        shortage_units: float,
        reorder_qty: float,
        risk: str,
    ) -> str:
        if latest_stock <= reorder_point:
            return (
                f"{risk.title()} risk. Current stock is below the reorder point, "
                f"with {coverage_days:.1f} days of cover. Reorder {reorder_qty:.0f} units "
                f"to restore service-level buffer including {safety_stock:.0f} units of safety stock."
            )
        if shortage_units > 0:
            return (
                f"{risk.title()} risk. Lead-time demand exceeds available stock by {shortage_units:.0f} units, "
                f"despite a safety stock target of {safety_stock:.0f}."
            )
        return (
            f"{risk.title()} risk. Stock remains above the reorder point with {coverage_days:.1f} days of cover "
            f"and a safety stock buffer of {safety_stock:.0f} units."
        )
