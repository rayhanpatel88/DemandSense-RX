"""Inventory intelligence engine: safety stock, reorder points, stockout detection."""

import numpy as np
import pandas as pd
from scipy import stats
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InventoryEngine:
    """Computes inventory decisions per SKU from forecast data."""

    def __init__(self, config: dict):
        inv_cfg = config["inventory"]
        self.lead_time = inv_cfg.get("default_lead_time_days", 7)
        self.service_level = inv_cfg.get("default_service_level", 0.95)
        self.reorder_qty_multiplier = inv_cfg.get("default_reorder_quantity_multiplier", 2.0)
        self.z = float(stats.norm.ppf(self.service_level))

    def compute(
        self,
        historical_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        current_stock: dict = None,
        lead_time: int = None,
        service_level: float = None,
    ) -> pd.DataFrame:
        """
        Compute inventory recommendations for all SKUs.

        Parameters
        ----------
        historical_df : historical demand data
        forecast_df : DataFrame with columns [date, sku, forecast, lower, upper]
        current_stock : dict {sku: current_stock_units} (defaults to 30 days avg demand)
        lead_time : override lead time in days
        service_level : override service level (0-1)

        Returns
        -------
        DataFrame with one row per SKU containing inventory metrics
        """
        lt = lead_time if lead_time is not None else self.lead_time
        sl = service_level if service_level is not None else self.service_level
        z = float(stats.norm.ppf(sl))

        rows = []
        for sku, hist_group in historical_df.groupby("sku"):
            demands = hist_group["demand"].values
            mean_demand = float(np.mean(demands))
            std_demand = float(np.std(demands)) if len(demands) > 1 else mean_demand * 0.2

            # Safety stock
            safety_stock = z * std_demand * np.sqrt(lt)

            # Reorder point
            reorder_point = mean_demand * lt + safety_stock

            # Reorder quantity (EOQ-like: cover 2x lead time period)
            reorder_qty = mean_demand * lt * self.reorder_qty_multiplier

            # Current stock (default: 30-day mean demand)
            if current_stock and sku in current_stock:
                stock = current_stock[sku]
            else:
                stock = mean_demand * 30

            # Days until stockout
            days_to_stockout = stock / mean_demand if mean_demand > 0 else float("inf")

            # Forecast demand (next 30 days)
            sku_fcst = forecast_df[forecast_df["sku"] == sku]
            forecast_total = sku_fcst["forecast"].sum() if len(sku_fcst) > 0 else mean_demand * 30
            forecast_mean = sku_fcst["forecast"].mean() if len(sku_fcst) > 0 else mean_demand

            # Risk flag
            if days_to_stockout < lt:
                risk = "critical"
            elif days_to_stockout < lt * 2:
                risk = "high"
            elif days_to_stockout < lt * 3:
                risk = "medium"
            else:
                risk = "low"

            rows.append({
                "sku": sku,
                "mean_daily_demand": round(mean_demand, 1),
                "demand_std": round(std_demand, 1),
                "safety_stock": round(safety_stock, 0),
                "reorder_point": round(reorder_point, 0),
                "reorder_qty": round(reorder_qty, 0),
                "current_stock": round(stock, 0),
                "days_to_stockout": round(days_to_stockout, 1),
                "forecast_30d_total": round(forecast_total, 0),
                "forecast_daily_avg": round(forecast_mean, 1),
                "reorder_needed": int(stock <= reorder_point),
                "stockout_risk": risk,
            })

        df = pd.DataFrame(rows).sort_values("days_to_stockout")
        logger.info(f"Computed inventory for {len(df)} SKUs; "
                    f"{df['reorder_needed'].sum()} need reorder")
        return df

    def compute_stockout_timeline(
        self, historical_df: pd.DataFrame, forecast_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Project daily stock level for each SKU over forecast horizon."""
        inv = self.compute(historical_df, forecast_df)
        rows = []
        for _, inv_row in inv.iterrows():
            sku = inv_row["sku"]
            stock = float(inv_row["current_stock"])
            sku_fcst = forecast_df[forecast_df["sku"] == sku].sort_values("date")
            for _, fcst_row in sku_fcst.iterrows():
                stock = max(0.0, stock - fcst_row["forecast"])
                rows.append({
                    "date": fcst_row["date"],
                    "sku": sku,
                    "projected_stock": round(stock, 1),
                    "reorder_point": inv_row["reorder_point"],
                })
        return pd.DataFrame(rows)
