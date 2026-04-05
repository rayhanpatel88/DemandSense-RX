"""Synthetic multi-SKU retail demand generator with richer operational realism."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

CATEGORY_BLUEPRINTS = {
    "Electronics": {"base_price": 180, "elasticity": -1.35, "promo_lift": 0.28, "service_level": 0.97},
    "Apparel": {"base_price": 55, "elasticity": -0.95, "promo_lift": 0.22, "service_level": 0.94},
    "Grocery": {"base_price": 14, "elasticity": -0.45, "promo_lift": 0.12, "service_level": 0.985},
    "Home": {"base_price": 72, "elasticity": -0.75, "promo_lift": 0.18, "service_level": 0.95},
    "Sports": {"base_price": 88, "elasticity": -0.85, "promo_lift": 0.2, "service_level": 0.95},
}

FAMILIES = {
    "Electronics": ["Mobility", "Audio", "Accessories"],
    "Apparel": ["Basics", "Outerwear", "Footwear"],
    "Grocery": ["DryGoods", "Snacks", "Beverages"],
    "Home": ["Kitchen", "Storage", "Decor"],
    "Sports": ["Fitness", "Outdoor", "Recovery"],
}


def generate_synthetic_data(config: dict) -> pd.DataFrame:
    """Generate realistic synthetic demand with hierarchy, price, promos, and stock behavior."""
    cfg = config["data"]
    rng = np.random.default_rng(cfg.get("seed", 42))
    dates = pd.date_range(cfg["start_date"], cfg["end_date"], freq="D")
    n_days = len(dates)
    n_skus = int(cfg["n_skus"])

    records: list[dict] = []
    sku_counter = 1
    category_names = list(CATEGORY_BLUEPRINTS.keys())

    for sku_idx in range(n_skus):
        category = category_names[sku_idx % len(category_names)]
        family = FAMILIES[category][sku_idx % len(FAMILIES[category])]
        blueprint = CATEGORY_BLUEPRINTS[category]
        sku = f"SKU_{sku_counter:03d}"
        sku_counter += 1

        base_demand = rng.uniform(35, 240)
        trend = rng.uniform(-0.015, 0.045)
        weekly_amp = rng.uniform(0.08, 0.28)
        annual_amp = rng.uniform(0.04, 0.22)
        monthly_amp = rng.uniform(0.02, 0.12)
        noise_scale = rng.uniform(0.05, 0.16)
        lead_time_days = int(rng.integers(3, 11))
        unit_cost = blueprint["base_price"] * rng.uniform(0.35, 0.62)
        base_price = blueprint["base_price"] * rng.uniform(0.55, 1.35)
        elasticity = blueprint["elasticity"] * rng.uniform(0.75, 1.15)
        promo_lift = blueprint["promo_lift"] * rng.uniform(0.85, 1.3)

        inventory_position = base_demand * rng.uniform(18, 28)
        open_purchase_orders: list[tuple[int, float]] = []
        reorder_trigger = base_demand * lead_time_days * rng.uniform(1.2, 1.8)
        target_cover = lead_time_days + int(rng.integers(8, 18))

        for day_idx, date in enumerate(dates):
            dow = date.dayofweek
            month = date.month
            t = day_idx

            weekly_factor = 1 + weekly_amp * np.sin((2 * np.pi * dow / 7) + rng.uniform(-0.2, 0.2))
            monthly_factor = 1 + monthly_amp * np.sin((2 * np.pi * date.day / 30.5) + 0.8)
            annual_factor = 1 + annual_amp * np.sin((2 * np.pi * t / 365.25) + (sku_idx / 5))
            trend_factor = 1 + trend * (t / max(n_days, 1))

            promo = 1 if rng.random() < (0.03 + 0.02 * (dow in [4, 5])) else 0
            if month in [11, 12]:
                promo = max(promo, int(rng.random() < 0.12))

            seasonal_price = base_price * (1 + 0.02 * np.cos(2 * np.pi * dow / 7))
            price = seasonal_price * (0.88 if promo else 1.0) * rng.uniform(0.97, 1.03)
            price_index = price / max(base_price, 1.0)

            category_season = 1.08 if category == "Sports" and month in [5, 6, 7] else 1.0
            category_season *= 1.12 if category == "Electronics" and month in [11, 12] else 1.0

            expected = base_demand * weekly_factor * monthly_factor * annual_factor * trend_factor * category_season
            expected *= (1 + promo_lift * promo)
            expected *= max(price_index ** elasticity, 0.35)
            expected += rng.normal(0, base_demand * noise_scale)
            demand = max(int(round(expected)), 0)

            received_today = 0.0
            remaining_orders: list[tuple[int, float]] = []
            for eta, qty in open_purchase_orders:
                if eta == t:
                    inventory_position += qty
                    received_today += qty
                else:
                    remaining_orders.append((eta, qty))
            open_purchase_orders = remaining_orders

            realised_sales = min(demand, max(inventory_position, 0))
            lost_sales = max(demand - realised_sales, 0)
            inventory_position = max(inventory_position - realised_sales, 0)

            if inventory_position <= reorder_trigger:
                replenishment_qty = max(base_demand * target_cover - inventory_position, base_demand * lead_time_days)
                replenishment_qty *= rng.uniform(0.95, 1.15)
                arrival = t + lead_time_days + int(rng.integers(0, 3))
                open_purchase_orders.append((arrival, replenishment_qty))
                reorder_trigger = base_demand * lead_time_days * rng.uniform(1.15, 1.65)

            records.append(
                {
                    "date": date,
                    "sku": sku,
                    "category": category,
                    "family": family,
                    "demand": realised_sales,
                    "unconstrained_demand": demand,
                    "lost_sales": lost_sales,
                    "price": round(float(price), 2),
                    "promotion": promo,
                    "inventory_position": round(float(inventory_position), 1),
                    "received_inventory": round(float(received_today), 1),
                    "lead_time_days": lead_time_days,
                    "unit_cost": round(float(unit_cost), 2),
                    "service_level_target": blueprint["service_level"],
                }
            )

    df = pd.DataFrame(records).sort_values(["sku", "date"]).reset_index(drop=True)
    logger.info("Generated %s synthetic demand rows across %s SKUs", len(df), n_skus)
    return df
