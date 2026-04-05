"""Synthetic multi-SKU demand data generator."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.utils.logger import get_logger

logger = get_logger(__name__)

CATEGORIES = ["Electronics", "Apparel", "Grocery", "Home", "Sports"]

SKU_PROFILES = [
    # (base_demand, trend, weekly_amp, monthly_amp, noise, promo_prob, price_base)
    (100, 0.02, 30, 15, 0.10, 0.05, 49.99),
    (250, 0.05, 60, 30, 0.12, 0.08, 19.99),
    (50,  0.00, 10, 5,  0.20, 0.03, 129.99),
    (180, 0.01, 45, 20, 0.15, 0.06, 34.99),
    (75,  0.03, 20, 10, 0.18, 0.04, 79.99),
    (300, 0.08, 80, 40, 0.08, 0.10, 9.99),
    (120, -0.01, 35, 18, 0.22, 0.07, 24.99),
    (90,  0.04, 25, 12, 0.14, 0.05, 59.99),
    (200, 0.02, 50, 25, 0.11, 0.09, 14.99),
    (60,  0.00, 15, 8,  0.25, 0.04, 99.99),
    (150, 0.06, 40, 22, 0.13, 0.07, 29.99),
    (80,  0.01, 22, 11, 0.19, 0.05, 69.99),
    (220, 0.03, 55, 28, 0.10, 0.08, 17.99),
    (45,  0.00, 12, 6,  0.28, 0.03, 149.99),
    (170, 0.02, 42, 21, 0.16, 0.06, 39.99),
    (110, 0.04, 30, 15, 0.12, 0.07, 44.99),
    (280, 0.07, 70, 35, 0.09, 0.11, 12.99),
    (95,  0.01, 28, 14, 0.17, 0.05, 54.99),
    (130, 0.03, 35, 18, 0.14, 0.06, 32.99),
    (65,  0.00, 18, 9,  0.23, 0.04, 89.99),
]


def generate_synthetic_data(config: dict) -> pd.DataFrame:
    """Generate synthetic multi-SKU demand data."""
    cfg = config["data"]
    n_skus = cfg["n_skus"]
    start = pd.Timestamp(cfg["start_date"])
    end = pd.Timestamp(cfg["end_date"])
    seed = cfg.get("seed", 42)
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start, end, freq="D")
    n_days = len(dates)
    t = np.arange(n_days)

    records = []
    for sku_idx in range(n_skus):
        sku_id = f"SKU_{sku_idx+1:03d}"
        category = CATEGORIES[sku_idx % len(CATEGORIES)]
        profile = SKU_PROFILES[sku_idx % len(SKU_PROFILES)]
        base, trend, w_amp, m_amp, noise_frac, promo_prob, price_base = profile

        # Trend component
        trend_component = base + trend * t

        # Weekly seasonality (peak mid-week / weekend effect)
        dow = np.array([d.dayofweek for d in dates])
        weekly = w_amp * np.sin(2 * np.pi * dow / 7 + rng.uniform(0, np.pi))

        # Monthly / annual seasonality
        monthly = m_amp * np.sin(2 * np.pi * t / 365.25 + rng.uniform(0, 2 * np.pi))

        # Promotions
        promo_flags = rng.random(n_days) < promo_prob
        # Smooth out single-day promos into runs of 3–7 days
        promo_smooth = np.zeros(n_days, dtype=float)
        i = 0
        while i < n_days:
            if promo_flags[i]:
                run = int(rng.integers(3, 8))
                promo_smooth[i:i + run] = 1.0
                i += run
            else:
                i += 1
        promo_boost = promo_smooth * base * rng.uniform(0.20, 0.50)

        # Price (slightly lower during promos)
        price = np.full(n_days, price_base)
        price[promo_smooth > 0] *= rng.uniform(0.80, 0.95, size=int(promo_smooth.sum()))

        # Noise
        noise = rng.normal(0, noise_frac * base, n_days)

        demand = trend_component + weekly + monthly + promo_boost + noise
        demand = np.clip(demand, 0, None).round().astype(int)

        for j, date in enumerate(dates):
            records.append({
                "date": date,
                "sku": sku_id,
                "category": category,
                "demand": demand[j],
                "price": round(float(price[j]), 2),
                "promotion": int(promo_smooth[j]),
            })

    df = pd.DataFrame(records)
    df = df.sort_values(["sku", "date"]).reset_index(drop=True)
    logger.info(f"Generated {len(df)} rows for {n_skus} SKUs over {n_days} days")
    return df
