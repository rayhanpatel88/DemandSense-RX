"""Realistic pharmaceutical and medical-supply warehouse demand generator.

Modelled on real-world distribution-centre dynamics:
  - Product catalogue drawn from common Rx, OTC, IV-supply, surgical and
    durable-medical-equipment (DME) lines.
  - Seasonal demand patterns match published pharmacy consumption data
    (flu season Oct–Feb, allergy Apr–Jun, summer OTC Jul–Aug, holiday
    dispensing spikes Nov–Dec).
  - ABC inventory classification (A=20% SKUs / 80% volume, etc.).
  - Lead times reflect actual supplier tiers (generic Rx 3–5d, branded Rx
    7–14d, IV/infusion 2–4d, DME 10–21d).
  - Promotion cadence models formulary changes and GPO contract events.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Product catalogue
# Each entry: (display_name, category, family, abc_class,
#              base_daily_units, price_usd, lead_time_days,
#              service_level_target, elasticity, promo_lift_pct)
# ---------------------------------------------------------------------------
PRODUCT_CATALOGUE = [
    # --- Rx / Branded ---
    ("Lisinopril 10mg", "Rx_Generic", "Cardiovascular", "A", 95, 0.42, 4, 0.985, -0.30, 0.06),
    ("Metformin 500mg", "Rx_Generic", "Metabolic", "A", 88, 0.38, 4, 0.985, -0.28, 0.05),
    ("Atorvastatin 20mg", "Rx_Generic", "Cardiovascular", "A", 82, 0.55, 4, 0.985, -0.32, 0.07),
    ("Amoxicillin 500mg", "Rx_Generic", "Antibiotic", "A", 76, 0.61, 4, 0.980, -0.40, 0.12),
    ("Omeprazole 20mg", "Rx_Generic", "GI", "A", 70, 0.49, 4, 0.980, -0.35, 0.08),
    ("Sertraline 50mg", "Rx_Generic", "CNS", "B", 58, 0.72, 5, 0.975, -0.25, 0.04),
    ("Amlodipine 5mg", "Rx_Generic", "Cardiovascular", "B", 54, 0.44, 4, 0.975, -0.30, 0.06),
    ("Levothyroxine 50mcg", "Rx_Generic", "Endocrine", "B", 50, 0.53, 5, 0.985, -0.20, 0.03),
    ("Azithromycin 250mg", "Rx_Generic", "Antibiotic", "B", 46, 0.88, 5, 0.975, -0.45, 0.15),
    ("Hydrochlorothiazide 25mg", "Rx_Generic", "Cardiovascular", "B", 44, 0.36, 4, 0.975, -0.28, 0.05),
    # --- OTC / Consumer ---
    ("Ibuprofen 200mg 100ct", "OTC", "Pain_Relief", "A", 110, 8.99, 3, 0.975, -0.65, 0.22),
    ("Acetaminophen 500mg 100ct", "OTC", "Pain_Relief", "A", 105, 7.49, 3, 0.975, -0.60, 0.20),
    ("Diphenhydramine 25mg 48ct", "OTC", "Allergy", "A", 72, 9.49, 3, 0.970, -0.70, 0.28),
    ("Loratadine 10mg 45ct", "OTC", "Allergy", "A", 68, 14.99, 3, 0.970, -0.68, 0.25),
    ("Cetirizine 10mg 45ct", "OTC", "Allergy", "B", 62, 13.49, 3, 0.970, -0.65, 0.24),
    ("Omeprazole OTC 20mg 42ct", "OTC", "GI", "B", 55, 22.99, 3, 0.965, -0.55, 0.18),
    ("Famotidine 20mg 50ct", "OTC", "GI", "B", 48, 11.99, 3, 0.965, -0.50, 0.16),
    ("NyQuil 12oz", "OTC", "Cold_Flu", "A", 90, 12.99, 3, 0.975, -0.72, 0.30),
    ("DayQuil 16ct", "OTC", "Cold_Flu", "A", 85, 11.49, 3, 0.975, -0.70, 0.28),
    ("Mucinex 600mg 20ct", "OTC", "Cold_Flu", "A", 78, 14.99, 3, 0.970, -0.68, 0.26),
    # --- IV / Infusion ---
    ("NS 0.9% 1L Bag", "IV_Supply", "Crystalloid", "A", 140, 3.50, 2, 0.995, -0.10, 0.02),
    ("D5W 1L Bag", "IV_Supply", "Crystalloid", "A", 95, 3.80, 2, 0.995, -0.10, 0.02),
    ("LR 1L Bag", "IV_Supply", "Crystalloid", "A", 88, 3.70, 2, 0.995, -0.10, 0.02),
    ("Vancomycin 500mg Vial", "IV_Supply", "Antibiotic_IV", "B", 38, 18.50, 3, 0.990, -0.15, 0.03),
    ("Piperacillin-Tazobactam 4.5g", "IV_Supply", "Antibiotic_IV", "B", 32, 24.00, 3, 0.990, -0.12, 0.02),
    # --- Surgical / Disposables ---
    ("Nitrile Exam Gloves M 100ct", "Surgical", "PPE", "A", 200, 12.50, 5, 0.990, -0.35, 0.10),
    ("Surgical Mask ASTM II 50ct", "Surgical", "PPE", "A", 180, 9.99, 5, 0.990, -0.30, 0.08),
    ("3mL Syringe 21G 100ct", "Surgical", "Injection_Supply", "A", 160, 18.00, 4, 0.990, -0.20, 0.05),
    ("Gauze 4x4 Sterile 10ct", "Surgical", "Wound_Care", "B", 90, 5.25, 4, 0.985, -0.25, 0.06),
    ("IV Catheter 20G 50ct", "Surgical", "Injection_Supply", "B", 75, 32.00, 4, 0.990, -0.18, 0.04),
]

# Seasonal profiles by category family (month index 1-12 → multiplier)
# Based on published pharmacy dispensing and hospital supply consumption studies
_SEASONAL = {
    "Antibiotic":      [1.05, 1.10, 1.05, 0.95, 0.90, 0.88, 0.88, 0.90, 0.95, 1.05, 1.15, 1.15],
    "Antibiotic_IV":   [1.08, 1.12, 1.06, 0.96, 0.90, 0.88, 0.88, 0.90, 0.96, 1.06, 1.16, 1.14],
    "Cold_Flu":        [1.40, 1.30, 1.10, 0.85, 0.75, 0.70, 0.72, 0.75, 0.90, 1.10, 1.35, 1.50],
    "Allergy":         [0.75, 0.78, 0.90, 1.30, 1.40, 1.35, 1.15, 1.05, 0.95, 0.80, 0.72, 0.72],
    "Pain_Relief":     [1.10, 1.05, 1.00, 0.98, 0.97, 0.95, 1.00, 1.02, 1.00, 1.00, 1.08, 1.15],
    "Crystalloid":     [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.05, 1.05, 1.00, 1.00, 1.00, 0.95],
    "PPE":             [1.10, 1.08, 1.02, 0.98, 0.96, 0.95, 0.95, 0.96, 1.00, 1.02, 1.05, 1.08],
    "_default":        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
}

# Tier-based reorder and cover parameters (ABC class)
_ABC_COVER = {"A": (18, 28), "B": (14, 22), "C": (10, 16)}
_ABC_REORDER_MULT = {"A": (1.25, 1.65), "B": (1.15, 1.55), "C": (1.05, 1.45)}


def generate_synthetic_data(config: dict) -> pd.DataFrame:
    """Generate realistic pharmaceutical warehouse demand.

    Uses the product catalogue above; ignores ``n_skus`` from config
    (catalogue size takes precedence) but respects ``start_date``,
    ``end_date``, and ``seed``.
    """
    cfg = config["data"]
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    dates = pd.date_range(cfg["start_date"], cfg["end_date"], freq="D")
    n_days = len(dates)

    records: list[dict] = []

    for product in PRODUCT_CATALOGUE:
        (name, category, family, abc_class,
         base_daily, price, lead_time, svc_level,
         elasticity, promo_lift) = product

        sku = "SKU-" + name.replace(" ", "_").replace("/", "-")[:18]

        # Per-SKU random variation around catalogue values
        base_demand = base_daily * rng.uniform(0.82, 1.18)
        trend = rng.uniform(-0.008, 0.025)          # slight growth bias
        weekly_amp = rng.uniform(0.04, 0.14)
        noise_scale = rng.uniform(0.04, 0.11)
        unit_cost = price * rng.uniform(0.42, 0.68)
        base_price = price * rng.uniform(0.90, 1.10)
        lt = lead_time + int(rng.integers(0, 3))    # supplier jitter

        cover_lo, cover_hi = _ABC_COVER[abc_class]
        ro_lo, ro_hi = _ABC_REORDER_MULT[abc_class]
        inventory_position = base_demand * rng.uniform(cover_lo, cover_hi)
        reorder_trigger = base_demand * lt * rng.uniform(ro_lo, ro_hi)
        target_cover = lt + int(rng.integers(cover_lo, cover_hi))

        seasonal_profile = _SEASONAL.get(family, _SEASONAL["_default"])
        open_purchase_orders: list[tuple[int, float]] = []

        for day_idx, date in enumerate(dates):
            dow = date.dayofweek
            month = date.month
            t = day_idx

            # --- Seasonal & trend ---
            season = seasonal_profile[month - 1]
            trend_factor = 1 + trend * (t / max(n_days, 1))
            weekly_factor = 1 + weekly_amp * np.sin((2 * np.pi * dow / 7) + rng.uniform(-0.15, 0.15))

            # --- Promotion cadence ---
            # GPO contract events cluster around quarter starts; cold/flu promos in Oct-Feb
            base_promo_p = 0.025
            if dow in [4, 5]:                   # weekend retail lift
                base_promo_p += 0.015
            if month in [10, 11, 12, 1, 2] and family == "Cold_Flu":
                base_promo_p += 0.045
            if month in [4, 5, 6] and family == "Allergy":
                base_promo_p += 0.040
            if month in [1, 4, 7, 10] and day_idx % 90 < 10:  # quarter-start formulary
                base_promo_p += 0.03
            promo = 1 if rng.random() < base_promo_p else 0

            price_today = base_price * (0.90 if promo else 1.0) * rng.uniform(0.98, 1.02)
            price_index = price_today / max(base_price, 1e-6)

            expected = (base_demand
                        * season
                        * trend_factor
                        * weekly_factor
                        * (1 + promo_lift * promo)
                        * max(price_index ** elasticity, 0.35))
            expected += rng.normal(0, base_demand * noise_scale)
            demand = max(int(round(expected)), 0)

            # --- Inventory simulation ---
            received_today = 0.0
            remaining: list[tuple[int, float]] = []
            for eta, qty in open_purchase_orders:
                if eta == t:
                    inventory_position += qty
                    received_today += qty
                else:
                    remaining.append((eta, qty))
            open_purchase_orders = remaining

            realised_sales = min(demand, max(inventory_position, 0))
            lost_sales = max(demand - realised_sales, 0)
            inventory_position = max(inventory_position - realised_sales, 0)

            if inventory_position <= reorder_trigger:
                replen_qty = max(base_demand * target_cover - inventory_position,
                                 base_demand * lt)
                replen_qty *= rng.uniform(0.94, 1.10)
                arrival = t + lt + int(rng.integers(0, 3))
                open_purchase_orders.append((arrival, replen_qty))
                reorder_trigger = base_demand * lt * rng.uniform(ro_lo, ro_hi)

            records.append({
                "date": date,
                "sku": sku,
                "category": category,
                "family": family,
                "demand": realised_sales,
                "unconstrained_demand": demand,
                "lost_sales": lost_sales,
                "price": round(float(price_today), 2),
                "promotion": promo,
                "inventory_position": round(float(inventory_position), 1),
                "received_inventory": round(float(received_today), 1),
                "lead_time_days": lt,
                "unit_cost": round(float(unit_cost), 2),
                "service_level_target": svc_level,
            })

    df = pd.DataFrame(records).sort_values(["sku", "date"]).reset_index(drop=True)
    logger.info(
        "Generated %s realistic warehouse demand rows across %s SKUs (%s categories)",
        len(df), df["sku"].nunique(), df["category"].nunique(),
    )
    return df
