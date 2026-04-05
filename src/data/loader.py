"""Data loading and validation utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_COLS = {"date", "sku", "demand"}
OPTIONAL_DEFAULTS = {
    "price": 1.0,
    "promotion": 0,
    "category": "Unknown",
    "family": "Unknown",
    "lead_time_days": 7,
    "inventory_position": 0.0,
    "received_inventory": 0.0,
    "unit_cost": 0.0,
    "service_level_target": 0.95,
}


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in frame.columns:
            frame[col] = default

    frame["demand"] = frame["demand"].fillna(0).clip(lower=0)
    frame["promotion"] = frame["promotion"].fillna(0).astype(int)
    frame = _fill_date_gaps(frame)
    logger.info("Validated %s rows across %s SKUs", len(frame), frame["sku"].nunique())
    return frame


def load_or_generate(config: dict) -> pd.DataFrame:
    from src.data.generator import generate_synthetic_data

    root = Path(__file__).resolve().parents[2]
    raw_dir = root / "data" / "raw"
    parquet_path = raw_dir / "demand.parquet"
    csv_path = raw_dir / "demand.csv"

    if parquet_path.exists():
        logger.info("Loading demand data from %s", parquet_path)
        data = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        logger.info("Loading demand data from %s", csv_path)
        data = pd.read_csv(csv_path)
    else:
        logger.info("Generating synthetic data because no raw dataset was found")
        data = generate_synthetic_data(config)
        raw_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(csv_path, index=False)
        try:
            data.to_parquet(parquet_path, index=False)
        except Exception as exc:
            logger.warning("Parquet export skipped: %s", exc)
    return validate_data(data)


def _fill_date_gaps(df: pd.DataFrame) -> pd.DataFrame:
    all_frames: list[pd.DataFrame] = []
    for sku, group in df.groupby("sku"):
        group = group.sort_values("date")
        date_range = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
        filled = group.set_index("date").reindex(date_range).reset_index()
        filled = filled.rename(columns={"index": "date"})
        filled["sku"] = sku
        for col, default in OPTIONAL_DEFAULTS.items():
            if col in {"price", "inventory_position"}:
                filled[col] = filled[col].ffill().fillna(default)
            elif col in {"category", "family"}:
                filled[col] = filled[col].ffill().bfill().fillna(default)
            else:
                filled[col] = filled[col].fillna(default)
        filled["demand"] = filled["demand"].fillna(0).clip(lower=0)
        all_frames.append(filled)
    return pd.concat(all_frames, ignore_index=True)
