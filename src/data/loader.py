"""Data loading and validation utilities."""

import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_COLS = {"date", "sku", "demand"}
OPTIONAL_COLS = {"price", "promotion", "category"}


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean a demand DataFrame."""
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Fill optional columns with defaults
    if "price" not in df.columns:
        df["price"] = 0.0
    if "promotion" not in df.columns:
        df["promotion"] = 0
    if "category" not in df.columns:
        df["category"] = "Unknown"

    # Handle missing demand values
    n_missing = df["demand"].isna().sum()
    if n_missing > 0:
        logger.warning(f"Filling {n_missing} missing demand values via forward fill")
        df["demand"] = df.groupby("sku")["demand"].transform(
            lambda x: x.ffill().fillna(0)
        )

    # Clip negative demand to 0
    n_neg = (df["demand"] < 0).sum()
    if n_neg > 0:
        logger.warning(f"Clipping {n_neg} negative demand values to 0")
        df["demand"] = df["demand"].clip(lower=0)

    # Ensure date continuity per SKU (fill gaps)
    df = _fill_date_gaps(df)

    logger.info(f"Validated data: {len(df)} rows, {df['sku'].nunique()} SKUs")
    return df


def _fill_date_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing dates within each SKU's time range."""
    all_dfs = []
    for sku, group in df.groupby("sku"):
        date_range = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
        group = group.set_index("date").reindex(date_range).reset_index()
        group.rename(columns={"index": "date"}, inplace=True, errors="ignore")
        group["sku"] = sku
        group["demand"] = group["demand"].fillna(0)
        group["promotion"] = group["promotion"].fillna(0).astype(int)
        group["price"] = group["price"].ffill()
        group["category"] = group["category"].ffill()
        all_dfs.append(group)
    return pd.concat(all_dfs, ignore_index=True)


def load_or_generate(config: dict) -> pd.DataFrame:
    """Load data from disk if available, otherwise generate synthetic data."""
    from src.data.generator import generate_synthetic_data

    raw_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "demand.parquet"

    if raw_path.exists():
        logger.info(f"Loading data from {raw_path}")
        df = pd.read_parquet(raw_path)
    else:
        logger.info("Generating synthetic data")
        df = generate_synthetic_data(config)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(raw_path, index=False)
        logger.info(f"Saved synthetic data to {raw_path}")

    return validate_data(df)
