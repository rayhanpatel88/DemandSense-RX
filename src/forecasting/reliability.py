"""Forecast reliability scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


class ReliabilityScorer:
    """Blend backtest error, interval quality, and stability into a 0-1 score."""

    def __init__(self, config: dict):
        rel_cfg = config.get("reliability", {})
        self.high_threshold = float(rel_cfg.get("high_threshold", 0.72))
        self.low_threshold = float(rel_cfg.get("low_threshold", 0.45))

    def score(self, predictions: pd.DataFrame) -> pd.DataFrame:
        if predictions.empty:
            return pd.DataFrame(columns=["sku", "reliability_score", "reliability_category", "reliability_explanation"])

        scored = predictions.copy().sort_values(["sku", "date"])
        scored["abs_error"] = (scored["demand"] - scored["forecast"]).abs()
        denom = scored["demand"].abs().clip(lower=1.0)
        scored["ape"] = scored["abs_error"] / denom
        scored["interval_width"] = (scored["upper"] - scored["lower"]).clip(lower=0.0)
        scored["covered"] = (
            (scored["demand"] >= scored["lower"]) & (scored["demand"] <= scored["upper"])
        ).astype(float)

        scored["interval_width_ratio"] = scored["interval_width"] / scored["forecast"].clip(lower=1.0)
        stability = scored.groupby("sku")["forecast"].pct_change().abs().replace([np.inf, -np.inf], np.nan)
        scored["forecast_jump"] = stability.fillna(0.0)

        grouped = scored.groupby("sku")
        summary = grouped.agg(
            historical_wape=("abs_error", "sum"),
            demand_total=("demand", "sum"),
            interval_width_ratio=("interval_width_ratio", "mean"),
            coverage=("covered", "mean"),
            recent_stability=("forecast_jump", lambda x: float(np.mean(x.tail(7)))),
        ).reset_index()
        summary["historical_wape"] = summary["historical_wape"] / summary["demand_total"].clip(lower=1.0)
        summary["score_error"] = (1.0 - summary["historical_wape"]).clip(0.0, 1.0)
        summary["score_width"] = (1.0 - summary["interval_width_ratio"]).clip(0.0, 1.0)
        summary["score_coverage"] = (1.0 - (summary["coverage"] - 0.8).abs() / 0.8).clip(0.0, 1.0)
        summary["score_stability"] = (1.0 - summary["recent_stability"]).clip(0.0, 1.0)
        summary["reliability_score"] = (
            0.45 * summary["score_error"]
            + 0.20 * summary["score_width"]
            + 0.20 * summary["score_coverage"]
            + 0.15 * summary["score_stability"]
        ).clip(0.0, 1.0)
        summary["reliability_category"] = summary["reliability_score"].map(self._category)
        summary["reliability_explanation"] = summary.apply(self._explanation, axis=1)
        return summary[
            [
                "sku",
                "historical_wape",
                "coverage",
                "interval_width_ratio",
                "recent_stability",
                "reliability_score",
                "reliability_category",
                "reliability_explanation",
            ]
        ]

    def _category(self, score: float) -> str:
        if score >= self.high_threshold:
            return "High"
        if score >= self.low_threshold:
            return "Medium"
        return "Low"

    def _explanation(self, row: pd.Series) -> str:
        reasons: list[str] = []
        if row["historical_wape"] > 0.28:
            reasons.append("high historical forecast error")
        if row["interval_width_ratio"] > 0.55:
            reasons.append("wide prediction intervals")
        if row["coverage"] < 0.65:
            reasons.append("poor interval coverage")
        if row["recent_stability"] > 0.30:
            reasons.append("unstable recent predictions")
        if not reasons:
            reasons.append("stable backtest performance and controlled uncertainty")
        lead = row["reliability_category"]
        return f"{lead} reliability due to " + " and ".join(reasons)
