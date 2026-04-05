"""Rolling-origin backtesting engine aligned with recursive inference."""

from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import compute_all_metrics
from src.features.engineer import create_features, get_feature_cols
from src.forecasting.recursive import RecursiveForecaster
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RollingBacktester:
    """Rolling-origin backtester that respects recursive multi-step forecasting."""

    def __init__(self, n_splits: int = 4, test_size: int = 30, gap: int = 0):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def run(self, raw_df: pd.DataFrame, models: dict, config: dict) -> dict:
        raw = raw_df.sort_values(["sku", "date"]).copy()
        dates = sorted(raw["date"].unique())
        folds = self._build_folds(dates)
        logger.info("Running %s rolling folds across %s models", len(folds), len(models))

        all_predictions: list[pd.DataFrame] = []
        summary_rows: list[dict] = []
        by_sku_rows: list[dict] = []

        for model_name, model in models.items():
            fold_metrics: list[dict] = []
            model_predictions: list[pd.DataFrame] = []

            for fold_idx, (train_dates, test_dates) in enumerate(folds):
                train_raw = raw[raw["date"].isin(train_dates)].copy()
                test_raw = raw[raw["date"].isin(test_dates)].copy()
                if train_raw.empty or test_raw.empty:
                    continue

                train_features = create_features(train_raw, config)
                feature_cols = get_feature_cols(train_features)
                train_model_df = train_features.dropna(subset=feature_cols + ["demand"]).copy()
                if train_model_df.empty:
                    continue
                model.fit(train_model_df, feature_cols)

                test_exogenous = test_raw[["date", "sku", "category", "family", "promotion", "price", "lead_time_days"]].copy()
                if hasattr(model, "recursive_forecast"):
                    pred_df = model.recursive_forecast(train_raw, test_exogenous)
                elif model_name == "LightGBM":
                    recursive = RecursiveForecaster(model)
                    forecast_result = recursive.forecast(train_features, test_exogenous, feature_cols)
                    pred_df = forecast_result.forecast_frame
                else:
                    test_features = create_features(pd.concat([train_raw, test_raw], ignore_index=True), config)
                    test_features = test_features[test_features["date"].isin(test_dates)].copy()
                    intervals = model.predict_with_intervals(test_features, feature_cols)
                    pred_df = test_features[["date", "sku", "category", "family", "promotion", "price"]].copy()
                    pred_df["forecast"] = intervals["forecast"].values
                    pred_df["lower"] = intervals["lower"].values
                    pred_df["upper"] = intervals["upper"].values

                merged = pred_df.merge(
                    test_raw[["date", "sku", "demand"]],
                    on=["date", "sku"],
                    how="left",
                )
                merged["fold"] = fold_idx
                merged["model"] = model_name
                model_predictions.append(merged)
                all_predictions.append(merged)

                metrics = compute_all_metrics(merged["demand"].values, merged["forecast"].values)
                metrics["fold"] = fold_idx
                fold_metrics.append(metrics)

            if fold_metrics:
                fold_df = pd.DataFrame(fold_metrics)
                summary_rows.append(
                    {
                        "model": model_name,
                        "MAE": round(fold_df["MAE"].mean(), 3),
                        "RMSE": round(fold_df["RMSE"].mean(), 3),
                        "MAPE": round(fold_df["MAPE"].mean(), 3),
                        "WAPE": round(fold_df["WAPE"].mean(), 3),
                    }
                )

            if model_predictions:
                pred_all = pd.concat(model_predictions, ignore_index=True)
                for sku, group in pred_all.groupby("sku"):
                    metrics = compute_all_metrics(group["demand"].values, group["forecast"].values)
                    metrics["model"] = model_name
                    metrics["sku"] = sku
                    by_sku_rows.append(metrics)

        predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
        summary = pd.DataFrame(summary_rows).set_index("model") if summary_rows else pd.DataFrame()
        by_sku = pd.DataFrame(by_sku_rows) if by_sku_rows else pd.DataFrame()
        return {"summary": summary, "by_sku": by_sku, "predictions": predictions}

    def _build_folds(self, dates: list[pd.Timestamp]) -> list[tuple[set, set]]:
        folds: list[tuple[set, set]] = []
        n_dates = len(dates)
        min_train = max(90, n_dates // 3)
        for idx in range(self.n_splits):
            test_end = n_dates - idx * self.test_size
            test_start = test_end - self.test_size
            train_end = test_start - self.gap
            if train_end < min_train or test_start < 0:
                break
            folds.append((set(dates[:train_end]), set(dates[test_start:test_end])))
        return list(reversed(folds))
