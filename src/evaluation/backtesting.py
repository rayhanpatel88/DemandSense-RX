"""Rolling window backtesting engine."""

import pandas as pd
import numpy as np
from src.evaluation.metrics import compute_all_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RollingBacktester:
    """Rolling-origin backtester for demand forecasting models."""

    def __init__(self, n_splits: int = 4, test_size: int = 30, gap: int = 0):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def run(self, df: pd.DataFrame, models: dict,
            feature_cols: list, config: dict) -> dict:
        """
        Run backtesting for all provided models.

        Parameters
        ----------
        df : featured DataFrame (output of create_features)
        models : dict of {model_name: BaseForecaster}
        feature_cols : list of feature column names
        config : config dict

        Returns
        -------
        dict with keys:
          - 'summary': overall metrics per model
          - 'by_sku': metrics per SKU per model
          - 'predictions': all fold predictions
        """
        df = df.dropna(subset=feature_cols + ["demand"]).copy()
        dates = sorted(df["date"].unique())
        n_dates = len(dates)

        # Build fold splits
        folds = self._build_folds(dates, n_dates)
        logger.info(f"Running {len(folds)} folds × {len(models)} models")

        all_preds = []
        model_metrics = {m: [] for m in models}

        for fold_idx, (train_dates, test_dates) in enumerate(folds):
            train = df[df["date"].isin(train_dates)].copy()
            test = df[df["date"].isin(test_dates)].copy()

            if len(train) == 0 or len(test) == 0:
                continue

            for model_name, model in models.items():
                # Re-fit on training fold
                model.fit(train, feature_cols)
                preds = model.predict(test, feature_cols)

                test_copy = test.copy()
                test_copy["forecast"] = np.clip(preds, 0, None)
                test_copy["model"] = model_name
                test_copy["fold"] = fold_idx
                all_preds.append(test_copy[["date", "sku", "demand", "forecast", "model", "fold"]])

                metrics = compute_all_metrics(test["demand"].values, preds)
                metrics["fold"] = fold_idx
                model_metrics[model_name].append(metrics)

        pred_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

        # Aggregate summary metrics
        summary_rows = []
        for model_name, fold_metrics in model_metrics.items():
            if not fold_metrics:
                continue
            fold_df = pd.DataFrame(fold_metrics)
            row = {"model": model_name}
            for col in ["MAE", "RMSE", "MAPE", "WAPE"]:
                row[col] = round(fold_df[col].mean(), 3)
            summary_rows.append(row)

        summary = pd.DataFrame(summary_rows).set_index("model") if summary_rows else pd.DataFrame()

        # Per-SKU metrics
        by_sku_rows = []
        for model_name in models:
            model_preds = pred_df[pred_df["model"] == model_name] if len(pred_df) else pd.DataFrame()
            if len(model_preds) == 0:
                continue
            for sku, group in model_preds.groupby("sku"):
                m = compute_all_metrics(group["demand"].values, group["forecast"].values)
                m["model"] = model_name
                m["sku"] = sku
                by_sku_rows.append(m)

        by_sku = pd.DataFrame(by_sku_rows) if by_sku_rows else pd.DataFrame()

        logger.info("Backtesting complete")
        return {"summary": summary, "by_sku": by_sku, "predictions": pred_df}

    def _build_folds(self, dates: list, n_dates: int) -> list:
        folds = []
        step = max(self.test_size, (n_dates - self.test_size * self.n_splits) // self.n_splits)
        min_train = max(60, n_dates // 4)

        for i in range(self.n_splits):
            test_end_idx = n_dates - i * self.test_size
            test_start_idx = test_end_idx - self.test_size
            train_end_idx = test_start_idx - self.gap

            if train_end_idx < min_train:
                break

            train_dates = set(dates[:train_end_idx])
            test_dates = set(dates[test_start_idx:test_end_idx])
            folds.append((train_dates, test_dates))

        return list(reversed(folds))
