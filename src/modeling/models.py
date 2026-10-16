"""
Multi-Horizon Predictive Models Module
Trains dedicated gradient boosted tree models (LightGBM) for horizons n+0 through n+5.
Evaluates out-of-sample statistical metrics and feature importance.
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor

from config import config
from src.features.feature_pipeline import FeaturePipeline

logger = logging.getLogger(__name__)


class MultiHorizonPredictor:
    """
    Manages separate predictive models for horizons n+0, n+1, n+2, n+3, n+4, and n+5.
    Forecasts target returns, reconstructs price forecasts, and evaluates out-of-sample accuracy.
    """

    def __init__(self, horizons: Optional[List[int]] = None, random_state: int = 42):
        self.horizons = horizons if horizons is not None else config.forecast_horizons
        self.random_state = random_state
        self.feature_cols = FeaturePipeline.get_feature_columns()
        self.models: Dict[int, LGBMRegressor] = {}
        self.feature_importances_: Dict[int, pd.DataFrame] = {}

    def _create_base_model(self) -> LGBMRegressor:
        """Instantiates a conservative, regularized LightGBM regressor to prevent overfitting."""
        return LGBMRegressor(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=self.random_state,
            verbose=-1,
            n_jobs=-1
        )

    def fit_and_evaluate_walk_forward(
        self,
        df: pd.DataFrame,
        cv_splits: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, Any]:
        """
        Executes purged walk-forward cross validation across all horizons.
        Accumulates strictly out-of-sample predictions, metrics, and feature importances.
        """
        results_by_horizon = {}
        all_oos_predictions = []

        X = df[self.feature_cols].copy()
        current_prices = df["close"].values

        for h in self.horizons:
            target_col = f"target_ret_{h}"
            price_col = f"target_price_{h}"
            y = df[target_col].values
            y_price = df[price_col].values

            oos_preds_ret = np.full(len(df), np.nan)
            oos_preds_price = np.full(len(df), np.nan)
            fold_metrics = []
            importances_list = []

            for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
                X_train, y_train = X.iloc[train_idx], y[train_idx]
                X_test, y_test = X.iloc[test_idx], y[test_idx]
                test_prices = current_prices[test_idx]

                # Train model
                model = self._create_base_model()
                model.fit(X_train, y_train)

                # Predict return
                pred_ret = model.predict(X_test)
                # Reconstruct price forecast: P_{n+h} = P_n * (1 + pred_ret)
                pred_price = test_prices * (1.0 + pred_ret)

                oos_preds_ret[test_idx] = pred_ret
                oos_preds_price[test_idx] = pred_price

                # Store importance
                imp_df = pd.DataFrame({
                    "feature": self.feature_cols,
                    "importance": model.feature_importances_
                })
                importances_list.append(imp_df)

                # Fold evaluation
                mae_ret = mean_absolute_error(y_test, pred_ret)
                rmse_ret = np.sqrt(mean_squared_error(y_test, pred_ret))
                mae_price = mean_absolute_error(y_price[test_idx], pred_price)
                rmse_price = np.sqrt(mean_squared_error(y_price[test_idx], pred_price))
                hit_rate = np.mean(np.sign(pred_ret) == np.sign(y_test))
                ic, _ = pearsonr(pred_ret, y_test) if len(y_test) > 2 else (0.0, 1.0)
                rank_ic, _ = spearmanr(pred_ret, y_test) if len(y_test) > 2 else (0.0, 1.0)

                fold_metrics.append({
                    "fold": fold_idx,
                    "mae_ret": mae_ret,
                    "rmse_ret": rmse_ret,
                    "mae_price": mae_price,
                    "rmse_price": rmse_price,
                    "hit_rate": hit_rate,
                    "ic": ic if not np.isnan(ic) else 0.0,
                    "rank_ic": rank_ic if not np.isnan(rank_ic) else 0.0
                })

            # Fit final model on full dataset for live forecasting
            final_model = self._create_base_model()
            final_model.fit(X, y)
            self.models[h] = final_model

            # Aggregate feature importances
            avg_imp = pd.concat(importances_list).groupby("feature").mean().reset_index()
            avg_imp = avg_imp.sort_values("importance", ascending=False).reset_index(drop=True)
            self.feature_importances_[h] = avg_imp

            # Aggregate summary metrics
            df_fold_metrics = pd.DataFrame(fold_metrics)
            results_by_horizon[h] = {
                "horizon": f"n+{h}",
                "mean_mae_price": float(df_fold_metrics["mae_price"].mean()),
                "mean_rmse_price": float(df_fold_metrics["rmse_price"].mean()),
                "mean_mae_return": float(df_fold_metrics["mae_ret"].mean()),
                "mean_rmse_return": float(df_fold_metrics["rmse_ret"].mean()),
                "directional_accuracy": float(df_fold_metrics["hit_rate"].mean()),
                "mean_ic": float(df_fold_metrics["ic"].mean()),
                "mean_rank_ic": float(df_fold_metrics["rank_ic"].mean()),
                "oos_pred_ret": oos_preds_ret,
                "oos_pred_price": oos_preds_price,
                "actual_ret": y,
                "actual_price": y_price
            }

        return results_by_horizon

    def predict_horizons_latest(self, latest_row: pd.Series, current_price: float) -> Dict[str, Dict[str, float]]:
        """
        Generates predictions for n+0 through n+5 for the most recent observation.
        """
        X_latest = pd.DataFrame([latest_row[self.feature_cols]])
        forecasts = {}

        for h in self.horizons:
            model = self.models.get(h)
            if model is None:
                continue
            pred_ret = float(model.predict(X_latest)[0])
            pred_price = float(current_price * (1.0 + pred_ret))
            forecasts[f"n+{h}"] = {
                "predicted_return_pct": round(pred_ret * 100, 3),
                "predicted_price": round(pred_price, 2)
            }

        return forecasts
