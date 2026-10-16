"""
Tests for Walk-Forward CV Splitter and Multi-Horizon Predictor
"""

import pytest
import numpy as np
import pandas as pd

from src.modeling.walk_forward import PurgedWalkForwardCV
from src.modeling.models import MultiHorizonPredictor
from src.features.feature_pipeline import FeaturePipeline


def test_purged_walk_forward_cv():
    """Ensures embargo window prevents target leakage across folds."""
    df = pd.DataFrame({"val": range(100)})
    cv = PurgedWalkForwardCV(n_splits=3, embargo_periods=5, min_train_pct=0.4)
    splits = list(cv.split(df))

    assert len(splits) == 3
    for train_idx, test_idx in splits:
        # Check no overlap
        assert len(set(train_idx).intersection(set(test_idx))) == 0
        # Check embargo buffer
        assert test_idx[0] - train_idx[-1] > 5


def test_multi_horizon_predictor_fit_eval():
    """Tests training and walk-forward evaluation on synthetic feature matrix."""
    n_samples = 150
    feature_cols = FeaturePipeline.get_feature_columns()

    # Synthetic feature values
    np.random.seed(42)
    data = {col: np.random.normal(0, 1, n_samples) for col in feature_cols}
    data["close"] = np.linspace(200, 250, n_samples) + np.random.normal(0, 2, n_samples)
    data["timestamp_utc"] = [f"2026-01-01T{i:02d}:00:00Z" for i in range(n_samples)]

    # Targets
    for h in [0, 1, 2, 3, 4, 5]:
        data[f"target_ret_{h}"] = np.random.normal(0.001 * (h + 1), 0.01, n_samples)
        data[f"target_price_{h}"] = data["close"] * (1 + data[f"target_ret_{h}"])

    df = pd.DataFrame(data)

    cv = PurgedWalkForwardCV(n_splits=2, embargo_periods=3, min_train_pct=0.5)
    splits = list(cv.split(df))

    predictor = MultiHorizonPredictor(horizons=[0, 1, 2])
    results = predictor.fit_and_evaluate_walk_forward(df, splits)

    assert 0 in results
    assert 1 in results
    assert 2 in results
    assert "directional_accuracy" in results[1]
    assert "mean_mae_price" in results[1]

    # Test latest forecast
    latest = predictor.predict_horizons_latest(df.iloc[-1], current_price=250.0)
    assert "n+0" in latest
    assert "n+1" in latest
    assert "predicted_price" in latest["n+1"]
