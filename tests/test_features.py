"""
Tests for Quantitative Feature Engineering and Multi-Horizon Alignment
"""

import pytest
import numpy as np
import pandas as pd

from src.features.price_features import PriceFeatureExtractor
from src.features.feature_pipeline import FeaturePipeline


def test_price_features():
    """Tests technical indicators derived exclusively from price."""
    extractor = PriceFeatureExtractor()

    dates = pd.date_range("2026-01-01", periods=60, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "timestamp_utc": [d.strftime("%Y-%m-%dT%H:%M:%SZ") for d in dates],
        "open": np.linspace(200, 250, 60),
        "high": np.linspace(202, 252, 60),
        "low": np.linspace(198, 248, 60),
        "close": np.linspace(201, 251, 60),
        "volume": np.random.uniform(1e5, 5e5, 60)
    })

    feats = extractor.extract_features(df)
    assert "ret_lag1" in feats.columns
    assert "rsi_14" in feats.columns
    assert "macd" in feats.columns
    assert "volatility_24" in feats.columns
    assert len(feats) == 60


def test_multi_horizon_targets():
    """Verifies targets for n+0 through n+5 are correctly constructed."""
    pipeline = FeaturePipeline()

    dates = pd.date_range("2026-01-01", periods=50, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "timestamp_utc": [d.strftime("%Y-%m-%dT%H:%M:%SZ") for d in dates],
        "close": [100.0 + i for i in range(50)]
    })

    with_targets = pipeline.compute_multi_horizon_targets(df)

    # Check n+0: nowcast close
    assert with_targets["target_price_0"].iloc[10] == 110.0

    # Check n+1: close at t+1 = 111.0, return = (111 - 110) / 110
    assert with_targets["target_price_1"].iloc[10] == 111.0
    assert pytest.approx(with_targets["target_ret_1"].iloc[10], rel=1e-3) == 1.0 / 110.0

    # Check n+5: close at t+5 = 115.0
    assert with_targets["target_price_5"].iloc[10] == 115.0
    assert pytest.approx(with_targets["target_ret_5"].iloc[10], rel=1e-3) == 5.0 / 110.0
