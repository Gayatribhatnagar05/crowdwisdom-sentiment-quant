"""
Tests for Quantitative Portfolio Backtester
"""

import pytest
import numpy as np
import pandas as pd

from src.modeling.backtester import PortfolioBacktester


def test_backtester_metrics():
    """Verifies calculation of Sharpe, Sortino, Max Drawdown, and Win Rate."""
    backtester = PortfolioBacktester(transaction_cost_bps=0.0005, annual_rf_rate=0.04)

    n_bars = 200
    df = pd.DataFrame({
        "timestamp_utc": [f"2026-01-01T{i:02d}:00:00Z" for i in range(n_bars)],
        "close": np.linspace(200, 220, n_bars),
        "target_ret_1": np.random.normal(0.0008, 0.01, n_bars)
    })

    # Perfect or positive signal
    signal = np.where(df["target_ret_1"] > 0, 0.005, -0.005)

    res = backtester.run_backtest(df, signal)
    metrics = res["metrics"]

    assert "sharpe_ratio" in metrics
    assert "sortino_ratio" in metrics
    assert "max_drawdown_pct" in metrics
    assert "win_rate_pct" in metrics
    assert metrics["max_drawdown_pct"] >= 0.0
    assert 0 <= metrics["win_rate_pct"] <= 100
    assert len(res["timeline"]) == n_bars
