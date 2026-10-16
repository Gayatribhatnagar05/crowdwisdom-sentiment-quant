"""
Quantitative Backtester Module
Simulates walk-forward out-of-sample trading performance.
Computes institutional risk-adjusted metrics: Sharpe Ratio, Sortino Ratio, Max Drawdown,
Calmar Ratio, Win Rate, and cumulative equity curves with realistic transaction friction.
"""

import math
from typing import Dict, Any
import numpy as np
import pandas as pd

from config import config


class PortfolioBacktester:
    """
    Backtests an asset allocation or market-timing strategy driven by model predictions.
    Enforces transaction costs and computes annualized risk and return metrics.
    """

    def __init__(
        self,
        transaction_cost_bps: float = 0.0005,
        annual_rf_rate: float = 0.04,
        allow_short: bool = True
    ):
        self.transaction_cost_bps = transaction_cost_bps
        self.annual_rf_rate = annual_rf_rate
        self.allow_short = allow_short
        # Periods per year: 252 * 7 = 1764 for 1h US equity bars, 252 for daily
        self.periods_per_year = 1764 if config.price_interval == "1h" else 252

    def run_backtest(
        self,
        df: pd.DataFrame,
        oos_predictions_ret: np.ndarray,
        threshold: float = 0.0005
    ) -> Dict[str, Any]:
        """
        Executes backtest on out-of-sample test segments.
        Signal is generated strictly from out-of-sample predictions at bar t-1 to trade bar t.
        """
        # Filter strictly out-of-sample indices (where prediction is not NaN)
        valid_mask = ~np.isnan(oos_predictions_ret)
        if not np.any(valid_mask):
            raise ValueError("No valid out-of-sample predictions found for backtesting.")

        sub_df = df[valid_mask].copy().reset_index(drop=True)
        pred_rets = oos_predictions_ret[valid_mask]

        # Asset realized 1-period forward return
        asset_rets = sub_df["target_ret_1"].values

        # Generate Positions: +1 (Long), -1 (Short) or 0 (Flat)
        positions = np.zeros(len(pred_rets))
        for i in range(len(pred_rets)):
            if pred_rets[i] > threshold:
                positions[i] = 1.0
            elif pred_rets[i] < -threshold:
                positions[i] = -1.0 if self.allow_short else 0.0
            else:
                positions[i] = 0.0

        # Position execution: Signal generated at t is held from t to t+1
        # Lag positions by 1 bar to prevent lookahead
        held_positions = np.zeros_like(positions)
        held_positions[1:] = positions[:-1]

        # Turnover and transaction costs
        turnover = np.abs(np.diff(held_positions, prepend=0.0))
        costs = turnover * self.transaction_cost_bps

        # Strategy returns
        gross_strat_rets = held_positions * asset_rets
        net_strat_rets = gross_strat_rets - costs

        # Benchmark: Buy-and-Hold
        bnh_rets = asset_rets

        # Cumulative equity curves
        equity_curve_strat = np.cumprod(1.0 + net_strat_rets)
        equity_curve_bnh = np.cumprod(1.0 + bnh_rets)

        # Performance Metrics
        metrics = self._calculate_metrics(net_strat_rets, bnh_rets, equity_curve_strat, equity_curve_bnh)

        # Build output timeline DataFrame
        timeline_df = pd.DataFrame({
            "timestamp_utc": sub_df["timestamp_utc"],
            "close": sub_df["close"],
            "predicted_return": pred_rets,
            "realized_return": asset_rets,
            "position": held_positions,
            "turnover": turnover,
            "net_strategy_return": net_strat_rets,
            "equity_strategy": equity_curve_strat,
            "equity_bnh": equity_curve_bnh,
            "drawdown": metrics["drawdown_series"]
        })

        return {
            "metrics": metrics,
            "timeline": timeline_df
        }

    def _calculate_metrics(
        self,
        strat_rets: np.ndarray,
        bnh_rets: np.ndarray,
        equity_strat: np.ndarray,
        equity_bnh: np.ndarray
    ) -> Dict[str, Any]:
        """Calculates annualized Sharpe, Sortino, Max Drawdown, and trading statistics."""
        n = len(strat_rets)
        if n == 0:
            return {}

        rf_period = self.annual_rf_rate / self.periods_per_year

        # Strategy Returns Statistics
        mean_ret = np.mean(strat_rets)
        std_ret = np.std(strat_rets, ddof=1) if len(strat_rets) > 1 else 1e-6
        excess_rets = strat_rets - rf_period

        # Annualized Return
        ann_return = mean_ret * self.periods_per_year
        bnh_mean = np.mean(bnh_rets)
        ann_bnh_return = bnh_mean * self.periods_per_year

        # Annualized Volatility
        ann_vol = std_ret * math.sqrt(self.periods_per_year)

        # Sharpe Ratio
        sharpe = (np.mean(excess_rets) / (std_ret + 1e-9)) * math.sqrt(self.periods_per_year)

        # Sortino Ratio (Downside deviation only)
        downside_rets = strat_rets[strat_rets < rf_period] - rf_period
        if len(downside_rets) > 1:
            downside_std = math.sqrt(np.mean(downside_rets ** 2))
            sortino = (np.mean(excess_rets) / (downside_std + 1e-9)) * math.sqrt(self.periods_per_year)
        else:
            sortino = sharpe

        # Maximum Drawdown
        running_max = np.maximum.accumulate(equity_strat)
        drawdowns = (running_max - equity_strat) / running_max
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Calmar Ratio
        calmar = ann_return / (max_drawdown + 1e-9)

        # Benchmark Max Drawdown
        bnh_running_max = np.maximum.accumulate(equity_bnh)
        bnh_drawdowns = (bnh_running_max - equity_bnh) / bnh_running_max
        bnh_max_drawdown = float(np.max(bnh_drawdowns)) if len(bnh_drawdowns) > 0 else 0.0

        # Win Rate & Profit Factor
        winning_trades = strat_rets[strat_rets > 0]
        losing_trades = strat_rets[strat_rets < 0]
        active_periods = strat_rets[strat_rets != 0]

        win_rate = len(winning_trades) / len(active_periods) if len(active_periods) > 0 else 0.5
        gross_profit = float(np.sum(winning_trades)) if len(winning_trades) > 0 else 0.0
        gross_loss = float(abs(np.sum(losing_trades))) if len(losing_trades) > 0 else 1e-6
        profit_factor = gross_profit / gross_loss

        # Total Cumulative Return
        total_return_strat = float(equity_strat[-1] - 1.0)
        total_return_bnh = float(equity_bnh[-1] - 1.0)

        return {
            "annualized_return_pct": round(ann_return * 100, 2),
            "annualized_bnh_return_pct": round(ann_bnh_return * 100, 2),
            "annualized_volatility_pct": round(ann_vol * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "bnh_max_drawdown_pct": round(bnh_max_drawdown * 100, 2),
            "calmar_ratio": round(calmar, 3),
            "total_return_pct": round(total_return_strat * 100, 2),
            "bnh_total_return_pct": round(total_return_bnh * 100, 2),
            "win_rate_pct": round(win_rate * 100, 2),
            "profit_factor": round(profit_factor, 2),
            "drawdown_series": drawdowns
        }
