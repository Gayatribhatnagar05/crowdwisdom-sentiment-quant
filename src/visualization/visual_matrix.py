"""
Visual Matrix Output Generator Module
Generates institutional-grade quantitative charts visualizing X sentiment vs price,
multi-horizon forecast matrices, equity curves, underwater drawdowns, and feature importance.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from config import config

logger = logging.getLogger(__name__)

# Apply publication-ready styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


class VisualMatrixGenerator:
    """Generates visual charts and plots for the evaluation report."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or config.figures_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_sentiment_price_overlay(self, df: pd.DataFrame, ticker: str) -> Path:
        """
        Generates dual-axis chart: Price time series with sentiment momentum & tweet volume.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})

        dates = pd.to_datetime(df["timestamp_utc"])

        # Top Plot: Price and Sentiment Heat
        ax1.plot(dates, df["close"], color="#1f77b4", linewidth=1.8, label=f"{ticker} Close Price")
        if "sma_ratio_6_24" in df.columns:
            sma_24 = df["close"].rolling(24).mean()
            ax1.plot(dates, sma_24, color="#ff7f0e", linestyle="--", linewidth=1.2, alpha=0.8, label="24-bar SMA")

        # Color fill based on weighted sentiment
        norm_sent = df["compound_weighted"].clip(-0.5, 0.5)
        ax1.fill_between(
            dates,
            df["close"].min() * 0.98,
            df["close"],
            where=(norm_sent > 0),
            color="#2ca02c",
            alpha=0.12,
            label="Bullish Sentiment Zone"
        )
        ax1.fill_between(
            dates,
            df["close"].min() * 0.98,
            df["close"],
            where=(norm_sent <= 0),
            color="#d62728",
            alpha=0.12,
            label="Bearish Sentiment Zone"
        )

        ax1.set_title(f"{ticker} Price Action vs X (Twitter) Sentiment Dynamics (180 Days)", fontsize=14, fontweight="bold", pad=12)
        ax1.set_ylabel("Price (USD)", fontsize=11, fontweight="bold")
        ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
        ax1.grid(True, linestyle=":", alpha=0.6)

        # Bottom Plot: Sentiment Momentum & Tweet Volume
        ax2_vol = ax2.twinx()
        ax2_vol.bar(dates, df["tweet_count"], width=0.03, color="#7f7f7f", alpha=0.3, label="Tweet Volume")
        ax2_vol.set_ylabel("Tweets / Bar", fontsize=10, color="#555555")
        ax2_vol.grid(False)

        ax2.plot(dates, df["sentiment_ema_6"], color="#2ca02c", linewidth=1.4, label="Fast Sentiment EMA(6)")
        ax2.plot(dates, df["sentiment_ema_24"], color="#9467bd", linewidth=1.4, linestyle="--", label="Slow Sentiment EMA(24)")
        ax2.axhline(0.0, color="#333333", linestyle=":", linewidth=0.8)

        ax2.set_ylabel("Sentiment Score", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Date (UTC)", fontsize=11, fontweight="bold")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
        ax2.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        out_path = self.output_dir / "sentiment_price_overlay.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved sentiment vs price chart to {out_path}")
        return out_path

    def plot_multi_horizon_metrics(self, horizon_results: Dict[int, Dict[str, Any]]) -> Path:
        """
        Visual Matrix: Compares out-of-sample directional accuracy, MAE, and IC across n+0 to n+5.
        """
        horizons = [f"n+{h}" for h in sorted(horizon_results.keys())]
        hit_rates = [horizon_results[h]["directional_accuracy"] * 100 for h in sorted(horizon_results.keys())]
        mae_prices = [horizon_results[h]["mean_mae_price"] for h in sorted(horizon_results.keys())]
        ics = [horizon_results[h]["mean_ic"] for h in sorted(horizon_results.keys())]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

        # Directional Accuracy
        bars1 = ax1.bar(horizons, hit_rates, color="#2b5c8f", width=0.55, edgecolor="#1c3d5e")
        ax1.axhline(50.0, color="#d62728", linestyle="--", linewidth=1.2, label="Random Guess (50%)")
        ax1.set_title("Directional Accuracy by Horizon (%)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Hit Ratio (%)", fontsize=10)
        ax1.set_ylim(45, max(max(hit_rates) + 5, 60))
        ax1.legend(loc="lower right")
        for bar in bars1:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

        # Mean Absolute Error
        bars2 = ax2.bar(horizons, mae_prices, color="#e67e22", width=0.55, edgecolor="#b95d13")
        ax2.set_title("Price Forecast MAE (USD)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Mean Absolute Error ($)", fontsize=10)
        for bar in bars2:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"${yval:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        # Information Coefficient
        bars3 = ax3.bar(horizons, ics, color="#27ae60", width=0.55, edgecolor="#1e8449")
        ax3.axhline(0.0, color="#333333", linestyle="-", linewidth=0.8)
        ax3.set_title("Information Coefficient (IC) by Horizon", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Pearson Correlation", fontsize=10)
        for bar in bars3:
            yval = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.003, f"{yval:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        plt.suptitle("Multi-Horizon Predictive Performance Matrix (n+0 to n+5)", fontsize=14, fontweight="bold", y=1.03)
        plt.tight_layout()
        out_path = self.output_dir / "forecast_matrix_horizons.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved multi-horizon matrix chart to {out_path}")
        return out_path

    def plot_backtest_performance(self, timeline_df: pd.DataFrame, metrics: Dict[str, Any]) -> Tuple[Path, Path]:
        """
        Generates Equity Curve and Underwater Drawdown charts.
        """
        dates = pd.to_datetime(timeline_df["timestamp_utc"])

        # 1. Equity Curve
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(dates, timeline_df["equity_strategy"], color="#16a085", linewidth=2.0, label=f"Sentiment Strategy (Net of Costs)")
        ax1.plot(dates, timeline_df["equity_bnh"], color="#7f8c8d", linestyle="--", linewidth=1.5, label="Buy & Hold Benchmark")

        ax1.set_title("Out-of-Sample Walk-Forward Cumulative Equity Curve", fontsize=13, fontweight="bold", pad=12)
        ax1.set_ylabel("Cumulative Wealth ($1 Initial)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Date (UTC)", fontsize=11, fontweight="bold")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

        stats_text = (
            f"Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}\n"
            f"Sortino Ratio: {metrics.get('sortino_ratio', 'N/A')}\n"
            f"Max Drawdown: {metrics.get('max_drawdown_pct', 'N/A')}%\n"
            f"Win Rate: {metrics.get('win_rate_pct', 'N/A')}%\n"
            f"Calmar Ratio: {metrics.get('calmar_ratio', 'N/A')}"
        )
        ax1.text(0.03, 0.95, stats_text, transform=ax1.transAxes, verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#bdc3c7", alpha=0.95),
                 fontsize=10, family="monospace")

        ax1.legend(loc="lower right", frameon=True, facecolor="white")
        ax1.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        equity_path = self.output_dir / "backtest_equity_curve.png"
        fig1.savefig(equity_path, dpi=300, bbox_inches="tight")
        plt.close(fig1)

        # 2. Drawdown Chart
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.fill_between(dates, -timeline_df["drawdown"] * 100, 0, color="#c0392b", alpha=0.4, label="Strategy Drawdown (%)")
        ax2.plot(dates, -timeline_df["drawdown"] * 100, color="#c0392b", linewidth=1.2)
        ax2.set_title("Strategy Underwater Drawdown Profile", fontsize=13, fontweight="bold", pad=10)
        ax2.set_ylabel("Drawdown (%)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Date (UTC)", fontsize=11, fontweight="bold")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax2.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        dd_path = self.output_dir / "drawdown_underwater.png"
        fig2.savefig(dd_path, dpi=300, bbox_inches="tight")
        plt.close(fig2)

        return equity_path, dd_path

    def plot_feature_importance(self, feature_importances: Dict[int, pd.DataFrame]) -> Path:
        """
        Visualizes the top predictive features across horizons, contrasting sentiment vs price features.
        """
        # Average importance across all horizons
        all_dfs = list(feature_importances.values())
        combined = pd.concat(all_dfs).groupby("feature").mean().reset_index()
        combined = combined.sort_values("importance", ascending=True).tail(12).reset_index(drop=True)

        sentiment_features = {
            "tweet_count", "compound_mean", "compound_weighted", "pos_ratio", "neg_ratio",
            "bull_bear_ratio", "sentiment_ema_6", "sentiment_ema_24", "sentiment_momentum",
            "sentiment_volatility_12", "tweet_volume_zscore", "sentiment_volume_interaction"
        }

        colors = ["#27ae60" if f in sentiment_features else "#2980b9" for f in combined["feature"]]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(combined["feature"], combined["importance"], color=colors, height=0.6, edgecolor="#333333", alpha=0.85)

        ax.set_title("Top Predictive Features (Sentiment vs Price Alpha Contribution)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Mean Feature Importance (Split Count / Gain across Walk-Forward Folds)", fontsize=10, fontweight="bold")

        # Custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#27ae60", edgecolor="#333333", label="X Sentiment Feature"),
            Patch(facecolor="#2980b9", edgecolor="#333333", label="Market Price Feature")
        ]
        ax.legend(handles=legend_elements, loc="lower right", frameon=True, facecolor="white")
        ax.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        imp_path = self.output_dir / "feature_importance.png"
        fig.savefig(imp_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved feature importance chart to {imp_path}")
        return imp_path
