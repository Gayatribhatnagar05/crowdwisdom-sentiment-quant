"""
Feature Pipeline & Alignment Module
Strictly aligns X (Twitter) sentiment data with market price bars using UTC timestamps.
Generates multi-horizon targets for n+0, n+1, n+2, n+3, n+4, and n+5.
Ensures zero look-ahead bias and strict out-of-sample data hygiene.
"""

import logging
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd

from config import config
from src.features.price_features import PriceFeatureExtractor

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Builds the complete quantitative feature matrix derived exclusively
    from X sentiment and asset price data.
    """

    def __init__(self):
        self.price_extractor = PriceFeatureExtractor()
        self.horizons = config.forecast_horizons

    def aggregate_sentiment_by_bar(self, tweets_df: pd.DataFrame, price_bars_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates tweets into price bar intervals [t-1, t) in UTC.
        Guarantees that only posts published BEFORE or AT the close of bar t are included.
        """
        if tweets_df.empty or price_bars_df.empty:
            raise ValueError("Tweets and price bars cannot be empty for feature alignment.")

        tweets = tweets_df.copy()
        prices = price_bars_df.copy()

        tweets["datetime"] = pd.to_datetime(tweets["timestamp_utc"], utc=True)
        prices["datetime"] = pd.to_datetime(prices["timestamp_utc"], utc=True)

        tweets = tweets.sort_values("datetime").reset_index(drop=True)
        prices = prices.sort_values("datetime").reset_index(drop=True)

        # Cut tweet timestamps into price bar bins
        price_times = prices["datetime"].tolist()
        
        # Prepend initial boundary for first bar
        delta = price_times[1] - price_times[0] if len(price_times) > 1 else pd.Timedelta(hours=1)
        bin_edges = [price_times[0] - delta] + price_times

        # Bin each tweet into the price bar it belongs to
        tweets["bar_time"] = pd.cut(
            tweets["datetime"],
            bins=bin_edges,
            labels=price_times,
            right=True
        )

        # Aggregate sentiment metrics per price bar
        grouped = tweets.groupby("bar_time", observed=False)

        def weighted_avg_sentiment(group):
            w = group["engagement_weight"]
            c = group["sentiment_compound"]
            if w.sum() > 0:
                return float((c * w).sum() / w.sum())
            return float(c.mean()) if len(c) > 0 else 0.0

        agg_metrics = grouped.apply(
            lambda g: pd.Series({
                "tweet_count": len(g),
                "compound_mean": float(g["sentiment_compound"].mean()) if len(g) > 0 else 0.0,
                "compound_weighted": weighted_avg_sentiment(g),
                "pos_ratio": float((g["sentiment_compound"] > 0.05).mean()) if len(g) > 0 else 0.0,
                "neg_ratio": float((g["sentiment_compound"] < -0.05).mean()) if len(g) > 0 else 0.0,
                "total_likes": float(g["likes"].sum()),
                "total_reposts": float(g["reposts"].sum()),
                "total_views": float(g["views"].sum()),
            }),
            include_groups=False
        ).reset_index()

        agg_metrics["datetime"] = pd.to_datetime(agg_metrics["bar_time"], utc=True)
        agg_metrics = agg_metrics.drop(columns=["bar_time"])

        # Merge with price bars
        merged = pd.merge(prices, agg_metrics, on="datetime", how="left")

        # Fill bars that had zero tweets with neutral sentiment
        merged["tweet_count"] = merged["tweet_count"].fillna(0.0)
        merged["compound_mean"] = merged["compound_mean"].fillna(0.0)
        merged["compound_weighted"] = merged["compound_weighted"].fillna(0.0)
        merged["pos_ratio"] = merged["pos_ratio"].fillna(0.0)
        merged["neg_ratio"] = merged["neg_ratio"].fillna(0.0)
        merged["total_likes"] = merged["total_likes"].fillna(0.0)
        merged["total_reposts"] = merged["total_reposts"].fillna(0.0)
        merged["total_views"] = merged["total_views"].fillna(0.0)

        return merged

    def compute_sentiment_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes rolling dynamics, momentum, and retail volume anomalies from sentiment.
        """
        df = df.copy()

        # Bull / Bear Ratio
        pos_cnt = df["pos_ratio"] * df["tweet_count"]
        neg_cnt = df["neg_ratio"] * df["tweet_count"]
        df["bull_bear_ratio"] = ((pos_cnt + 1.0) / (neg_cnt + 1.0)).clip(upper=10.0)

        # Sentiment Moving Averages and Momentum (Fast - Slow EMA)
        ema_short = df["compound_weighted"].ewm(span=config.sentiment_ema_short, adjust=False).mean()
        ema_long = df["compound_weighted"].ewm(span=config.sentiment_ema_long, adjust=False).mean()
        df["sentiment_ema_6"] = ema_short
        df["sentiment_ema_24"] = ema_long
        df["sentiment_momentum"] = ema_short - ema_long

        # Sentiment Volatility / Disagreement
        df["sentiment_volatility_12"] = df["compound_weighted"].rolling(12, min_periods=2).std().fillna(0.0)

        # Tweet Volume Rolling Z-Score (Retail frenzy anomaly detector)
        vol_mean = df["tweet_count"].rolling(window=config.volume_z_window, min_periods=4).mean()
        vol_std = df["tweet_count"].rolling(window=config.volume_z_window, min_periods=4).std()
        df["tweet_volume_zscore"] = ((df["tweet_count"] - vol_mean) / (vol_std + 1e-6)).fillna(0.0).clip(-3.0, 5.0)

        # Interaction: Sentiment amplified by volume surges
        df["sentiment_volume_interaction"] = df["compound_weighted"] * df["tweet_volume_zscore"]

        return df

    def compute_multi_horizon_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates target variables for horizons n+0, n+1, n+2, n+3, n+4, and n+5.
        Target Returns r_{n, n+h} = (P_{n+h} - P_n) / P_n.
        Target Prices P_{n+h}.
        For n+0: Instantaneous current period return (Close_n - Close_{n-1}) / Close_{n-1}.
        """
        df = df.copy()
        close = df["close"]

        # n+0: Nowcast / current period return
        df["target_ret_0"] = close.pct_change(1).fillna(0.0)
        df["target_price_0"] = close

        # Forward horizons n+1 to n+5
        for h in [1, 2, 3, 4, 5]:
            forward_close = close.shift(-h)
            df[f"target_ret_{h}"] = ((forward_close - close) / (close + 1e-9))
            df[f"target_price_{h}"] = forward_close

        return df

    def build_feature_dataset(self, tweets_df: pd.DataFrame, price_bars_df: pd.DataFrame) -> pd.DataFrame:
        """
        Orchestrates full feature extraction and target construction.
        """
        logger.info("Aligning tweets and price bars...")
        aligned = self.aggregate_sentiment_by_bar(tweets_df, price_bars_df)

        logger.info("Computing price technical features...")
        with_price_feats = self.price_extractor.extract_features(aligned)

        logger.info("Computing sentiment dynamics and momentum features...")
        with_sent_feats = self.compute_sentiment_time_features(with_price_feats)

        logger.info("Generating multi-horizon targets (n+0 to n+5)...")
        final_df = self.compute_multi_horizon_targets(with_sent_feats)

        # Drop initial burn-in rows and trailing NaN target rows
        burn_in = max(config.volume_z_window, 26)
        valid_df = final_df.iloc[burn_in:-5].reset_index(drop=True)
        logger.info(f"Feature engineering complete. Usable records: {len(valid_df)}.")
        return valid_df

    @staticmethod
    def get_feature_columns() -> List[str]:
        """Returns the complete list of predictive features (sentiment and price only)."""
        return [
            # Price Features
            "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5", "ret_lag10",
            "volatility_6", "volatility_24", "hl_range_pct", "sma_ratio_6_24",
            "volume_ratio_24", "rsi_14", "bb_pct_b", "macd", "macd_signal", "macd_hist",
            # Sentiment Features
            "tweet_count", "compound_mean", "compound_weighted", "pos_ratio", "neg_ratio",
            "bull_bear_ratio", "sentiment_ema_6", "sentiment_ema_24", "sentiment_momentum",
            "sentiment_volatility_12", "tweet_volume_zscore", "sentiment_volume_interaction"
        ]
