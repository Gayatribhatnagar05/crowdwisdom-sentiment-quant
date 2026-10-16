"""
Price Feature Engineering Module
Calculates price-only technical indicators, volatility, and lagged returns.
Strictly adheres to quantitative requirement: features derived from price only.
"""

import numpy as np
import pandas as pd


class PriceFeatureExtractor:
    """Computes technical, momentum, and volatility features from price series."""

    @staticmethod
    def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Computes Relative Strength Index (RSI)."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(window=period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def compute_bollinger_pct_b(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
        """Computes Bollinger Bands %B indicator."""
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = sma + (num_std * std)
        lower = sma - (num_std * std)
        pct_b = (series - lower) / (upper - lower + 1e-9)
        return pct_b.clip(lower=-0.5, upper=1.5).fillna(0.5)

    @staticmethod
    def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Computes MACD line and Signal line normalized by price."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = (ema_fast - ema_slow) / (series + 1e-9)
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return pd.DataFrame({"macd": macd, "macd_signal": macd_signal, "macd_hist": macd_hist})

    def extract_features(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts comprehensive price-derived features.
        """
        df = price_df.copy()
        df = df.sort_values("timestamp_utc").reset_index(drop=True)

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # Lagged Returns
        df["ret_1"] = close.pct_change(1).fillna(0.0)
        df["ret_lag1"] = df["ret_1"].shift(1).fillna(0.0)
        df["ret_lag2"] = df["ret_1"].shift(2).fillna(0.0)
        df["ret_lag3"] = df["ret_1"].shift(3).fillna(0.0)
        df["ret_lag5"] = close.pct_change(5).shift(1).fillna(0.0)
        df["ret_lag10"] = close.pct_change(10).shift(1).fillna(0.0)

        # Realized Volatility
        df["volatility_6"] = df["ret_1"].rolling(window=6, min_periods=2).std().fillna(0.0)
        df["volatility_24"] = df["ret_1"].rolling(window=24, min_periods=5).std().fillna(0.0)

        # High-Low Range / Bar Volatility
        df["hl_range_pct"] = (high - low) / (close + 1e-9)

        # Moving Average Price Momentum
        sma_6 = close.rolling(6).mean()
        sma_24 = close.rolling(24).mean()
        df["sma_ratio_6_24"] = (sma_6 / (sma_24 + 1e-9) - 1.0).fillna(0.0)

        # Volume Ratio
        vol_sma_24 = volume.rolling(24).mean()
        df["volume_ratio_24"] = (volume / (vol_sma_24 + 1e-9)).fillna(1.0).clip(upper=5.0)

        # Technical Oscillators
        df["rsi_14"] = self.compute_rsi(close, period=14) / 100.0  # Normalized to [0, 1]
        df["bb_pct_b"] = self.compute_bollinger_pct_b(close, period=20)

        macd_df = self.compute_macd(close)
        df["macd"] = macd_df["macd"].fillna(0.0)
        df["macd_signal"] = macd_df["macd_signal"].fillna(0.0)
        df["macd_hist"] = macd_df["macd_hist"].fillna(0.0)

        return df
