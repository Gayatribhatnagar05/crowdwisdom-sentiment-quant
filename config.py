"""
Central Configuration Module
CrowdWisdomTrading - X Sentiment Price Prediction Agent
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class AppConfig:
    project_root: Path = PROJECT_ROOT
    # Asset Settings
    ticker: str = os.getenv("ASSET_TICKER", "TSLA")
    asset_name: str = os.getenv("ASSET_NAME", "Tesla")
    cashtag: str = os.getenv("ASSET_CASHTAG", "$TSLA")
    keywords: List[str] = field(default_factory=lambda: [
        "$TSLA", "#TSLA", "Tesla", "Tesla stock", "Elon Musk TSLA"
    ])
    
    # Ingestion Parameters
    days_back: int = int(os.getenv("DAYS_BACK", "180"))
    price_interval: str = os.getenv("TIME_INTERVAL", "1h")  # '1h' or '1d'
    
    # API Credentials & Actor IDs
    apify_token: str = os.getenv("APIFY_API_TOKEN", "")
    apify_actor_id: str = os.getenv("APIFY_TWITTER_ACTOR_ID", "apidojo/tweet-scraper")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    exa_api_key: str = os.getenv("EXA_API_KEY", "")
    
    # Storage Paths
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    db_path: Path = PROJECT_ROOT / os.getenv("DATABASE_PATH", "data/crowdwisdom.db")
    reports_dir: Path = PROJECT_ROOT / "reports"
    figures_dir: Path = PROJECT_ROOT / "reports" / "figures"
    
    # Multi-Horizon Forecasting
    # n is the current period. We predict price/return at n+0, n+1, n+2, n+3, n+4, n+5
    forecast_horizons: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    
    # Feature Engineering Hyperparameters
    volume_z_window: int = 24  # 24 periods for volume rolling z-score
    sentiment_ema_short: int = 6
    sentiment_ema_long: int = 24
    
    # Walk-Forward Validation & Backtesting
    n_splits: int = 5
    embargo_periods: int = 6   # Embargo of 6 bars to prevent target leakage across horizons
    transaction_cost_bps: float = 0.0005  # 5 basis points per trade (0.05%)
    annual_risk_free_rate: float = 0.04   # 4% annual risk-free rate for Sharpe
    
    # Random Seed
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))


# Global Config Instance
config = AppConfig()
config.data_dir.mkdir(parents=True, exist_ok=True)
config.raw_data_dir.mkdir(parents=True, exist_ok=True)
config.reports_dir.mkdir(parents=True, exist_ok=True)
config.figures_dir.mkdir(parents=True, exist_ok=True)
