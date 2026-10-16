"""
Tests for Data Ingestion and SQLite Database Storage
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile

from src.ingestion.database import DatabaseManager
from src.features.sentiment_analyzer import QuantitativeSentimentAnalyzer


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        yield DatabaseManager(db_path)


def test_database_init(temp_db):
    """Verifies tables and schema creation."""
    with temp_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]
        assert "raw_tweets" in tables
        assert "price_bars" in tables
        assert "asset_news" in tables


def test_insert_and_load_price_bars(temp_db):
    """Tests price bars ingestion and retrieval."""
    df = pd.DataFrame([
        {
            "ticker": "TSLA",
            "timestamp_utc": "2026-05-01T14:30:00Z",
            "open": 200.0,
            "high": 205.0,
            "low": 199.0,
            "close": 204.0,
            "volume": 1000000.0
        },
        {
            "ticker": "TSLA",
            "timestamp_utc": "2026-05-01T15:30:00Z",
            "open": 204.0,
            "high": 208.0,
            "low": 203.0,
            "close": 207.5,
            "volume": 1200000.0
        }
    ])
    inserted = temp_db.insert_price_bars(df)
    assert inserted == 2

    loaded = temp_db.load_price_bars("TSLA")
    assert len(loaded) == 2
    assert loaded["close"].iloc[1] == 207.5


def test_sentiment_scoring():
    """Verifies VADER financial lexicon and engagement weighting."""
    analyzer = QuantitativeSentimentAnalyzer()
    
    bullish = analyzer.analyze_text("$TSLA calls breakout to ATH! Moon!")
    assert bullish["compound"] > 0.5

    bearish = analyzer.analyze_text("$TSLA dump incoming, margin breakdown and puts")
    assert bearish["compound"] < -0.4

    # Engagement weighting test
    weight_viral = analyzer.calculate_engagement_weight(likes=5000, reposts=1200, views=250000, author_followers=50000)
    weight_low = analyzer.calculate_engagement_weight(likes=1, reposts=0, views=20, author_followers=10)
    assert weight_viral > weight_low * 2.0
