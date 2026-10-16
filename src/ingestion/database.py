"""
Database Management Module
Manages SQLite storage for raw tweets, price bars, news, and engineered features.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator
import pandas as pd


class DatabaseManager:
    """Manages relational data storage and retrieval using SQLite."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Yields a connection to the SQLite database and ensures closure on exit."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_tables(self) -> None:
        """Initializes schema and indices if not already present."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Raw Tweets Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS raw_tweets (
                    tweet_id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    text TEXT NOT NULL,
                    author_username TEXT,
                    author_followers INTEGER DEFAULT 0,
                    likes INTEGER DEFAULT 0,
                    reposts INTEGER DEFAULT 0,
                    replies INTEGER DEFAULT 0,
                    views INTEGER DEFAULT 0,
                    sentiment_compound REAL DEFAULT 0.0,
                    sentiment_pos REAL DEFAULT 0.0,
                    sentiment_neg REAL DEFAULT 0.0,
                    sentiment_neu REAL DEFAULT 1.0,
                    engagement_weight REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tweets_time ON raw_tweets (ticker, timestamp_utc)")

            # Price Bars Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS price_bars (
                    ticker TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    PRIMARY KEY (ticker, timestamp_utc)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_time ON price_bars (ticker, timestamp_utc)")

            # News Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS asset_news (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    title TEXT NOT NULL,
                    snippet TEXT,
                    url TEXT,
                    published_date TEXT,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_time ON asset_news (ticker, published_date)")

            # Features Aligned Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS features_aligned (
                    ticker TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    PRIMARY KEY (ticker, timestamp_utc)
                )
            """)
            conn.commit()

    def insert_tweets(self, tweets_df: pd.DataFrame) -> int:
        """Batch inserts or replaces tweets into SQLite."""
        if tweets_df.empty:
            return 0

        records = tweets_df.to_dict(orient="records")
        query = """
            INSERT OR REPLACE INTO raw_tweets (
                tweet_id, ticker, timestamp_utc, text, author_username,
                author_followers, likes, reposts, replies, views,
                sentiment_compound, sentiment_pos, sentiment_neg, sentiment_neu,
                engagement_weight
            ) VALUES (
                :tweet_id, :ticker, :timestamp_utc, :text, :author_username,
                :author_followers, :likes, :reposts, :replies, :views,
                :sentiment_compound, :sentiment_pos, :sentiment_neg, :sentiment_neu,
                :engagement_weight
            )
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, records)
            conn.commit()
            return len(records)

    def insert_price_bars(self, price_df: pd.DataFrame) -> int:
        """Batch inserts price bars into SQLite."""
        if price_df.empty:
            return 0

        records = price_df.to_dict(orient="records")
        query = """
            INSERT OR REPLACE INTO price_bars (
                ticker, timestamp_utc, open, high, low, close, volume
            ) VALUES (
                :ticker, :timestamp_utc, :open, :high, :low, :close, :volume
            )
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, records)
            conn.commit()
            return len(records)

    def insert_news(self, news_items: List[Dict[str, Any]]) -> int:
        """Inserts recent news into SQLite."""
        if not news_items:
            return 0

        query = """
            INSERT INTO asset_news (
                ticker, title, snippet, url, published_date, source
            ) VALUES (
                :ticker, :title, :snippet, :url, :published_date, :source
            )
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, news_items)
            conn.commit()
            return len(news_items)

    def load_tweets(self, ticker: str, start_time: Optional[str] = None) -> pd.DataFrame:
        """Loads tweets from database."""
        query = "SELECT * FROM raw_tweets WHERE ticker = ?"
        params = [ticker]
        if start_time:
            query += " AND timestamp_utc >= ?"
            params.append(start_time)
        query += " ORDER BY timestamp_utc ASC"

        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def load_price_bars(self, ticker: str, start_time: Optional[str] = None) -> pd.DataFrame:
        """Loads price bars from database."""
        query = "SELECT * FROM price_bars WHERE ticker = ?"
        params = [ticker]
        if start_time:
            query += " AND timestamp_utc >= ?"
            params.append(start_time)
        query += " ORDER BY timestamp_utc ASC"

        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def load_news(self, ticker: str, limit: int = 20) -> pd.DataFrame:
        """Loads recent news items from database."""
        query = "SELECT * FROM asset_news WHERE ticker = ? ORDER BY id DESC LIMIT ?"
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=[ticker, limit])
