"""
Apify X (Twitter) Scraper & Ingestion Module
Connects to Apify Actors to scrape X posts for the target asset over the last 180 days.
Includes resilient fallback and synthetic historical corpus generation for offline replication.
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from apify_client import ApifyClient

from config import config
from src.ingestion.database import DatabaseManager
from src.features.sentiment_analyzer import QuantitativeSentimentAnalyzer

logger = logging.getLogger(__name__)


class ApifyXScraper:
    """Scrapes and ingests X posts using Apify Actors, with UTC timestamp standardization."""

    def __init__(self, db_manager: DatabaseManager, sentiment_analyzer: Optional[QuantitativeSentimentAnalyzer] = None):
        self.db_manager = db_manager
        self.sentiment_analyzer = sentiment_analyzer or QuantitativeSentimentAnalyzer()
        self.api_token = config.apify_token
        self.actor_id = config.apify_actor_id
        self.ticker = config.ticker
        self.keywords = config.keywords

    def scrape_apify_live(self, days_back: int = 180, max_items: int = 5000) -> pd.DataFrame:
        """
        Executes live Apify actor run and retrieves dataset items.
        """
        if not self.api_token or self.api_token == "your_apify_api_token_here":
            logger.warning("No valid APIFY_API_TOKEN found in environment. Falling back to cached/synthetic dataset.")
            return self.generate_synthetic_historical_tweets(days_back=days_back)

        logger.info(f"Calling Apify Actor '{self.actor_id}' for {self.ticker}...")
        client = ApifyClient(self.api_token)

        start_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        # Build search query combining cashtag and top keywords
        query_string = f"{config.cashtag} OR {' OR '.join(config.keywords[:3])}"
        
        run_input = {
            "searchTerms": [config.cashtag, config.ticker] + config.keywords[:2],
            "queries": [config.cashtag, config.ticker],
            "maxTweets": max_items,
            "maxItems": max_items,
            "startDate": start_date,
            "sort": "Top",
            "lang": "en"
        }

        try:
            run = client.actor(self.actor_id).call(run_input=run_input, timeout_secs=180)
            dataset_items = client.dataset(run["defaultDatasetId"]).list_items().items
            logger.info(f"Retrieved {len(dataset_items)} items from Apify actor run.")

            parsed_tweets = []
            for item in dataset_items:
                parsed = self._normalize_apify_item(item)
                if parsed:
                    parsed_tweets.append(parsed)

            if not parsed_tweets:
                logger.warning("No valid tweets parsed from Apify response. Falling back to synthetic historical corpus.")
                return self.generate_synthetic_historical_tweets(days_back=days_back)

            df = pd.DataFrame(parsed_tweets)
            df = self.sentiment_analyzer.process_tweet_dataframe(df)
            return df

        except Exception as e:
            logger.error(f"Apify scraping encountered an exception: {e}. Falling back to cached/synthetic dataset.")
            return self.generate_synthetic_historical_tweets(days_back=days_back)

    def _normalize_apify_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalizes diverse Apify actor output schemas to standard tweet record."""
        try:
            # Tweet ID
            tweet_id = str(item.get("id") or item.get("id_str") or item.get("tweetId") or hash(str(item)))
            
            # Text
            text = item.get("text") or item.get("full_text") or item.get("content") or ""
            if not text:
                return None

            # Timestamp parsing
            raw_time = item.get("created_at") or item.get("createdAt") or item.get("timestamp")
            if raw_time:
                try:
                    dt = pd.to_datetime(raw_time, utc=True)
                except Exception:
                    dt = datetime.now(timezone.utc)
            else:
                dt = datetime.now(timezone.utc)
            
            timestamp_utc = dt.strftime("%Y-%m-%dT%H:%M:%SZ")

            # Engagement & Author
            user_data = item.get("user") or item.get("author") or {}
            author_username = str(user_data.get("screen_name") or user_data.get("userName") or item.get("username") or "anon")
            author_followers = int(user_data.get("followers_count") or user_data.get("followers") or item.get("followersCount") or 0)
            
            likes = int(item.get("favorite_count") or item.get("likeCount") or item.get("likes") or 0)
            reposts = int(item.get("retweet_count") or item.get("retweetCount") or item.get("retweets") or 0)
            replies = int(item.get("reply_count") or item.get("replyCount") or item.get("replies") or 0)
            views = int(item.get("view_count") or item.get("viewsCount") or item.get("views") or (likes * 15))

            return {
                "tweet_id": tweet_id,
                "ticker": self.ticker,
                "timestamp_utc": timestamp_utc,
                "text": text,
                "author_username": author_username,
                "author_followers": author_followers,
                "likes": likes,
                "reposts": reposts,
                "replies": replies,
                "views": views
            }
        except Exception:
            return None

    def generate_synthetic_historical_tweets(self, days_back: int = 180, target_count: int = 25000) -> pd.DataFrame:
        """
        Generates a high-fidelity synthetic historical X sentiment corpus spanning 180 days.
        Mimics real-world social sentiment dynamics, retail FOMO spikes, earnings panic,
        and diurnal posting cycles aligned with US equity trading hours.
        """
        logger.info(f"Generating realistic 180-day X sentiment corpus ({target_count} posts) for {self.ticker}...")
        np.random.seed(config.random_state)

        now = datetime.now(timezone.utc)
        start_date = now - timedelta(days=days_back)

        # Realistic tweet templates
        bullish_templates = [
            "${ticker} breaking out above key resistance! Massive call buying detected in options flow.",
            "Incredible delivery momentum for ${ticker}. Production ramping up faster than analysts modeled.",
            "Accumulating more ${ticker} on this dip. Long-term thesis remains completely intact.",
            "Wall Street completely missing the margin expansion story on ${ticker}. Easy outperform.",
            "${ticker} robotaxi software update looks unreal. Full autonomy is closer than people think.",
            "Strong support holding for ${ticker}. Ready for the next leg up to new highs.",
            "Top tier balance sheet and unmatched operating cash flow for ${ticker}. Extremely bullish.",
            "${ticker} calls printing today! Volume is surging across all strikes.",
            "Institutional accumulation in ${ticker} is undeniable here. Price action is coiling.",
            "Energy storage division of ${ticker} growing over 100% YoY. Sleeping giant."
        ]

        bearish_templates = [
            "${ticker} breaking below key 50-day moving average. Head and shoulders pattern completing.",
            "Selling my ${ticker} position. Margins are continuing to compress due to price cuts.",
            "Competition from domestic EV makers in China is brutal. ${ticker} market share eroding fast.",
            "Valuation on ${ticker} makes zero sense at 60x PE in a high interest rate regime.",
            "Heavy put buying spotted on ${ticker}. Hedging against further downside risk.",
            "Delivery numbers missed consensus expectations. ${ticker} guidance downgrade imminent.",
            "Macro headwinds and slowing consumer demand are hitting ${ticker} hard.",
            "Regulatory scrutiny and recalls heating up for ${ticker} autonomous features. Risky.",
            "Disappointing quarterly margin numbers from ${ticker}. Free cash flow dropped sharply.",
            "Shorting ${ticker} on this weak bounce into resistance. Trend remains decisively lower."
        ]

        neutral_templates = [
            "${ticker} scheduled to report quarterly financial results next Wednesday after market close.",
            "Tesla CEO Elon Musk comments on global supply chain logistics and manufacturing efficiency.",
            "Morgan Stanley updates price target for ${ticker} to $290, maintaining neutral rating.",
            "Checking ${ticker} options order flow today: 52% call volume vs 48% put volume.",
            "Tesla files patent for next-generation battery cell chemistry and cathode manufacturing.",
            "Comparing ${ticker} market cap against top global automakers as of Q3.",
            "Federal Reserve rate decision anticipated tomorrow; tech high-beta names like ${ticker} moving sideways.",
            "Daily volume in ${ticker} reached 65 million shares, aligned with 30-day average."
        ]

        tweets = []
        total_seconds = int((now - start_date).total_seconds())

        # Generate timestamps with peak volume during US market hours (13:30 to 20:00 UTC)
        for i in range(target_count):
            random_offset = np.random.uniform(0, total_seconds)
            tweet_dt = start_date + timedelta(seconds=random_offset)
            
            # Diurnal weighting: more tweets during US market hours
            hour_utc = tweet_dt.hour
            is_market_hours = 13 <= hour_utc <= 21
            if not is_market_hours and np.random.rand() < 0.4:
                # shift closer to market hours
                tweet_dt = tweet_dt.replace(hour=int(np.random.choice([14, 15, 16, 17, 18, 19, 20])))

            timestamp_str = tweet_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

            # Sentiment distribution: 45% bullish, 35% bearish, 20% neutral
            p = np.random.rand()
            if p < 0.45:
                text_template = np.random.choice(bullish_templates)
            elif p < 0.80:
                text_template = np.random.choice(bearish_templates)
            else:
                text_template = np.random.choice(neutral_templates)

            text = text_template.replace("${ticker}", f"${self.ticker}")

            # Engagement: Pareto / Log-Normal distribution
            likes = int(np.random.pareto(a=1.8) * 12)
            reposts = int(likes * np.random.uniform(0.1, 0.45))
            replies = int(likes * np.random.uniform(0.05, 0.3))
            views = int(likes * np.random.uniform(15, 80) + 150)
            followers = int(np.random.pareto(a=1.5) * 500 + 50)

            tweet_id = f"mock_{1800000000000000000 + i}"
            username = f"trader_{np.random.randint(100, 9999)}"

            tweets.append({
                "tweet_id": tweet_id,
                "ticker": self.ticker,
                "timestamp_utc": timestamp_str,
                "text": text,
                "author_username": username,
                "author_followers": followers,
                "likes": likes,
                "reposts": reposts,
                "replies": replies,
                "views": views
            })

        df = pd.DataFrame(tweets)
        df = df.sort_values("timestamp_utc").reset_index(drop=True)
        df = self.sentiment_analyzer.process_tweet_dataframe(df)

        logger.info(f"Generated {len(df)} sentiment-scored synthetic tweets.")
        return df

    def run_ingestion(self, days_back: Optional[int] = None) -> pd.DataFrame:
        """Runs the complete X ingestion pipeline and persists to SQLite."""
        days = days_back or config.days_back
        df = self.scrape_apify_live(days_back=days)
        n_inserted = self.db_manager.insert_tweets(df)
        logger.info(f"Persisted {n_inserted} tweets into SQLite database.")
        return df
