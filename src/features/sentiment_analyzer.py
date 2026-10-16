"""
Sentiment Analysis & Engagement Weighting Module
Processes raw post text using VADER with financial lexicon enhancement
and applies quantitative engagement weighting based on likes, retweets, replies,
impressions, and author follower influence.
"""

import math
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class QuantitativeSentimentAnalyzer:
    """
    Computes VADER sentiment scores enhanced with quantitative financial lexicon
    and engagement-weighted authority metrics.
    """

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self._inject_financial_lexicon()

    def _inject_financial_lexicon(self) -> None:
        """Enriches standard VADER lexicon with financial and crypto/equity market jargon."""
        financial_lexicon = {
            "bullish": 2.8,
            "calls": 2.0,
            "breakout": 2.2,
            "ath": 2.6,
            "moon": 2.5,
            "upgrade": 2.4,
            "outperform": 2.2,
            "beat": 2.0,
            "long": 1.5,
            "support": 1.2,
            "rally": 2.3,
            "surge": 2.4,
            "undervalued": 2.1,
            "overweight": 2.0,
            
            "bearish": -2.8,
            "puts": -2.0,
            "breakdown": -2.2,
            "dump": -2.7,
            "crash": -3.0,
            "downgrade": -2.5,
            "underperform": -2.2,
            "miss": -2.0,
            "short": -1.5,
            "resistance": -1.0,
            "plunge": -2.5,
            "selloff": -2.6,
            "dilution": -2.4,
            "underweight": -2.0,
            "investigation": -2.5,
            "recall": -2.0
        }
        self.analyzer.lexicon.update(financial_lexicon)

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Calculates raw VADER sentiment scores for a given text."""
        if not text or not isinstance(text, str):
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
        return self.analyzer.polarity_scores(text)

    def calculate_engagement_weight(
        self,
        likes: int = 0,
        reposts: int = 0,
        replies: int = 0,
        views: int = 0,
        author_followers: int = 0
    ) -> float:
        """
        Calculates an engagement & author authority multiplier:
        Weight = (1 + ln(1 + likes + 2*reposts + 0.5*replies + views/1000)) * (1 + 0.15*ln(1 + followers))
        """
        likes = max(0, int(likes or 0))
        reposts = max(0, int(reposts or 0))
        replies = max(0, int(replies or 0))
        views = max(0, int(views or 0))
        author_followers = max(0, int(author_followers or 0))

        # Engagement term (reposts are weighted double as high-conviction virality)
        engagement_metric = likes + (2.0 * reposts) + (0.5 * replies) + (views / 1000.0)
        engagement_factor = 1.0 + math.log1p(engagement_metric)

        # Authority term (diminishing returns on follower count)
        authority_factor = 1.0 + 0.15 * math.log1p(author_followers)

        return float(engagement_factor * authority_factor)

    def process_tweet_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Augments a DataFrame of raw tweets with compound, pos, neg, neu sentiment
        and engagement weights.
        """
        if df.empty:
            return df

        df = df.copy()

        # Vectorized or row-wise sentiment scoring
        scores = [self.analyze_text(str(t)) for t in df["text"]]
        df["sentiment_compound"] = [s["compound"] for s in scores]
        df["sentiment_pos"] = [s["pos"] for s in scores]
        df["sentiment_neg"] = [s["neg"] for s in scores]
        df["sentiment_neu"] = [s["neu"] for s in scores]

        # Calculate engagement weights
        weights = [
            self.calculate_engagement_weight(
                likes=row.get("likes", 0),
                reposts=row.get("reposts", 0),
                replies=row.get("replies", 0),
                views=row.get("views", 0),
                author_followers=row.get("author_followers", 0)
            )
            for _, row in df.iterrows()
        ]
        df["engagement_weight"] = weights

        return df
