"""
Interactive Visual Matrix Output Generator & Sentiment Dashboard
Streamlit Application for CrowdWisdomTrading Assessment
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import config
from src.ingestion.database import DatabaseManager

# Page Configuration
st.set_page_config(
    page_title="CrowdWisdomTrading - Sentiment Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark / Modern Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #1E3A8A;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Database Connection
db_manager = DatabaseManager(config.db_path)

# Sidebar
st.sidebar.image("https://crowdwisdomtrading.com/wp-content/uploads/2021/04/cropped-CrowdWisdom-Logo-1-1.png", width=220)
st.sidebar.title("Configuration")
asset_ticker = st.sidebar.selectbox("Target Asset", [config.ticker, "NVDA", "BTC-USD"], index=0)
analysis_window = st.sidebar.slider("Analysis Window (Days)", 30, 180, config.days_back)
st.sidebar.markdown("---")
st.sidebar.info(
    "**Intern Assessment System**\n\n"
    "Predicting asset prices at **n+0, n+1, n+2, n+3, n+4, and n+5** based on **X (Twitter) Sentiment** and **Price only**.\n\n"
    "**Data Providers:**\n- Apify (X Scraping)\n- Tavily (News)\n- Yahoo Finance (OHLCV)"
)

# Header
st.markdown(f"<div class='main-title'>📈 {asset_ticker} X Sentiment Price Prediction Agent</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>CrowdWisdomTrading Quantitative Data Science Pipeline & Visual Matrix Output Generator</div>", unsafe_allow_html=True)

# Load Data from SQLite
price_df = db_manager.load_price_bars(asset_ticker)
tweets_df = db_manager.load_tweets(asset_ticker)
news_df = db_manager.load_news(asset_ticker, limit=5)

if price_df.empty or tweets_df.empty:
    st.warning("⚠️ No database records found. Run the ingestion pipeline script: `python scripts/run_pipeline.py` to ingest 180-day data.")
    st.stop()

# Top Metric Banner
col1, col2, col3, col4, col5 = st.columns(5)
latest_price = price_df["close"].iloc[-1]
prev_price = price_df["close"].iloc[-2] if len(price_df) > 1 else latest_price
price_chg = ((latest_price - prev_price) / prev_price) * 100
total_tweets = len(tweets_df)
avg_sent = tweets_df["sentiment_compound"].mean()

with col1:
    st.metric("Latest Close", f"${latest_price:.2f}", f"{price_chg:+.2f}%")
with col2:
    st.metric("Total X Posts Ingested", f"{total_tweets:,}")
with col3:
    st.metric("Mean Compound Sentiment", f"{avg_sent:+.3f}")
with col4:
    bull_pct = (tweets_df["sentiment_compound"] > 0.05).mean() * 100
    st.metric("Bullish Share", f"{bull_pct:.1f}%")
with col5:
    st.metric("Forward Horizons", "n+0 to n+5")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Sentiment vs Price Overlay",
    "🎯 Multi-Horizon Forecast Matrix",
    "📈 Walk-Forward Backtest & Risk",
    "📝 1-Page Sentiment Outlook",
    "🔍 Feature Importance & News"
])

with tab1:
    st.subheader("Dynamic Visual Matrix: X Sentiment vs Price Action")
    st.caption("Synchronized in UTC: Upper pane displays price bars; lower pane shows engagement-weighted sentiment with volume bursts.")

    # Prepare data for Plotly
    price_df["datetime"] = pd.to_datetime(price_df["timestamp_utc"])
    tweets_df["datetime"] = pd.to_datetime(tweets_df["timestamp_utc"])

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=(f"{asset_ticker} Price Action", "X Engagement-Weighted Sentiment & Volume")
    )

    # Price Line
    fig.add_trace(
        go.Scatter(x=price_df["datetime"], y=price_df["close"], mode="lines", name="Close Price", line=dict(color="#1f77b4", width=2)),
        row=1, col=1
    )

    # 24-period SMA
    if len(price_df) > 24:
        sma = price_df["close"].rolling(24).mean()
        fig.add_trace(
            go.Scatter(x=price_df["datetime"], y=sma, mode="lines", name="24-bar SMA", line=dict(color="#ff7f0e", dash="dot", width=1.5)),
            row=1, col=1
        )

    # Downsampled or aggregated sentiment
    hourly_tweets = tweets_df.set_index("datetime").resample("1h").agg({
        "sentiment_compound": "mean",
        "engagement_weight": "sum",
        "tweet_id": "count"
    }).dropna()

    colors = ["#2ca02c" if s >= 0 else "#d62728" for s in hourly_tweets["sentiment_compound"]]

    fig.add_trace(
        go.Bar(
            x=hourly_tweets.index,
            y=hourly_tweets["sentiment_compound"],
            name="Hourly Mean Sentiment",
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=650,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Multi-Horizon Price Forecasts (n+0, n+1, n+2, n+3, n+4, n+5)")
    st.markdown("Forecasting model trained on **sentiment and price only** using LightGBM.")

    # Check for saved figures or compute on the fly
    matrix_img_path = config.figures_dir / "forecast_matrix_horizons.png"
    if matrix_img_path.exists():
        st.image(str(matrix_img_path), caption="Multi-Horizon Performance Matrix across Walk-Forward Folds", use_column_width=True)

    st.markdown("### Interactive Scenario Simulator")
    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        sim_sent = st.slider("Hypothetical Sentiment Shift (Compound)", -1.0, 1.0, 0.35, 0.05)
        sim_vol_z = st.slider("Hypothetical Tweet Volume Z-Score", -2.0, 4.0, 1.5, 0.5)
    with col_sim2:
        st.markdown("**Simulated Price Trajectory from Current Price:**")
        sim_horizons = ["n+0", "n+1", "n+2", "n+3", "n+4", "n+5"]
        sim_rets = [0.001 * (i + 1) * (sim_sent * 1.5) for i in range(6)]
        sim_prices = [latest_price * (1.0 + r) for r in sim_rets]

        sim_df = pd.DataFrame({
            "Horizon": sim_horizons,
            "Expected Return (%)": [f"{r*100:+.2f}%" for r in sim_rets],
            "Simulated Price ($)": [f"${p:.2f}" for p in sim_prices]
        })
        st.dataframe(sim_df, use_container_width=True)

with tab3:
    st.subheader("Walk-Forward Validation & Backtesting Performance")
    col_b1, col_b2 = st.columns(2)

    equity_img = config.figures_dir / "backtest_equity_curve.png"
    dd_img = config.figures_dir / "drawdown_underwater.png"

    with col_b1:
        if equity_img.exists():
            st.image(str(equity_img), caption="Out-of-Sample Cumulative Equity Curve vs Buy & Hold", use_column_width=True)
        else:
            st.info("Run `python scripts/run_pipeline.py` to generate the backtest charts.")

    with col_b2:
        if dd_img.exists():
            st.image(str(dd_img), caption="Underwater Drawdown Comparison", use_column_width=True)
        else:
            st.info("Underwater drawdown chart will appear here.")

with tab4:
    st.subheader("1-Page Sentiment Outlook Report")
    outlook_path = config.PROJECT_ROOT / "CURRENT_SENTIMENT_OUTLOOK.md"
    if outlook_path.exists():
        with open(outlook_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("Run the pipeline script to compile the latest 1-page sentiment outlook.")

with tab5:
    st.subheader("Feature Importance & Institutional Context")
    col_f1, col_f2 = st.columns([1.2, 0.8])

    with col_f1:
        feat_img = config.figures_dir / "feature_importance.png"
        if feat_img.exists():
            st.image(str(feat_img), caption="Sentiment vs Price Features Predictive Importance", use_column_width=True)

    with col_f2:
        st.markdown("#### Recent News Headwinds & Catalysts (via Tavily)")
        if not news_df.empty:
            for _, r in news_df.iterrows():
                st.markdown(f"**{r['title']}**")
                st.caption(f"Source: {r.get('source', 'Tavily')} | {r.get('snippet', '')[:160]}...")
                st.markdown("---")
        else:
            st.info("No news ingested yet.")
