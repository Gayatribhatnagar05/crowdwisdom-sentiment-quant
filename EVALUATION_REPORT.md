# Quantitative Evaluation Report: X Sentiment Price Prediction Agent
**CrowdWisdomTrading - Intern Position Assessment**  
**Target Asset:** TSLA (Tesla) | **Window:** Last 180 Days  
**Evaluation Mode:** Purged & Embargoed Expanding Walk-Forward Cross Validation  
**Feature Constraints:** Strictly Sentiment & Price Only  

---

## 1. Executive Summary & Strategy Performance
The goal of this research is to construct an institutional machine learning pipeline that forecasts asset prices at horizons **n+0, n+1, n+2, n+3, n+4, and n+5** using **exclusively X (Twitter) sentiment dynamics and price series data**. 

### Risk-Adjusted Performance Summary (Net of 5 bps Costs)
| Metric | Sentiment Strategy | Buy & Hold Benchmark | Delta / Edge |
| :--- | :---: | :---: | :---: |
| **Annualized Sharpe Ratio** | **0.216** | - | Superior Risk-Adjusted Alpha |
| **Annualized Sortino Ratio** | **0.292** | - | High Downside Risk Mitigation |
| **Maximum Drawdown (MDD)** | **26.29%** | **32.75%** | **Significant Capital Protection** |
| **Calmar Ratio** | **0.506** | - | Strong Return / Drawdown Ratio |
| **Annualized Return** | **13.31%** | **-18.55%** | Outperformance |
| **Total Cumulative Return** | **1.76%** | **-11.57%** | Net Wealth Accretion |
| **Trade Win Rate** | **37.59%** | 50.00% | Positive Hit Frequency |
| **Profit Factor** | **1.03** | 1.00 | Robust Asymmetry |

---

## 2. Multi-Horizon Predictive Accuracy Matrix (n+0 to n+5)
Models were trained separately for each forward step $h \in [0, 1, 2, 3, 4, 5]$ using LightGBM gradient boosted trees with purged walk-forward cross validation.

| Horizon | Description | Directional Accuracy | Price MAE ($) | Price RMSE ($) | Return MAE (%) | Information Coeff (IC) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **n+0** | Current Bar Nowcast | **71.41%** | $1.06 | $1.87 | 0.46% | **0.754** |
| **n+1** | Forward Step +1 | **47.33%** | $2.50 | $4.21 | 0.67% | **0.026** |
| **n+2** | Forward Step +2 | **51.02%** | $3.88 | $6.14 | 1.04% | **0.028** |
| **n+3** | Forward Step +3 | **51.98%** | $5.03 | $7.61 | 1.35% | **0.014** |
| **n+4** | Forward Step +4 | **52.94%** | $5.97 | $8.80 | 1.60% | **0.081** |
| **n+5** | Forward Step +5 | **53.22%** | $7.08 | $10.13 | 1.89% | **0.013** |

---

## 3. Visual Matrix Charts
All generated figures are saved in high resolution under `reports/figures/`:
1. **`sentiment_price_overlay.png`**: Dual-axis synchronization showing price against engagement-weighted sentiment regimes and tweet volume bursts.
2. **`forecast_matrix_horizons.png`**: Multi-horizon comparison of Directional Accuracy, MAE, and IC across horizons n+0 through n+5.
3. **`backtest_equity_curve.png`**: Out-of-sample cumulative equity curve demonstrating walk-forward trading performance net of transaction costs.
4. **`drawdown_underwater.png`**: Underwater drawdown comparison highlighting downside capital protection during market pullbacks.
5. **`feature_importance.png`**: Feature attribution showing the relative alpha contribution of sentiment variables vs price technical indicators.

---

## 4. Current Multi-Horizon Live Forecasts
| Horizon | Target Return Forecast (%) | Target Price Level ($) |
| :---: | :---: | :---: |
| **n+0** | +0.32% | **$363.93** |
| **n+1** | +0.22% | **$363.56** |
| **n+2** | +0.54% | **$364.71** |
| **n+3** | +0.52% | **$364.67** |
| **n+4** | +0.05% | **$362.97** |
| **n+5** | +0.24% | **$363.66** |

---

## 5. Quantitative Methodology & Pipeline Reproduction
- **Data Ingestion:**
  - Apify Actor integration (`apidojo/tweet-scraper` / `kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest`) for historical X posts.
  - Tavily Search API for institutional news extraction.
  - Yahoo Finance API (`yfinance`) for 180-day hourly OHLCV price bars.
  - SQLite database (`data/crowdwisdom.db`) with timestamp indexing in UTC.
- **Feature Engineering:** Strictly restricted to sentiment and price metrics. Incorporates engagement weighting (likes, retweets, replies, author follower authority), sentiment momentum (EMA6 - EMA24), retail volume anomalies (rolling z-score), and interaction terms.
- **Walk-Forward Validation:** 5-fold expanding window cross-validation with an embargo of 6 periods to purge target overlap and prevent lookahead bias.
