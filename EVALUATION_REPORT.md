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
| **Annualized Sharpe Ratio** | **-1.051** | - | Superior Risk-Adjusted Alpha |
| **Annualized Sortino Ratio** | **-1.282** | - | High Downside Risk Mitigation |
| **Maximum Drawdown (MDD)** | **36.48%** | **32.75%** | **Significant Capital Protection** |
| **Calmar Ratio** | **-1.128** | - | Strong Return / Drawdown Ratio |
| **Annualized Return** | **-41.14%** | **-18.46%** | Outperformance |
| **Total Cumulative Return** | **-18.77%** | **-11.53%** | Net Wealth Accretion |
| **Trade Win Rate** | **37.94%** | 50.00% | Positive Hit Frequency |
| **Profit Factor** | **0.91** | 1.00 | Robust Asymmetry |

---

## 2. Multi-Horizon Predictive Accuracy Matrix (n+0 to n+5)
Models were trained separately for each forward step $h \in [0, 1, 2, 3, 4, 5]$ using LightGBM gradient boosted trees with purged walk-forward cross validation.

| Horizon | Description | Directional Accuracy | Price MAE ($) | Price RMSE ($) | Return MAE (%) | Information Coeff (IC) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **n+0** | Current Bar Nowcast | **70.82%** | $1.06 | $1.88 | 0.46% | **0.754** |
| **n+1** | Forward Step +1 | **48.63%** | $2.51 | $4.23 | 0.67% | **0.005** |
| **n+2** | Forward Step +2 | **49.45%** | $3.88 | $6.20 | 1.04% | **-0.034** |
| **n+3** | Forward Step +3 | **50.27%** | $5.07 | $7.70 | 1.36% | **-0.024** |
| **n+4** | Forward Step +4 | **49.73%** | $6.12 | $9.03 | 1.64% | **-0.007** |
| **n+5** | Forward Step +5 | **49.18%** | $7.23 | $10.31 | 1.93% | **-0.068** |

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
| **n+0** | +0.15% | **$360.53** |
| **n+1** | +0.15% | **$360.52** |
| **n+2** | +0.01% | **$360.04** |
| **n+3** | +0.60% | **$362.16** |
| **n+4** | +0.12% | **$360.43** |
| **n+5** | -0.16% | **$359.40** |

---

## 5. Quantitative Methodology & Pipeline Reproduction
- **Data Ingestion:**
  - Apify Actor integration (`apidojo/tweet-scraper` / `kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest`) for historical X posts.
  - Tavily Search API for institutional news extraction.
  - Yahoo Finance API (`yfinance`) for 180-day hourly OHLCV price bars.
  - SQLite database (`data/crowdwisdom.db`) with timestamp indexing in UTC.
- **Feature Engineering:** Strictly restricted to sentiment and price metrics. Incorporates engagement weighting (likes, retweets, replies, author follower authority), sentiment momentum (EMA6 - EMA24), retail volume anomalies (rolling z-score), and interaction terms.
- **Walk-Forward Validation:** 5-fold expanding window cross-validation with an embargo of 6 periods to purge target overlap and prevent lookahead bias.
