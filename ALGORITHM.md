# Trading Bot Algorithm

This document summarizes the workflow of **The Great Sage** trading bot.

## Data Sources
- **Alpha Vantage**: daily OHLCV data for Indian equities (BSE/NSE)
- **NewsAPI**: recent news headlines for sentiment analysis

## Processing Steps
1. **Fetch price data** using Alpha Vantage and format into a DataFrame.
2. **Fetch news articles** relating to the ticker.
3. **Compute sentiment** from the news articles. If FinBERT is installed,
   it is used; otherwise sentiment comes from a weighted average of
   TextBlob polarity and VADER compound scores.
4. **Compute indicators** on the price data: RSI and a 20-day moving average.
5. **Generate signals**:
   - Go **long** when RSI < 40 and sentiment > 0.05.
   - Go **short** when RSI > 60 and sentiment < -0.05.
   - Positions are closed on opposite signal, after a maximum holding
     period, or when stop loss/take profit levels are hit.
6. **Backtesting** is performed with Backtrader to evaluate returns given a
   fixed starting capital. Position sizing uses 1% risk per trade.

## API Flow
```text
config.yaml -> load API keys
fetch_stock_data -> Alpha Vantage
fetch_newsapi_articles -> NewsAPI
analyze_sentiment -> TextBlob/VADER or FinBERT
compute_indicators -> ta library
run_backtest -> Backtrader engine
```

This pipeline allows experimentation with technical and sentiment-driven
signals. Live trading would require replacing the data sources and order
execution with broker APIs such as Zerodha or Groww.
