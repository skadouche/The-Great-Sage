# The Great Sage Trading Bot

This repository contains an experimental trading bot for Indian equities.
It fetches market data from Alpha Vantage, performs headline sentiment
analysis blending TextBlob with the VADER model from NLTK, and executes
a simple long/short strategy that can be backtested with
[Backtrader](https://www.backtrader.com/).

## Features
- Fetches daily stock data from Alpha Vantage for BSE/NSE symbols
- Retrieves recent news articles and computes sentiment using a weighted
  blend of TextBlob and the VADER model (70% VADER, 30% TextBlob)
- Calculates RSI and a 20-day moving average
- Opens long positions when RSI falls below 40, price is above its 20-day
  moving average, and sentiment > 0.05
- Opens short positions when RSI rises above 60, price is below its 20-day
  moving average, and sentiment < -0.05
- Basic risk management via position sizing and stop loss
- Command line interface for running backtests

## Usage
```bash
pip install -r requirements.txt
python trading_bot.py --ticker RELIANCE --exchange BSE --backtest
```

Configuration such as API keys resides in `config/config.yaml`.
