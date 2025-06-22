# The Great Sage Trading Bot

This repository contains an experimental trading bot for Indian equities.
It fetches market data from Alpha Vantage, performs headline sentiment
analysis combining TextBlob with the VADER model from NLTK, and executes
a simple long/short strategy that can be backtested with
[Backtrader](https://www.backtrader.com/).

## Features
- Fetches daily stock data from Alpha Vantage for BSE/NSE symbols
- Retrieves recent news articles and computes sentiment with TextBlob and
  the VADER model for better accuracy
- Calculates RSI and a 20-day moving average
- Opens long positions when RSI is oversold and sentiment is bullish
- Opens short positions when RSI is overbought and sentiment is bearish
- Basic risk management via position sizing and stop loss
- Command line interface for running backtests

## Usage
```bash
pip install -r requirements.txt
python trading_bot.py --ticker RELIANCE --exchange BSE --backtest
```

Configuration such as API keys resides in `config/config.yaml`.
