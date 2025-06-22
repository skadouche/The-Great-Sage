import os
import sys
import yaml
import logging
import click
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from textblob import TextBlob
from datetime import datetime, timedelta
import backtrader as bt
from alpha_vantage.timeseries import TimeSeries
from ta.momentum import RSIIndicator

# ========== CONFIGURATION & LOGGING ===========

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(log_path='logs/trading.log'):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )

# ========== DATA ACQUISITION ===========

def fetch_stock_data(ticker, period="6mo", config=None):
    ts = get_alpha_vantage_client(config)
    data, meta = ts.get_daily(symbol=f"{ticker}.NS", outputsize='full')
    df = data.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    })
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.sort_index()
    # Filter by period
    if period == "6mo":
        cutoff = pd.Timestamp.today() - pd.DateOffset(months=6)
        df = df[df.index >= cutoff]
    df.dropna(inplace=True)
    return df

def fetch_newsapi_articles(api_key, query, count=100):
    try:
        url = (
            f'https://newsapi.org/v2/everything?'
            f'q={query}&language=en&sortBy=publishedAt&pageSize={min(count,100)}&apiKey={api_key}'
        )
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        articles = data.get('articles', [])
        return [article['title'] + ' ' + article.get('description', '') for article in articles]
    except Exception as e:
        logging.error(f"Error fetching NewsAPI articles: {e}")
        return []

def analyze_sentiment(posts):
    if not posts:
        return 0.0
    polarity = [TextBlob(post).sentiment.polarity for post in posts]
    return np.mean(polarity)

def get_alpha_vantage_client(config):
    return TimeSeries(key=config['alphavantage']['api_key'], output_format='pandas')

# ========== TECHNICAL ANALYSIS ===========

def compute_rsi(df, period=14):
    rsi_indicator = RSIIndicator(close=df['Close'], window=period)
    df['RSI'] = rsi_indicator.rsi()
    return df

# ========== RISK MANAGEMENT ===========

def calculate_position_size(capital, price, risk_pct=0.01):
    risk_amount = capital * risk_pct
    qty = int(risk_amount // price)
    return max(qty, 1)

# ========== STRATEGY LOGIC ===========

def generate_signals(df, sentiment, holding_days):
    signals = []
    in_short = False
    entry_price = 0
    entry_index = None

    for i in range(len(df)):
        rsi = df['RSI'].iloc[i]
        close = df['Close'].iloc[i]
        date = df.index[i]

        # Entry condition
        if not in_short and rsi > 70 and sentiment < -0.5:
            signals.append({'date': date, 'signal': 'SELL', 'price': close})
            in_short = True
            entry_price = close
            entry_index = i

        # Exit condition
        elif in_short and entry_index is not None:
            days_held = i - entry_index
            stop_loss_price = entry_price * 1.02
            if rsi < 30 or days_held >= holding_days or close > stop_loss_price:
                signals.append({'date': date, 'signal': 'BUY', 'price': close})
                in_short = False
                entry_price = 0
                entry_index = None

    return signals

# ========== BACKTESTING ===========

class ShortStrategy(bt.Strategy):
    params = (
        ('sentiment', 0.0),
        ('holding_days', 5),
        ('risk_pct', 0.01),
        ('stop_loss_pct', 0.02),
        ('capital', 100000),
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.datas[0], period=14)
        self.order = None
        self.entry_price = None
        self.entry_bar = None

    def next(self):
        if not self.position:
            if self.rsi[0] > 70 and self.p.sentiment < -0.5:
                size = calculate_position_size(self.p.capital, self.data.close[0], self.p.risk_pct)
                self.order = self.sell(size=size)
                self.entry_price = self.data.close[0]
                self.entry_bar = len(self)
        elif self.entry_bar is not None:
            days_held = len(self) - self.entry_bar
            stop_loss_price = self.entry_price * (1 + self.p.stop_loss_pct)
            if self.rsi[0] < 30 or days_held >= self.p.holding_days or self.data.close[0] > stop_loss_price:
                self.order = self.buy(size=self.position.size)
                self.entry_bar = None

def run_backtest(df, sentiment, capital=100000):
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(capital)
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(ShortStrategy, sentiment=sentiment)
    results = cerebro.run()
    strat = results[0]
    portfolio_value = cerebro.broker.getvalue()
    returns = (portfolio_value - capital) / capital
    print(f"Total Return: {returns*100:.2f}%")
    cerebro.plot()

# ========== CLI ===========

@click.command()
@click.option('--ticker', prompt='Stock ticker (NSE)', help='NSE stock ticker symbol (e.g., RELIANCE)')
@click.option('--capital', default=100000, help='Total trading capital')
@click.option('--backtest', is_flag=True, help='Run backtest (Alpha Vantage only)')
def main(ticker, capital, backtest):
    setup_logging()
    config = load_config()
    logging.info(f"Starting bot for {ticker}")

    # Data acquisition
    df = fetch_stock_data(ticker, config=config)
    df = compute_rsi(df)
    newsapi_key = config['newsapi']['api_key']
    articles = fetch_newsapi_articles(newsapi_key, ticker, count=100)
    sentiment = analyze_sentiment(articles)
    logging.info(f"Sentiment for {ticker}: {sentiment:.2f}")

    if backtest:
        run_backtest(df, sentiment, capital)
        return

    logging.info("Live trading is not supported with Alpha Vantage. Please use --backtest.")

if __name__ == '__main__':
    main()  # Click handles CLI args
