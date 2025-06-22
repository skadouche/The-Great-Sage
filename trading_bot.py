import os
import yaml
import logging
import click
import pandas as pd
import numpy as np
import requests
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import matplotlib
matplotlib.use('Agg', force=True)
nltk.download('vader_lexicon', quiet=True)
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

def fetch_stock_data(ticker, period="6mo", exchange="BSE", config=None):
    """Fetch daily stock data from Alpha Vantage.

    The exchange parameter allows pulling either BSE or NSE symbols
    supported by Alpha Vantage. Defaults to BSE as coverage is more
    reliable for Indian equities.
    """
    ts = get_alpha_vantage_client(config)
    symbol = f"{ticker}.{exchange}"
    data, meta = ts.get_daily(symbol=symbol, outputsize='full')
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
    """Return combined polarity score using TextBlob and VADER."""
    if not posts:
        return 0.0
    blob_scores = [TextBlob(post).sentiment.polarity for post in posts]
    sia = SentimentIntensityAnalyzer()
    vader_scores = [sia.polarity_scores(post)['compound'] for post in posts]
    return float(np.mean(blob_scores + vader_scores))

def get_alpha_vantage_client(config):
    return TimeSeries(key=config['alphavantage']['api_key'], output_format='pandas')

# ========== TECHNICAL ANALYSIS ===========

def compute_indicators(df, rsi_period=14, ma_period=20):
    """Add RSI and moving average indicators to the dataframe."""
    rsi_indicator = RSIIndicator(close=df['Close'], window=rsi_period)
    df['RSI'] = rsi_indicator.rsi()
    df['MA'] = df['Close'].rolling(window=ma_period).mean()
    df.dropna(inplace=True)
    return df

# ========== RISK MANAGEMENT ===========

def calculate_position_size(capital, price, risk_pct=0.01):
    risk_amount = capital * risk_pct
    qty = int(risk_amount // price)
    return max(qty, 1)

# ========== STRATEGY LOGIC ===========

def generate_signals(df, sentiment, holding_days):
    """Generate long or short signals based on RSI, MA and sentiment."""
    signals = []
    position = None
    entry_price = 0
    entry_index = None

    for i in range(len(df)):
        rsi = df['RSI'].iloc[i]
        close = df['Close'].iloc[i]
        date = df.index[i]

        if position is None:
            if rsi < 45 and sentiment > 0.1:
                position = 'LONG'
                entry_price = close
                entry_index = i
                signals.append({'date': date, 'signal': 'BUY', 'price': close})
            elif rsi > 55 and sentiment < -0.1:
                position = 'SHORT'
                entry_price = close
                entry_index = i
                signals.append({'date': date, 'signal': 'SELL', 'price': close})
        else:
            days_held = i - entry_index
            stop_loss = entry_price * (1 - 0.02) if position == 'LONG' else entry_price * (1 + 0.02)
            take_profit = entry_price * (1 + 0.02) if position == 'LONG' else entry_price * (1 - 0.02)
            exit_signal = False
            if position == 'LONG':
                exit_signal = rsi > 55 or close < stop_loss or close > take_profit
            else:
                exit_signal = rsi < 45 or close > stop_loss or close < take_profit
            if exit_signal or days_held >= holding_days:
                signals.append({'date': date, 'signal': 'CLOSE', 'price': close})
                position = None
                entry_price = 0
                entry_index = None

    return signals

# ========== BACKTESTING ===========

class LongShortStrategy(bt.Strategy):
    params = (
        ('sentiment', 0.0),
        ('holding_days', 5),
        ('risk_pct', 0.01),
        ('stop_loss_pct', 0.02),
        ('capital', 100000),
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.datas[0], period=14)
        self.ma = bt.indicators.SimpleMovingAverage(self.datas[0].close, period=20)
        self.order = None
        self.entry_price = None
        self.entry_bar = None
        self.direction = None

    def next(self):
        if not self.position:
            if (self.rsi[0] < 45 and self.p.sentiment > 0.1):
                size = calculate_position_size(self.p.capital, self.data.close[0], self.p.risk_pct)
                self.order = self.buy(size=size)
                self.entry_price = self.data.close[0]
                self.entry_bar = len(self)
                self.direction = 'long'
            elif (self.rsi[0] > 55 and self.p.sentiment < -0.1):
                size = calculate_position_size(self.p.capital, self.data.close[0], self.p.risk_pct)
                self.order = self.sell(size=size)
                self.entry_price = self.data.close[0]
                self.entry_bar = len(self)
                self.direction = 'short'
        elif self.entry_bar is not None:
            days_held = len(self) - self.entry_bar
            if self.direction == 'long':
                stop_loss = self.entry_price * (1 - self.p.stop_loss_pct)
                take_profit = self.entry_price * (1 + self.p.stop_loss_pct)
                exit_cond = (self.rsi[0] > 55 or self.data.close[0] < stop_loss
                             or self.data.close[0] > take_profit)
            else:
                stop_loss = self.entry_price * (1 + self.p.stop_loss_pct)
                take_profit = self.entry_price * (1 - self.p.stop_loss_pct)
                exit_cond = (self.rsi[0] < 45 or self.data.close[0] > stop_loss
                             or self.data.close[0] < take_profit)
            if exit_cond or days_held >= self.p.holding_days:
                self.order = self.close()
                self.entry_bar = None
                self.direction = None

def run_backtest(df, sentiment, capital=100000):
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(capital)
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(LongShortStrategy, sentiment=sentiment)
    results = cerebro.run()
    portfolio_value = cerebro.broker.getvalue()
    returns = (portfolio_value - capital) / capital
    print(f"Total Return: {returns*100:.2f}%")
    logging.info(f"Backtest return: {returns*100:.2f}%")
    try:
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        cerebro.plot(style='candlestick')
    except Exception as e:
        logging.error(f"Plotting failed: {e}")

# ========== CLI ===========

@click.command()
@click.option('--ticker', prompt='Stock ticker', help='Stock ticker symbol, e.g., RELIANCE')
@click.option('--exchange', default='BSE', show_default=True, help='Exchange suffix such as BSE or NSE')
@click.option('--capital', default=100000, help='Total trading capital')
@click.option('--backtest', is_flag=True, help='Run backtest (Alpha Vantage only)')
def main(ticker, exchange, capital, backtest):
    setup_logging()
    config = load_config()
    logging.info(f"Starting bot for {ticker}")

    # Data acquisition
    df = fetch_stock_data(ticker, config=config, exchange=exchange)
    df = compute_indicators(df)
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
