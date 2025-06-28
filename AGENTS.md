# Contributing Guide for The Great Sage

This repository contains an experimental trading bot. **It is for educational purposes only and not financial advice.**

## Local setup
- Install dependencies with `pip install -r requirements.txt`.
- Copy `config/config.example.yaml` to `config/config.yaml` and provide your own API keys.

## Development guidelines
- Keep code PEP8 compliant and run `pycodestyle trading_bot.py` after making changes.
- Do not commit real API keys or other secrets.
- Run the following checks before committing:
  1. `python trading_bot.py --help | head -n 20`  # ensure CLI works
  2. `python -m py_compile trading_bot.py`        # syntax check
  3. `pycodestyle trading_bot.py | head -n 20`    # style check
- Use conventional commit style messages.

## Backtesting
Backtesting requires historical data. Without API access you can generate a small
sample DataFrame to test the strategy logic:

```python
import pandas as pd
import numpy as np
from trading_bot import compute_indicators, run_backtest

rng = pd.date_range('2022-01-01', periods=60, freq='D')
price = np.cumsum(np.random.randn(60)) + 100
sample = pd.DataFrame({
    'Open': price + np.random.randn(60)*0.1,
    'High': price + np.random.rand(60)*0.2,
    'Low': price - np.random.rand(60)*0.2,
    'Close': price + np.random.randn(60)*0.1,
    'Volume': np.random.randint(1000, 2000, size=60)
}, index=rng)
run_backtest(compute_indicators(sample), sentiment=0.1)
```
