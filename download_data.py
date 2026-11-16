"""Optional utility to download OHLCV price history for the coursework demo.

This uses a third-party service; download availability and formatting can change.
Review Yahoo Finance's terms before using market data.
"""
from pathlib import Path
import pandas as pd
import yfinance as yf

TICKERS = ("AAPL", "GOOGL", "MSFT")
COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


def download_stock(ticker):
    prices = yf.download(
        ticker, start="2008-01-01", end="2023-01-01",
        auto_adjust=False, progress=False,
    )
    if prices.empty:
        raise RuntimeError(f"No prices returned for {ticker}")
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    prices = prices.reset_index()
    missing = set(COLUMNS) - set(prices.columns)
    if missing:
        raise RuntimeError(f"Missing fields for {ticker}: {sorted(missing)}")
    path = Path(ticker.lower() + ".csv")
    prices[COLUMNS].to_csv(path, index=False)
    print(f"Wrote {path} ({len(prices)} observations)")


if __name__ == "__main__":
    for ticker in TICKERS:
        download_stock(ticker)
