"""Coursework demonstration: Tabular Q-learning versus DQN on three stocks.

Run ``python download_data.py`` to obtain the CSVs first, or provide
permitted local CSVs with the same columns.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from model import save_model
from policy import tabular_q_policy, train_dqn

TICKERS = ("AAPL", "GOOGL", "MSFT")
STATE_FEATURES = ["Open", "High", "Low", "Close", "RSI"]
ACTION_SPACE = ["Buy", "Sell", "Hold"]
NUMERIC_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def read_stock_csv(filename):
    """Read flat CSVs or yfinance exports with two metadata header rows."""
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {filename}. Run 'python download_data.py' or provide "
            "an authorised OHLCV CSV with this name."
        )

    data = pd.read_csv(path)
    # Some Yahoo/yfinance CSV exports have `Price`, `Ticker`, `Date` as their
    # first-column entries in three successive header/metadata rows.
    if data.columns[0] == "Price" and len(data) >= 2 and str(data.iloc[0, 0]) == "Ticker":
        data = pd.read_csv(path, skiprows=[1, 2])
    if "Price" in data.columns and "Date" not in data.columns:
        data = data.rename(columns={"Price": "Date"})

    missing = set(NUMERIC_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"{filename} is missing required columns: {sorted(missing)}")
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.dropna(subset=["Date"]).sort_values("Date")
    return data.reset_index(drop=True)


def load_and_preprocess_data(data):
    """Apply the coursework's RSI calculation and full-sample min-max scaling."""
    data = data.copy()
    for col in NUMERIC_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=NUMERIC_COLUMNS)

    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    relative_strength = gain / loss
    data["RSI"] = 100 - (100 / (1 + relative_strength))
    data = data.replace([float("inf"), float("-inf")], float("nan"))
    data = data.dropna(subset=NUMERIC_COLUMNS + ["RSI"])

    if len(data) < 2:
        raise ValueError("Not enough usable rows after calculating RSI (need at least 16 days).")

    features = NUMERIC_COLUMNS + ["RSI"]
    scaler = MinMaxScaler()
    data[features] = pd.DataFrame(
        scaler.fit_transform(data[features]), index=data.index, columns=features
    )
    return data.reset_index(drop=True)


def plot_rewards(tabular_rewards, dqn_rewards):
    plt.figure(figsize=(10, 6))
    for ticker, rewards in zip(TICKERS, tabular_rewards):
        plt.plot(rewards, label=f"{ticker} Tabular Q-Learning")
    plt.title("Tabular Q-Learning: training episode rewards")
    plt.xlabel("Training episode")
    plt.ylabel("Simulated reward (scaled-price units)")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    for ticker, rewards in zip(TICKERS, dqn_rewards):
        plt.plot(rewards, label=f"{ticker} DQN")
    plt.title("Deep Q-Network: training episode rewards")
    plt.xlabel("Training episode")
    plt.ylabel("Simulated reward (scaled-price units)")
    plt.legend()
    plt.tight_layout()

    for ticker, tab, dqn in zip(TICKERS, tabular_rewards, dqn_rewards):
        plt.figure(figsize=(6, 4))
        plt.bar(["Tabular Q", "DQN"], [sum(tab), sum(dqn)])
        plt.title(f"{ticker}: summed training episode rewards")
        plt.ylabel("Simulated reward (scaled-price units)")
        plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20,
                        help="Training episodes per stock and policy (default: 20)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip interactive graphs (useful for test runs)")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    datasets = []
    for ticker in TICKERS:
        raw = read_stock_csv(ticker.lower() + ".csv")
        processed = load_and_preprocess_data(raw)
        datasets.append(processed)
        print(f"{ticker}: {len(processed)} usable trading days")

    tabular_rewards = []
    for ticker, data in zip(TICKERS, datasets):
        print(f"Tabular Q-Learning: {ticker}")
        tabular_rewards.append(tabular_q_policy(
            data, STATE_FEATURES, episodes=args.episodes))

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    dqn_rewards = []
    for ticker, data in zip(TICKERS, datasets):
        print(f"DQN: {ticker}")
        trained_model, rewards = train_dqn(
            data, STATE_FEATURES, ACTION_SPACE, episodes=args.episodes)
        dqn_rewards.append(rewards)
        save_model(trained_model, str(model_dir / f"{ticker.lower()}_dqn.pth"))

    if not args.no_plots:
        plot_rewards(tabular_rewards, dqn_rewards)
    print("Completed coursework demonstration. Rewards are NOT realised investment profit.")


if __name__ == "__main__":
    main()
