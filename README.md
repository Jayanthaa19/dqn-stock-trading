# Multi-Stock Trading with Reinforcement Learning

**Academic group project | Knowledge Representation and Reasoning | Python / PyTorch**

This coursework compares **Tabular Q-Learning** with a **Deep Q-Network (DQN)** in a simplified stock-trading simulation using historical price data for **Apple (AAPL)**, **Alphabet (GOOGL)** and **Microsoft (MSFT)**. The agent selects one of three actions: **Buy**, **Sell** or **Hold**.

> **Educational demonstration, not investment advice.** The reward graphs reflect an experimental training reward on scaled price changes. They do **not** measure real portfolio returns, trading profitability or out-of-sample performance.

## My contribution and group context

This was **group coursework**, not a solo project. The submitted report credits **Jayanthaa Dhanabal** with developing the Tabular Q-Learning and DQN policies and generating the cumulative-reward results. Team members contributed to the main functions/training and data preprocessing/visual analysis. This repository is a cleaned portfolio presentation of the group's coursework code, with attribution to the team.

## Methods

- **Market inputs:** OHLCV (open, high, low, close, volume) and adjusted closing price; RSI (14-day window) is computed from closing prices.
- **State features:** Open, High, Low, Close and RSI, min-max scaled.
- **Actions:** Buy, Sell and Hold.
- **Tabular Q-Learning:** Discretises each scaled feature into 10 bins and updates a Q-table with an epsilon-greedy policy.
- **DQN:** Three-layer feed-forward PyTorch network, experience-replay buffer, target network and epsilon-greedy action selection.
- **Outputs:** Per-episode training rewards and policy-comparison charts. Trained network weights are saved locally to `models/`.

## Repository files

| File | Purpose |
| --- | --- |
| `main.py` | Loads data, prepares features, trains each policy and compares reward traces |
| `policy.py` | Tabular Q-Learning and DQN training implementations |
| `model.py` | Q-network, replay buffer, save/load helpers |
| `download_data.py` | Optional script to obtain historical prices from Yahoo Finance via `yfinance` |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Excludes local environments, CSV data, model weights and caches |

## Set up and run

1. Install **Python 3.10+**, preferably inside a virtual environment.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Obtain the historical CSVs. If your use of Yahoo Finance data is permitted, run:

   ```bash
   python download_data.py
   ```

   Alternatively, provide authorised files named `aapl.csv`, `googl.csv` and `msft.csv` in the project directory. They must contain the columns `Open`, `High`, `Low`, `Close`, `Adj Close` and `Volume`; a `Date` column is recommended. The loader also recognises the multi-row CSV metadata format in the original coursework archive. **The raw data are intentionally not included in this repository.**

4. For a short smoke test, use:

   ```bash
   python main.py --episodes 1 --no-plots
   ```

5. For the coursework's default 20-episode training run and interactive charts:

   ```bash
   python main.py
   ```

The optional `yfinance` download uses the historical date interval **2008-01-01 to 2023-01-01** (end date exclusive). Data availability and adjusted-price conventions may change over time.

## Implementation notes

- The public version removes the local Python environment and IDE settings and fixes **two practical issues** in the archived code: reading multi-row Yahoo Finance CSV headers and binning each scaled state feature independently for Tabular Q-Learning. It also provides ticker-labelled plots and a configurable episode count.
- The coursework scales data using all available dates before training. There is **no chronological train/test separation** or unseen-period trading backtest in this code.
- The reward is based on the next-day movement of a **normalised closing price**, not on a portfolio ledger. The experiment does not model cash, holdings, position limits, execution slippage, transaction costs, taxes or risk.
- Consequently, summed rewards across training episodes cannot be interpreted as monetary gains or robust evidence that one strategy outperforms another in real markets.
- The cleaned code has been checked for syntax and lightly smoke-tested; it has **not** been independently validated as a real-world trading strategy.

## Potential improvements

Introduce a time-based split with a properly fitted training-only scaler, a holdings-aware trading environment, explicit fees/slippage, transaction logs, realistic backtesting and multiple random seeds. Compare performance against buy-and-hold and report risk-adjusted metrics.

## Acknowledgement

Created as MSc academic group coursework. Market price data, when downloaded separately, come from a third-party source and remain subject to the data provider's terms.
