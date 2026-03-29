import pandas as pd


import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ----------------------------
# Load Portfolio
# ----------------------------
df = pd.read_csv("data/20260211_portfolio.csv")

df = df[df["Slice"] != "Total"]
df["Value"] = pd.to_numeric(df["Value"])

total_value = df["Value"].sum()
df["Weight"] = df["Value"] / total_value

tickers = df["Slice"].tolist()

print("\nPortfolio Weights:")
print(df[["Slice", "Weight"]])

# ----------------------------
# Download Price Data
# ----------------------------
benchmark = "URTH"

all_tickers = tickers + [benchmark]

prices = yf.download(all_tickers, period="3y")["Adj Close"]

returns = prices.pct_change().dropna()

# ----------------------------
# Portfolio Returns
# ----------------------------
weights = df.set_index("Slice")["Weight"]
portfolio_returns = (returns[tickers] @ weights)

# ----------------------------
# Risk Metrics
# ----------------------------
vol = portfolio_returns.std() * np.sqrt(252)
benchmark_vol = returns[benchmark].std() * np.sqrt(252)

beta = np.cov(portfolio_returns, returns[benchmark])[0,1] / np.var(returns[benchmark])

print("\nPortfolio Vol:", round(vol,4))
print("Benchmark Vol:", round(benchmark_vol,4))
print("Portfolio Beta:", round(beta,4))

# ----------------------------
# Max Drawdown
# ----------------------------
cum_returns = (1 + portfolio_returns).cumprod()
rolling_max = cum_returns.cummax()
drawdown = cum_returns / rolling_max - 1

max_dd = drawdown.min()

print("Max Drawdown:", round(max_dd,4))