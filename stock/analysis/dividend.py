import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def show(ticker: str, scale: int=10):
  # Step 1: Fetch monthly stock price history
  bito = yf.Ticker(ticker)
  hist = bito.history(period="max", interval="1mo").reset_index()
  hist = hist[["Date", "Close"]]
  hist["Adj_Close_100"] = hist["Close"] / scale  # scale down for plotting
  
  # Ensure hist["Date"] is timezone-naive
  hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
  
  # Step 2: Fetch dividend data
  dividends_raw = bito.dividends.reset_index()
  dividends_raw.columns = ["Date", "Dividend"]
  
  # Remove timezone before processing
  dividends_raw["Date"] = dividends_raw["Date"].dt.tz_localize(None)
  
  # Only keep non-zero dividends
  dividends_raw = dividends_raw[dividends_raw["Dividend"] > 0]
  
  # Convert to month-level datetime (first of month)
  dividends_raw["Date"] = dividends_raw["Date"].dt.to_period("M").dt.to_timestamp()
  
  # Sum dividends if multiple in same month
  dividends_monthly = dividends_raw.groupby("Date", as_index=False).sum()
  
  # Step 3: Merge price and dividend data
  merged = pd.merge(hist, dividends_monthly, on="Date", how="left")
  merged["Dividend"].fillna(0, inplace=True)
  
  # Step 4: Plot both series
  plt.figure(figsize=(12, 6))
  plt.plot(merged["Date"], merged["Dividend"], marker="o", label="Dividend (USD)")
  plt.plot(merged["Date"], merged["Adj_Close_100"], marker="x", linestyle="--", label="Stock Price / 100 (USD)")
  plt.title("BITO Monthly Dividends and Adjusted Stock Price")
  plt.xlabel("Date")
  plt.ylabel("Amount (USD)")
  plt.grid(True)
  plt.legend()
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.show()

show("BITO", scale=10)
