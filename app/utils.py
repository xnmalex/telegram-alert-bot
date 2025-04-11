import requests
import os
import logging
from dotenv import load_dotenv
import json
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEFAULT_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_alert(message, chat_id=None):
    target_chat = chat_id if chat_id else DEFAULT_CHAT_ID
    logging.info(f"chatid: {target_chat}")
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": target_chat,
        "text": message,
        "parse_mode": "Markdown"
    }
    requests.post(url, data=payload)

def get_nasdaq_100_constituents():
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    tables = pd.read_html(url)
    
    # Usually the first table on the page
    df = tables[4]  # As of 2024, table 3 is the current list
    tickers = df["Ticker"].tolist()

    # Clean up symbols (e.g., BRK.B -> BRK-B for yfinance)
    tickers = [t.replace(".", "-") for t in tickers]

    return tickers, df

def load_tickers_from_json(path="data/large_cap_tickers.json"):
    """
    Load tickers from a JSON file.
    Format: ["AAPL", "MSFT", "GOOGL", ...]
    """
    try:
        with open(path, "r") as f:
            tickers = json.load(f)
        return tickers
    except Exception as e:
        print(f"Error loading tickers from {path}: {e}")
        return []


def read_tickers_df(ticker, period="12mo"):
    try:
        path = f"data/tickers/{ticker}_{period}.csv"
        df = pd.read_csv(path, skiprows=2, names=["Date","Open","High","Low","Close","Adj Close","Volume"], parse_dates=["Date"])
        return df
    except Exception as e:
        raise Exception(f"Error: {e}")