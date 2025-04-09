import yfinance as yf
import os
import json

DATA_DIR = "data/tickers"
os.makedirs(DATA_DIR, exist_ok=True)

def download_and_save_tickers(period="12mo", interval="1d"):
    with open("data/large_cap_tickers.json") as f:
        ticker_list = json.load(f)
    for ticker in ticker_list:
        try:
            print(f"📥 Downloading {ticker}...")
            df = yf.download(ticker, period=period, group_by="ticker", auto_adjust=False)
            if df.empty:
                print(f"No data for {ticker}")
                continue

            df.to_csv(f"{DATA_DIR}/{ticker}_12mo.csv")
            print(f"✅ Saved: {ticker}")
          
        except Exception as e:
            print(f"❌ Failed to download {ticker}: {e}")
            return {"status":"error", "message":f"Failed to download {ticker}: {e}"},400
    return {"status":"ok"},200