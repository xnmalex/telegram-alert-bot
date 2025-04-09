import yfinance as yf
from yfinance import EquityQuery
import os
from dotenv import load_dotenv
import json
import pandas as pd
from bs4 import BeautifulSoup
import requests
from datetime import datetime

# Load environment variables from .env
load_dotenv()

# CONFIGURATION
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
WATCHLIST = ['SPY', 'QQQ']

DATA_DIR = "data"
CACHE_FILE = os.path.join(DATA_DIR, "large_cap_tickers.json")
CSV_PATH = os.path.join(DATA_DIR, "sp500_companies.csv")

# Analyze a stock's technical indicators
def analyze_stock(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo")

    if len(df) < 200:
        print(f"Not enough data for {ticker}")
        return

    # Calculate MAs and RSI
    df["50MA"] = df["Close"].rolling(window=50).mean()
    df["200MA"] = df["Close"].rolling(window=200).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    golden_cross = (
        df["50MA"].iloc[-1] > df["200MA"].iloc[-1] and
        df["50MA"].iloc[-2] < df["200MA"].iloc[-2]
    )
    rsi_near_30 = df["RSI"].iloc[-1] < 35

    # Trigger alert
    if golden_cross or rsi_near_30:
        alert = f"*Technical Alert: {ticker}*\n"
        if golden_cross:
            alert += "- Golden Cross detected!\n"
        if rsi_near_30:
            alert += f"- RSI is near oversold: {df['RSI'].iloc[-1]:.2f}\n"
        return alert
        
# Display key indicators for a stock
def generate_summary_message(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="12mo")
        
        print(len(df))

        if len(df) < 100:
            print(f"Not enough data for {ticker}")
            return

        df["50MA"] = df["Close"].rolling(window=50).mean()
        df["200MA"] = df["Close"].rolling(window=200).mean()

        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        current_price = df["Close"].iloc[-1]
        rsi = df["RSI"].iloc[-1]
        ma_50 = df["50MA"].iloc[-1]
        ma_200 = df["200MA"].iloc[-1]
        is_above_50 = current_price > ma_50
        is_above_200 = current_price > ma_200

        summary = (
            f"\nSummary for {ticker}:\n"
            f"- Current Price: ${current_price:.2f}\n"
            f"- RSI: {rsi:.2f}\n"
            f"- 50-day MA: {ma_50:.2f} ({'Above' if is_above_50 else 'Below'})\n"
            f"- 200-day MA: {ma_200:.2f} ({'Above' if is_above_200 else 'Below'})\n"
        )
        return summary
    except Exception as e:
        return f"Error processing {ticker}: {str(e)}"
    
def handle_command(message_text):
    parts = message_text.strip().split()
    
    print(len(parts))
    if parts[0].lower() == '/summary' and len(parts) == 2:
        ticker = parts[1].upper()
        return generate_summary_message(ticker)
    elif parts[0].lower() == '/newhigh':
        return get_new_highs()
    elif parts[0].lower() == '/goldencross':
        return format_tickers_as_text(results=find_golden_cross_tickers(), fields=["ticker", "price", "sma20", "sma50"], dollar_fields=["price", "sma20", "sma50"] )
    elif parts[0].lower() == '/deathcross':
        return format_tickers_as_text(results=find_death_cross_tickers(), fields=["ticker", "price", "sma20", "sma50"], dollar_fields=["price", "sma20", "sma50"] )
    elif parts[0].lower() == '/ma_health':
        return ma_health_alert()
    return None


def analyze_new_highs():
    try:
        url = 'https://finviz.com/screener.ashx?v=111&f=cap_midover%2Cta_highlow52w_nh%2Cta_sma200_pa%2Cta_sma50_pa&ft=3&o=-change'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        print(response)
        # Locate the table containing the screener results
        table = soup.find('table', id="screener-views-table")

        # Extract headers
        headers = [th.text.strip() for th in table.find_all("th")]

        # Extract rows
        rows = table.find_all("tr")[1:]  # Skip header

        # Parse to list of dicts
        data = []
        
        for row in rows:
            cols = row.find_all("td")
            if len(cols) == len(headers):
                entry = {headers[i]: cols[i].text.strip() for i in range(len(cols))}      
                data.append(entry)
        
        with open("data/finviz_new_highs.json", "w") as f:
            json.dump(data, f, indent=4)

        return data, 200
    except Exception as e:
        print(f"error{e}")
        return {"status:":"error", "message":f"bad request {e}"}, 400
    
def truncate(text, max_len=25):
    return text if len(text) <= max_len else text[:max_len - 3] + "..."
        
def get_new_highs():
    analyze_new_highs()
    
    # Load from Finviz-style JSON
    with open("data/finviz_new_highs.json") as f:
        finviz_data = json.load(f)

    messages = []
    for entry in finviz_data:
        ticker = entry.get("Ticker")
        price = entry.get("Price")
        change = entry.get("Change")
        if ticker and price and change:
           
            msg = f"{ticker} - ${price} ({change})"
            messages.append(msg)

    return "\n".join(messages)

def load_or_download_history(ticker, period="6mo", force_refresh=False):
    path = f"data/{ticker}_history_cleaned.csv"

    if os.path.exists(path) and not force_refresh:
        df = pd.read_csv(
                path,
                index_col=0,
                parse_dates=True,
                date_format="%Y-%m-%d"
            )
        return df
       
    df = yf.download(ticker, period=period)
    if not df.empty:
        df.to_csv(path)
    return df

def download_and_cache_bulk(tickers, period="6mo", force_refresh=False):
    import os
    os.makedirs("data/tickers", exist_ok=True)

    # Check which tickers already cached
    to_download = []
    for t in tickers:
        path = f"data/tickers/{t}_{period}.csv"
        if force_refresh or not os.path.exists(path):
            to_download.append(t)

    # Download only missing ones
    if to_download:
        print(f"Downloading {len(to_download)} tickers from yfinance...")
        df_all = yf.download(to_download, period=period, group_by="ticker", auto_adjust=False, progress=False)

        for t in to_download:
            try:
                df = df_all[t]
                if not df.empty:
                    df.to_csv(f"data/tickers/{t}_{period}.csv")
            except Exception as e:
                print(f"⚠️ Failed to save {t}: {e}")
    else:
        print("Using cached data for all tickers.")
        
def load_cached_ticker(ticker, period="6mo"):
    path = f"data/tickers/{ticker}_{period}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f" Cached data not found for {ticker}")
    return pd.read_csv(path, index_col=0, parse_dates=True, date_format="%Y-%m-%d")


def find_golden_cross_tickers(ticker_list=None):
    if ticker_list is None:
        with open("data/large_cap_tickers.json") as f:
            ticker_list = json.load(f)

    goldencross = []

    for ticker in ticker_list:
        try:
            df = load_or_download_history(ticker, period="6mo")
           
            if df.empty or len(df) < 120:
                continue

            df["SMA20"] = df["Close"].rolling(window=20).mean()
            df["SMA50"] = df["Close"].rolling(window=50).mean()
            #df["SMA200"] = df["Close"].rolling(window=200).mean()
            
            sma_slope = df["SMA20"].iloc[-2] - df["SMA20"].iloc[-6]
            latest = df.iloc[-2]
            
            if sma_slope <= 0:
                continue  # skip if not sloping up
           
            #20-day MA crossed above 50-day MA (Death Cross)
            if latest["SMA20"] > latest["SMA50"] and latest["Close"] > latest["SMA20"]:
                goldencross.append({
                    "ticker": ticker,
                    "price": float(round(latest['Close'], 2)),
                    "sma20": float(round(latest['SMA20'], 2)),
                    "sma50": float(round(latest['SMA50'], 2)),
                    "signal": "golden_cross"
                })
           
        except Exception as e:
            print(f"Error with {ticker}: {e}")
            continue

    goldencross = sorted(goldencross, key=lambda x: x["price"], reverse=True)
    return goldencross

def find_death_cross_tickers(ticker_list=None):
    if ticker_list is None:
        with open("data/large_cap_tickers.json") as f:
            ticker_list = json.load(f)

    death_cross = []

    for ticker in ticker_list:
        try:
            df = load_or_download_history(ticker, period="6mo")
           
            if df.empty or len(df) < 120:
                continue

            df["SMA20"] = df["Close"].rolling(window=20).mean()
            df["SMA50"] = df["Close"].rolling(window=50).mean()
            #df["SMA200"] = df["Close"].rolling(window=200).mean()
            
            sma_slope = df["SMA20"].iloc[-2] - df["SMA20"].iloc[-6]
            if sma_slope >= 0:
                continue  # skip if not sloping up

            latest = df.iloc[-2]
            
            #50-day MA crossed below 200-day MA (Death Cross)
            if latest["SMA20"] < latest["SMA50"] and latest["Close"] < latest["SMA20"]:
                death_cross.append({
                    "ticker": ticker,
                    "price": float(round(latest['Close'], 2)),
                    "sma20": float(round(latest['SMA20'], 2)),
                    "sma50": float(round(latest['SMA50'], 2)),
                    "signal": "death cross"
                })
           
        except Exception as e:
            print(f"Error with {ticker}: {e}")
            continue

    death_cross = sorted(death_cross, key=lambda x: x["price"], reverse=True)
    return death_cross[:50]

def fix_csv_format(broken_csv_path, cleaned_csv_path):
    # Skip the first 2 rows and manually define proper column names
    df = pd.read_csv(
        broken_csv_path,
        skiprows=2,
        names=["Date", "Close", "High", "Low", "Open", "Volume"],
        parse_dates=["Date"]
    )

    df.set_index("Date", inplace=True)
    df = df.sort_index()

    # Save back to CSV in yfinance-compatible format
    df.to_csv(cleaned_csv_path)
    print(f"✅ Fixed and saved: {cleaned_csv_path}")
    
def format_tickers_as_text(results, fields, include_header=True, dollar_fields=None):
    """
    Convert a list of dictionaries into CSV-style plain text with optional $ formatting.

    Args:
        results (list): List of dictionaries to format.
        fields (list): Fields to include in the output (order matters).
        include_header (bool): Include the header line.
        dollar_fields (list): Fields that should be shown with "$" and 2 decimal places.

    Returns:
        str: Formatted multiline string.
    """
    if not results:
        return "No results"
    
    dollar_fields = dollar_fields or []
    lines = []

    if include_header:
        lines.append(", ".join(fields))

    for item in results:
        row = []
        for field in fields:
            value = item.get(field, "N/A")
            if field in dollar_fields and isinstance(value, (float, int)):
                value = f"${value:.2f}"
            row.append(str(value))
        lines.append(", ".join(row))

    return "\n".join(lines)


def findTopMarketCap():
    query = EquityQuery('and', [
        EquityQuery('eq', ['region', 'my']),  # Region code for Malaysia
        EquityQuery('gt', ['price.intradaymarketcap', 1_000_000_000])  # Market cap greater than 1 billion
    ])
    
    # Run the screener with the defined query
    results = yf.screen(query)
    
    # Check if any results were returned
    if results:
        for stock in results:
            ticker = stock.get('ticker')
            name = stock.get('shortName')
            market_cap = stock.get('marketCap')
            print(f"Ticker: {ticker}, Name: {name}, Market Cap: {market_cap}")
    else:
        print("No stocks found matching the criteria.")
    
import yfinance as yf
import pandas as pd
import numpy as np

def detect_divergence(tickers, indicator="macd", lookback=30, signal_filter=None):
    print(f"detect: {tickers}")
    results = []
    download_and_cache_bulk(tickers, period="6mo")

    for ticker in tickers:
        try:
            df = load_cached_ticker(ticker, period="6mo")
            df = df[["Close"]].dropna()

            if df.empty or len(df) < lookback:
                continue

            # Add indicator values
            if indicator == "macd":
                df["EMA12"] = df["Close"].ewm(span=12).mean()
                df["EMA26"] = df["Close"].ewm(span=26).mean()
                df["MACD"] = df["EMA12"] - df["EMA26"]
                signal_column = "MACD"

            elif indicator == "rsi":
                delta = df["Close"].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                df["RSI"] = 100 - (100 / (1 + rs))
                signal_column = "RSI"

            df = df.dropna().iloc[-lookback:]

            # Detect divergence
            recent = df.iloc[-1]
            lowest_close_idx = df["Close"].idxmin()
            highest_close_idx = df["Close"].idxmax()

            result = {
                "ticker": ticker,
                "indicator": indicator,
                "price_now": float(round(recent["Close"], 2)),
                "signal": None
            }

            if indicator == "macd":
                lowest_macd_idx = df[signal_column].idxmin()
                macd_then = df.loc[lowest_macd_idx]["MACD"]
                price_then = df.loc[lowest_macd_idx]["Close"]
                macd_now = recent["MACD"]
                price_now = recent["Close"]
                
                macd_slope_pct = ((macd_now - macd_then) / abs(macd_then)) * 100 if macd_then != 0 else 0
                price_change_pct = ((price_now - price_then) / price_then) * 100

                result["macd_slope_pct"] = round(macd_slope_pct, 2)
                result["price_change_pct"] = round(price_change_pct, 2)
                result["macd_now"] = round(macd_now, 4)
                result["macd_then"] = round(macd_then, 4)
                result["price_then"] = round(price_then, 2)
                result["date"] = lowest_macd_idx.strftime("%Y-%m-%d")
                
                if macd_slope_pct > 200:
                    result["strength"] = "strong"
                elif macd_slope_pct > 100:
                    result["strength"] = "moderate"
                elif macd_slope_pct > 50:
                    result["strength"] = "weak"
                else:
                    result["strength"] = "noise"
                
                # Apply proper divergence condition
                if price_now < price_then and macd_now > macd_then:
                    result["signal"] = "bullish_divergence"
                elif price_now > price_then and macd_now < macd_then:
                    result["signal"] = "bearish_divergence"
                    
            elif indicator == "rsi":
                lowest_rsi_idx = df[signal_column].idxmin()
                if df.loc[lowest_close_idx]["Close"] < recent["Close"] and df.loc[lowest_rsi_idx][signal_column] < recent[signal_column]:
                    result["signal"] = "bullish_divergence"
                elif df.loc[highest_close_idx]["Close"] > recent["Close"] and df.loc[highest_close_idx][signal_column] > recent[signal_column]:
                    result["signal"] = "bearish_divergence"

            if result["signal"]:
                if signal_filter is None or result["signal"] == signal_filter and result.get("strength") == "strong":
                    results.append(result)

        except Exception as e:
            print(f"{ticker} error: {e}")
            continue

    results.sort(key=lambda x: x["price_now"], reverse=True)
    return results

def scan_divergence_bulk(indicator="macd", lookback=30, signal_filter=None):
    with open("data/large_cap_tickers.json") as f:
        tickers = json.load(f)
    
    print(f"scan divergence: {tickers}")
      
    try:
        result = detect_divergence(tickers, indicator=indicator, lookback=lookback, signal_filter=signal_filter)
        if result:
               return result
    except Exception as e:
            print(f"error: {e}")
           

def ma_health_stats(tickers):
    download_and_cache_bulk(tickers, period="12mo") 
    count_50 = 0
    count_200 = 0
    count_both = 0
    total = 0

    for ticker in tickers:
        try:
            path = f"data/tickers/{ticker}_12mo.csv"
            df = pd.read_csv(path, skiprows=2, names=["Date", "Close", "High", "Low", "Open", "Volume"], parse_dates=["Date"])
            df.dropna(subset=["Close", "High", "Low"], inplace=True)
            
            df["SMA50"] = df["Close"].rolling(window=50).mean()
            df["SMA200"] = df["Close"].rolling(window=200).mean()
            
           
            latest = df.iloc[-1]
           
            above_50 = latest["Close"] > latest["SMA50"]
            above_200 = latest["Close"] > latest["SMA200"]

            if above_50:
                count_50 += 1
            if above_200:
                count_200 += 1
            if above_50 and above_200:
                count_both += 1

            total += 1
        except Exception as e:
            print(f"Error processing {ticker}: {e} ")
            continue

    if total == 0:
        return {
            "above_50ma": 0,
            "above_200ma": 0,
            "above_both": 0,
            "total": 0
        }
        
    pct_50 = round((count_50 / total) * 100, 2)
    pct_200 = round((count_200 / total) * 100, 2)
    pct_both = round((count_both / total) * 100, 2)
    
    def label_status(pct, thresholds):
        if pct >= thresholds["high"]:
            return "Healthy"
        elif pct >= thresholds["medium"]:
            return "Neutral"
        else:
            return "Weak"

    return {
        "above_50ma": round((count_50 / total) * 100, 2),
        "above_200ma": round((count_200 / total) * 100, 2),
        "above_both": round((count_both / total) * 100, 2),
        "status_50ma": label_status(pct_50, {"high": 60, "medium": 40}),
        "status_200ma": label_status(pct_200, {"high": 50, "medium": 35}),
        "status_overall": label_status(pct_both, {"high": 40, "medium": 25}),
        "total": total
    }
    
def handle_ma_health():
    with open("data/large_cap_tickers.json") as f:
        tickers = json.load(f)

    stats = ma_health_stats(tickers)

    return stats
    

def ma_health_alert():
    stats = handle_ma_health()
    if stats["total"] == 0:
        return "Could not calculate MA health."
        
    return format_market_breadth_telegram_message(stats)
    
def get_strength_emoji(value, ma_type="default"):
    """
    Returns strength label based on MA type:
    - For 50MA and 'both': <40 Weak, 40–60 Moderate, >60 Strong
    - For 200MA: <50 Weak, 50–70 Moderate, >70 Strong
    """
    if ma_type == "200ma":
        if value < 50:
            return "🟥 Weak"
        elif value < 70:
            return "🟧 Moderate"
        else:
            return "🟩 Strong"
    else:
        if value < 40:
            return "🟥 Weak"
        elif value < 60:
            return "🟧 Moderate"
        else:
            return "🟩 Strong"
  
def format_market_breadth_telegram_message(stats):
    """
    Formats the market breadth message for Telegram.
    Uses custom thresholds per MA type, but shows a clean, unified legend.
    
    stats: dict with 'above_50ma', 'above_200ma', 'above_both', 'date' (optional)
    """
    
    # Get today's date in YYYY-MM-DD format
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    msg = []
    msg.append(f"📊 Market Breadth Overview (as of {today_str})\n")

    msg.append(f"% Above 50MA   : {stats['above_50ma']}%  → {get_strength_emoji(stats['above_50ma'], '50ma')}")
    msg.append(f"% Above 200MA  : {stats['above_200ma']}%  → {get_strength_emoji(stats['above_200ma'], '200ma')}")
    msg.append(f"% Above Both   : {stats['above_both']}%  → {get_strength_emoji(stats['above_both'], 'both')}")

    msg.append("\n📘 Legend:")
    msg.append("🟩 Strong")
    msg.append("🟧 Moderate")
    msg.append("🟥 Weak")

    return "\n".join(msg)