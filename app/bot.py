import yfinance as yf
from yfinance import EquityQuery
import os
from dotenv import load_dotenv
import json
import pandas as pd
from bs4 import BeautifulSoup
import requests
from scipy.signal import argrelextrema
from scipy.stats import linregress 
import numpy as np
import traceback
from app.utils.telegram_utils import format_market_breadth_telegram_message
from app.utils.tickers_utils import load_tickers_from_json, read_tickers_df
from ta.momentum import RSIIndicator
from ta.trend import MACD
import logging

# Load environment variables from .env
load_dotenv()

# CONFIGURATION
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
WATCHLIST = ['SPY', 'QQQ']

DATA_DIR = "data"
CACHE_FILE = os.path.join(DATA_DIR, "large_cap_tickers.json")
CSV_PATH = os.path.join(DATA_DIR, "sp500_companies.csv")

def get_rsi(df): return RSIIndicator(close=df["Close"], window=14).rsi()
def get_macd(df):
    macd = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9) 
    return macd.macd()    

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

    df["RSI"] = get_rsi(df)

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

        df["RSI"] = get_rsi(df)

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
    
    try:
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
        elif parts[0].lower() == '/long' and len(parts) == 2:
            ticker = parts[1].upper()
            return trade(ticker, side="long")
        elif parts[0].lower() == '/short' and len(parts) == 2:
            ticker = parts[1].upper()
            return trade(ticker, side="short")
    except Exception as e:
        logging.info(f"error handling command {e}")
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

def find_golden_cross_tickers(ticker_list=None):
    if ticker_list is None:
        with open("data/large_cap_tickers.json") as f:
            ticker_list = json.load(f)

    goldencross = []

    for ticker in ticker_list:
        try:
            df = read_tickers_df(ticker)
           
            if df.empty or len(df) < 120:
                continue

            df["SMA20"] = df["Close"].rolling(window=20).mean()
            df["SMA50"] = df["Close"].rolling(window=50).mean()
            #df["SMA200"] = df["Close"].rolling(window=200).mean()
            
            sma_slope = df["SMA20"].iloc[-1] - df["SMA20"].iloc[-6]
            latest = df.iloc[-1]
            
            if sma_slope <= 0:
                continue  # skip if not sloping up
           
            # Find crossover (SMA20 > SMA50) within last 3 bars
            signal_found = False
            for i in range(-3, 0):
                if df["SMA20"].iloc[i - 1] < df["SMA50"].iloc[i - 1] and df["SMA20"].iloc[i] > df["SMA50"].iloc[i]:
                    signal_found = True
                    signal_date = df["Date"].iloc[i]
                    break

            if not signal_found:
                continue

            latest = df.iloc[-1]
            
            if latest["Close"] > latest["SMA20"]:
                goldencross.append({
                    "ticker": ticker,
                    "price": float(round(latest['Close'], 2)),
                    "sma20": float(round(latest['SMA20'], 2)),
                    "sma50": float(round(latest['SMA50'], 2)),
                    "signal": "golden_cross",
                    "signal_date": signal_date
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
            df = read_tickers_df(ticker)
           
            if df.empty or len(df) < 120:
                continue

            df["SMA20"] = df["Close"].rolling(window=20).mean()
            df["SMA50"] = df["Close"].rolling(window=50).mean()
            #df["SMA200"] = df["Close"].rolling(window=200).mean()
            
            sma_slope = df["SMA20"].iloc[-1] - df["SMA20"].iloc[-6]
            if sma_slope >= 0:
                continue  # skip if not sloping up

            latest = df.iloc[-1]
            
             # Find crossover (SMA20 > SMA50) within last 3 bars
            signal_found = False
            for i in range(-3, 0):
                if df["SMA20"].iloc[i - 1] > df["SMA50"].iloc[i - 1] and df["SMA20"].iloc[i] < df["SMA50"].iloc[i]:
                    signal_found = True
                    signal_date = df["Date"].iloc[i]
                    break

            if not signal_found:
                continue

            latest = df.iloc[-1]
            
            if latest["Close"] > latest["SMA20"]:
                death_cross.append({
                    "ticker": ticker,
                    "price": float(round(latest['Close'], 2)),
                    "sma20": float(round(latest['SMA20'], 2)),
                    "sma50": float(round(latest['SMA50'], 2)),
                    "signal": "death_cross",
                    "signal_date": signal_date
                })
        except Exception as e:
            print(f"Error with {ticker}: {e}")
            continue

    death_cross = sorted(death_cross, key=lambda x: x["price"], reverse=True)
    return death_cross[:50]
    
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
    
def detect_divergence(tickers, indicator="macd", lookback=30, signal_filter=None):
    results = []

    for ticker in tickers:
        try:
            df = read_tickers_df(ticker)
            df = df.dropna()

            if df.empty or len(df) < lookback:
                continue

            # Add indicator values
            if indicator == "macd":
                df["MACD"] = get_macd(df)
                
                signal_column = "MACD"

            elif indicator == "rsi":
                df["RSI"] = get_rsi(df)
                signal_column = "RSI"

            df = df.dropna().iloc[-lookback:]
            
            # Detect divergence
            recent = df.iloc[-1]
        
            result = {
                "ticker": ticker,
                "indicator": indicator,
                "price_now": float(round(recent["Close"], 2)),
                "signal": None
            }
            
            # Find indices of swing lows
            swing_low_idx = argrelextrema(df[signal_column].values, np.less_equal, order=3)[0]

            # Initialize column with NaN
            df["macd_swing_low"] = np.nan

            # Assign swing lows only at the detected positions
            df.iloc[swing_low_idx, df.columns.get_loc("macd_swing_low")] = df.iloc[swing_low_idx][signal_column].values
            # df["macd_swing_high"] = df[signal_column][argrelextrema(df[signal_column].values, np.greater_equal, order=3)[0]]
           
            if indicator == "macd" and len(swing_low_idx) >= 2:
               
                lowest_macd_idx = swing_low_idx[-2]
                        
                macd_then = df.iloc[lowest_macd_idx]["MACD"]
                price_then = df.iloc[lowest_macd_idx]["Close"]
                macd_now = recent["MACD"]
                price_now = recent["Close"]
                           
                
                macd_slope_pct = ((macd_now - macd_then) / abs(macd_then)) * 100 if macd_then != 0 else 0
                price_change_pct = ((price_now - price_then) / price_then) * 100
                
                macd_then_date = df.iloc[lowest_macd_idx]["Date"]

                result["macd_slope_pct"] = round(macd_slope_pct, 2)
                result["price_change_pct"] = round(price_change_pct, 2)
                result["macd_now"] = round(macd_now, 4)
                result["macd_then"] = round(macd_then, 4)
                result["price_then"] = round(price_then, 2)
                result["macd_then_date"] = macd_then_date
                
                divergence_strength = abs(macd_slope_pct) * abs(price_change_pct)
                
                if divergence_strength > 1000:
                    result["strength"] = "strong"
                elif divergence_strength > 500:
                    result["strength"] = "moderate"
                elif divergence_strength > 200:
                    result["strength"] = "weak"
                else:
                    result["strength"] = "noise"
                
                result["divergence_score"] = round(divergence_strength, 2)
                # Apply proper divergence condition
                if price_now < price_then and macd_now > macd_then:
                    result["signal"] = "bullish_divergence"
                elif price_now > price_then and macd_now < macd_then:
                    result["signal"] = "bearish_divergence"
                    
            elif indicator == "rsi":
                lowest_macd_idx = swing_low_idx[-2]
                # if df.loc[lowest_close_idx]["Close"] < recent["Close"] and df.loc[lowest_rsi_idx][signal_column] < recent[signal_column]:
                #     result["signal"] = "bullish_divergence"
                # elif df.loc[highest_close_idx]["Close"] > recent["Close"] and df.loc[highest_close_idx][signal_column] > recent[signal_column]:
                #     result["signal"] = "bearish_divergence"

            
            if result["signal"] and result["signal"] == signal_filter:
                # print(f"📊 Signal: {result['signal']} | MACD slope: {macd_slope_pct:.2f}% | Strength: {result.get('strength')}")
                if result.get("strength") == "strong" or result.get("strength") == "moderate":
                    results.append(result)

        except Exception as e:
            print(f"{ticker} error: {e}")
            traceback.print_exc()
            continue

    if results:
        results.sort(key=lambda x: x["divergence_score"], reverse=True)
    return results[:50]

def detect_combined_divergence(df, lookback=60, trend_bars=20):
    df = df[-lookback:].copy()

    # === MACD Calculation ===
    df["MACD"] = get_macd(df)

    # === RSI Calculation ===
    df["RSI"] = get_rsi(df)

    if len(df) < trend_bars + 2:
        return []

    macd_now = df.iloc[-1]["MACD"]
    rsi_now = df.iloc[-1]["RSI"]
    price_now = df.iloc[-1]["Close"]

    macd_trend = "up" if linregress(np.arange(trend_bars), df["MACD"].iloc[-trend_bars:]).slope > 0 else "down"
    rsi_trend = "up" if linregress(np.arange(trend_bars), df["RSI"].iloc[-trend_bars:]).slope > 0 else "down"
    signals = []

    # Detect Bullish Divergence
    if macd_trend == "up":
        macd_then_idx = df[-trend_bars:]["MACD"].idxmin()
        macd_then = df.loc[macd_then_idx]["MACD"]
        price_then = df.loc[macd_then_idx]["Close"]
        macd_date = df.loc[macd_then_idx]["Date"]
        if price_now < price_then and macd_now > macd_then:
            macd_score = abs(((macd_now - macd_then) / abs(macd_then)) * 100) * abs((price_now - price_then) / price_then) * 100
            signals.append(("MACD", "bullish_divergence", macd_score, macd_then_idx))

    if rsi_trend == "up":
        rsi_then_idx = df[-trend_bars:]["RSI"].idxmin()
        rsi_then = df.loc[rsi_then_idx]["RSI"]
        price_then = df.loc[rsi_then_idx]["Close"]
        rsi_date = df.loc[rsi_then_idx]["Date"]
        if price_now < price_then and rsi_now > rsi_then:
            rsi_score = abs(((rsi_now - rsi_then) / abs(rsi_then)) * 100) * abs((price_now - price_then) / price_then) * 100
            signals.append(("RSI", "bullish_divergence", rsi_score, rsi_then_idx))

    # Detect Bearish Divergence
    if macd_trend == "down":
        macd_then_idx = df[-trend_bars:]["MACD"].idxmax()
        macd_then = df.loc[macd_then_idx]["MACD"]
        price_then = df.loc[macd_then_idx]["Close"]
        macd_date = macd_then_idx
        if price_now > price_then and macd_now < macd_then:
            macd_score = abs(((macd_then - macd_now) / abs(macd_then)) * 100) * abs((price_now - price_then) / price_then) * 100
            signals.append(("MACD", "bearish_divergence", macd_score, macd_then_idx))

    if rsi_trend == "down":
        rsi_then_idx = df[-trend_bars:]["RSI"].idxmax()
        rsi_then = df.loc[rsi_then_idx]["RSI"]
        price_then = df.loc[rsi_then_idx]["Close"]
        rsi_date = df.loc[rsi_then_idx]["Date"]
        if price_now > price_then and rsi_now < rsi_then:
            rsi_score = abs(((rsi_then - rsi_now) / abs(rsi_then)) * 100) * abs((price_now - price_then) / price_then) * 100
            signals.append(("RSI", "bearish_divergence", rsi_score, rsi_then_idx))
            
    macd_slope_pct = ((macd_now - macd_then) / abs(macd_then)) * 100 if macd_then != 0 else 0
    price_change_pct = ((price_now - price_then) / price_then) * 100
    rsi_slope_pct = ((rsi_now - rsi_then) / rsi_then) * 100
    
    # Combine scores
    results = []
    for divergence_type in ["bullish_divergence", "bearish_divergence"]:
        related = [s for s in signals if s[1] == divergence_type]
        if related:
            base = len(related)
            bonus = 2 if base == 2 else 0
            magnitude = abs(macd_slope_pct) * abs(price_change_pct) * abs(rsi_slope_pct) / 100
            total_score = base + bonus + (magnitude * 0.1)
            
            if total_score > 10 and total_score > 6 and bonus > 0:
                results.append({
                    "signal": divergence_type,
                    "confirmed": bonus > 0,
                    "base_score": base,
                    "bonus_score": bonus,
                    "magnitude_score": round(magnitude, 2),
                    "composite_score": round(total_score, 2),
                    "strength": "strong" if total_score > 10 else "moderate" if total_score > 6 else "weak",
                    "price_now": round(price_now, 2),
                    "macd_then": round(macd_then, 4) if macd_then is not None else None,
                    "macd_now": round(macd_now, 4),
                    "rsi_then": round(rsi_then, 2) if rsi_then is not None else None,
                    "rsi_now": round(rsi_now, 2),
                    "macd_date" :macd_date,
                    "rsi_date" :rsi_date
                })

    return results


def scan_divergence_bulk(indicator="macd", lookback=30, signal_filter=None):
    with open("data/large_cap_tickers.json") as f:
        tickers = json.load(f)
     
    results = [] 
    try:
        for ticker in tickers:
            df = read_tickers_df(ticker)
            signals = detect_combined_divergence(df)
            for signal in signals:
                signal["ticker"] = ticker
                results.append(signal)
                
    except Exception as e:
            print(f"error: {e}")
    results.sort(key=lambda x: x["composite_score"], reverse=True)
    return results
           
def ma_health_stats(tickers):
    
    count_50 = 0
    count_200 = 0
    count_both = 0
    total = 0

    for ticker in tickers:
        try:
            df = read_tickers_df(ticker)
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
    

def ma_health_alert():
    tickers = load_tickers_from_json()
    stats = ma_health_stats(tickers)
    if stats["total"] == 0:
        return "Could not calculate MA health."
        
    return format_market_breadth_telegram_message(stats)
    
def trade(ticker, rr_ratio=2, side="long"):
    if not ticker:
        return "ticker is missing"
    
    df = read_tickers_df(ticker)
    
    if df.empty:
        return 'data not found'
    
    df["ATR"] = abs(df["High"] - df["Low"]).rolling(14).mean()
    
    latest = df.iloc[-1]
    close_price = latest["Close"]
    atr = latest["ATR"]
    
    sl_value = 1.5  * atr
    if side == "long":
        sl_by_atr = close_price - sl_value
        sl_by_pct = close_price * (1 - 8 / 100)
        stop_loss = max(sl_by_atr, sl_by_pct)

        take_profit = close_price + (close_price - stop_loss) * rr_ratio

        sl_pct = abs((close_price - stop_loss) / close_price) * 100
        tp_pct = abs((take_profit - close_price) / close_price) * 100

    elif side == "short":
        sl_by_atr = close_price + sl_value
        sl_by_pct = close_price * (1 + 8 / 100)
        stop_loss = min(sl_by_atr, sl_by_pct)

        take_profit = close_price - (stop_loss - close_price) * rr_ratio

        sl_pct = abs((stop_loss - close_price) / close_price) * 100
        tp_pct = abs((close_price - take_profit) / close_price) * 100
    
    msg = []
    msg.append(f"📊 Trading Plan for {ticker}\n")
    msg.append(f"Entry: ${round(close_price,2)} ({side})")
    msg.append(f"Take Profit: ${round(take_profit,2)} ({round(tp_pct,2)}%)")
    msg.append(f"Stop Loss: ${round(stop_loss,2)} ({round(sl_pct,2)}%)")
    msg.append(f"ATR: {round(atr,2)}")
    msg.append(f"Reward:Risk: {rr_ratio}")

    msg.append("\n📘 Please refer to trading view chart")
    return "\n".join(msg)