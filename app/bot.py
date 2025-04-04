import yfinance as yf
import requests
import datetime
import os
from dotenv import load_dotenv
import time

# Load environment variables from .env
load_dotenv()

# CONFIGURATION
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
WATCHLIST = ['SPY', 'QQQ']

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
    if len(parts) == 2 and parts[0].lower() == '/summary':
        ticker = parts[1].upper()
        return display_stock_summary(ticker)
    return None