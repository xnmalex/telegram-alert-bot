import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from app.utils.tickers_utils import load_tickers_from_json, read_tickers_df

def get_pcr_and_short_float_from_barchart(ticker):
    url = f"https://www.barchart.com/stocks/quotes/{ticker}/overview"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        stat_section = soup.find("div", class_="symbol-overview-statistics__section")
        if not stat_section:
            return None, None

        pcr = None
        short_float = None

        for item in stat_section.find_all("li"):
            label = item.find("span", class_="symbol-overview-statistics__label")
            value = item.find("span", class_="symbol-overview-statistics__value")

            if label and value:
                text = label.text.strip()
                val = value.text.strip().replace("%", "").replace(",", "")

                if "Put/Call Ratio" in text:
                    pcr = float(val)
                elif "Short Interest (% of Float)" in text:
                    short_float = float(val)

        return pcr, short_float

    except Exception as e:
        print(f"⚠️ Error fetching Barchart data for {ticker}: {e}")
        return None, None

def calculate_up_down_volume_ratio(df, window=25):
    # Define up/down days
    up_days =  df["Close"] > df["Open"]
    down_days = df["Close"] < df["Open"]

    # Sum volume on up/down days
    up_volume = df.loc[up_days, 'Volume'].tail(window).sum()
    down_volume = df.loc[down_days, 'Volume'].tail(window).sum()

    if down_volume == 0:
        return float("inf")  # Avoid division by zero

    ratio = up_volume / down_volume
    return round(ratio, 2)

def get_volume_profile_poc(df, bins=25):
    """
    Calculate volume profile and return the Point of Control (POC).
    """
    df = df.copy()
    
    # Make sure required columns are numeric
    for col in ["High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing data
    df.dropna(subset=["High", "Low", "Close", "Volume"], inplace=True)

    if df.empty:
        raise ValueError("DataFrame is empty after dropping NaNs.")

    # Use typical price (High + Low + Close) / 3
    df["typical_price"] = (df["High"] + df["Low"] + df["Close"]) / 3

    # Histogram volume by price
    hist, bin_edges = np.histogram(
        df["typical_price"],
        bins=bins,
        weights=df["Volume"]
    )

    # Midpoints of bins
    price_levels = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Get POC: price level with max volume
    max_idx = np.argmax(hist)
    poc_price = round(price_levels[max_idx], 2)
    poc_volume = int(hist[max_idx])

    return {
        "poc_price": poc_price,
        "poc_volume": poc_volume,
        "volume_profile": list(zip(price_levels, hist))
    }


def find_strong_short_signals(ticker_list = None):
    matching_tickers = []
    if not ticker_list:
        ticker_list = load_tickers_from_json("data/large_cap_tickers.json")
        
    for ticker in ticker_list:
        try:
            df = read_tickers_df(ticker)
            pcr = 0,
            short_float = 0,
            #pcr, short_float = get_pcr_and_short_float_from_barchart(ticker)
            up_down_ratio = calculate_up_down_volume_ratio(df)
            bins = min(30, max(10, int(len(df) / 5)))
            result = get_volume_profile_poc(df, bins)
            poc = result['poc_price']
            
            current_price = df["Close"].iloc[-1]
            df["SMA5"] = df["Close"].rolling(window=5).mean()
            df["SMA10"] = df["Close"].rolling(window=10).mean()
            df["SMA20"] = df["Close"].rolling(window=20).mean()
            df["SMA50"] = df["Close"].rolling(window=50).mean()
            
            # for future development
            df["EMA9"] = df["Close"].ewm(span=9).mean()
        
            latest = df.iloc[-1]
            sma10_slope = latest["SMA10"] - df["SMA10"].iloc[-6]
            sma20_slope = latest["SMA20"] - df["SMA20"].iloc[-6]
            sma50_slope = latest["SMA50"] - df["SMA50"].iloc[-6]
            
            all_ma_sloping_down = sma10_slope < 0 and sma20_slope < 0 and sma50_slope < 0
            below_all_ma = current_price < latest["SMA10"] and current_price < latest["SMA20"] and current_price < latest["SMA50"] 
            is_ma_stacking = latest["SMA50"] > latest["SMA20"] and latest["SMA20"] > latest["SMA10"]
            
            distance_pct = abs((latest['SMA20'] - current_price) / latest['SMA20']) * 100
            normalized_slope = sma20_slope / df["SMA20"].iloc[-6] * 100 
    
            # if below_all_ma and up_down_ratio <=0.7 and current_price < poc  and all_ma_sloping_down and distance_pct <=5:
            if latest["SMA20"] > latest["Close"] and normalized_slope  < -2.5 and distance_pct <=5 and current_price < poc :
                matching_tickers.append({
                        "Ticker": ticker,
                        "Price": float(round(current_price,2)),
                        "5, 10, 20 MA": f"{float(round(latest['SMA5'],2))}, {float(round(latest['SMA10'],2))}, {float(round(latest['SMA20'],2))}",
                        "distance_pct":float(round(distance_pct,2)),
                        "Up/Down Volume Ratio": float(round(up_down_ratio,2)),
                        "Normalised Slope": float(round(normalized_slope,2)),
                        "POC":f"{poc} - {round(poc * 1.05,2)}"
                })
                
            if ticker not in matching_tickers and  below_all_ma and up_down_ratio <=0.7 and current_price < poc  and all_ma_sloping_down and is_ma_stacking :
                matching_tickers.append({
                        "Ticker": ticker,
                        "Price": float(round(current_price,2)),
                        "5, 10, 20 MA": f"{float(round(latest['SMA5'],2))}, {float(round(latest['SMA10'],2))}, {float(round(latest['SMA20'],2))}",
                        "distance_pct":float(round(distance_pct,2)),
                        "Up/Down Volume Ratio": float(round(up_down_ratio,2)),
                        "Normalised Slope": float(round(normalized_slope,2)),
                        "POC":f"{poc} - {round(poc * 1.05,2)}",
                        "bonus":"below all ma"
                })


        except Exception as e:
            print(f"⚠️ Error processing {ticker}: {e}")

    
    return matching_tickers

def find_strong_long_signals(ticker_list=None):
    matching_tickers = []
    
    if not ticker_list:
        ticker_list = load_tickers_from_json("data/large_cap_tickers.json")

    for ticker in ticker_list:
        try:
            df = read_tickers_df(ticker)
            pcr = 0,
            short_float = 0,
            #pcr, short_float = get_pcr_and_short_float_from_barchart(ticker)
            up_down_ratio = calculate_up_down_volume_ratio(df)
            bins = min(30, max(10, int(len(df) / 5)))
            result = get_volume_profile_poc(df, bins)
            poc = result['poc_price']
            
            current_price = df["Close"].iloc[-1]
            df["SMA5"] = df["Close"].rolling(window=5).mean()
            df["SMA10"] = df["Close"].rolling(window=10).mean()
            df["SMA20"] = df["Close"].rolling(window=20).mean()
            df["SMA50"] = df["Close"].rolling(window=50).mean()
            df["EMA9"] = df["Close"].ewm(span=9).mean()
        
            latest = df.iloc[-1]
            sma10_slope = latest["SMA10"] - df["SMA10"].iloc[-6]
            sma20_slope = latest["SMA20"] - df["SMA20"].iloc[-6]
            sma50_slope = latest["SMA50"] - df["SMA50"].iloc[-6]
            
            distance_pct = abs(( current_price - latest['SMA20']) / current_price) * 100
            
            all_ma_curving_up = sma10_slope > 0 and sma20_slope > 0 and sma50_slope > 0
            above_all_ma = current_price > latest["SMA10"] and current_price > latest["SMA20"] and current_price > latest["SMA50"] 
            is_ma_stacking = latest["SMA10"] < latest["SMA20"] and latest["SMA20"] < latest["SMA50"]
            
            if above_all_ma and up_down_ratio >= 1.2 and current_price >= poc and all_ma_curving_up and distance_pct <= 5:
                matching_tickers.append({
                        "Ticker": ticker,
                        "Price": float(round(current_price,2)),
                        "MA": f"{float(round(latest['SMA10'],2))}, {float(round(latest['SMA20'],2))}, {float(round(latest['SMA50'],2))}",
                        "distance_pct":float(round(distance_pct , 2)),
                        "Up/Down Volume Ratio": float(round(up_down_ratio,2)),
                        "SMA 20 slope": float(round(sma20_slope,2)),
                        "POC":f"{poc} - {float(round(poc * 1.05, 2))}"
                })
    
            if ticker not in matching_tickers and above_all_ma and up_down_ratio > 1.2 and current_price >= poc  and all_ma_curving_up  and  is_ma_stacking :
                matching_tickers.append({
                        "Ticker": ticker,
                        "Price": float(round(current_price,2)),
                        "MA": f"{float(round(latest['SMA10'],2))}, {float(round(latest['SMA20'],2))}, {float(round(latest['SMA50'],2))}",
                        "distance_pct":float(round(distance_pct , 2)),
                        "Up/Down Volume Ratio": float(round(up_down_ratio,2)),
                        "SMA 20 slope": float(round(sma20_slope,2)),
                        "POC":f"{poc} - {float(round(poc * 1.05, 2))}",
                        "bonus":"above all ma"
                })

        except Exception as e:
            print(f"⚠️ Error processing {ticker}: {e}")

   
    return matching_tickers
