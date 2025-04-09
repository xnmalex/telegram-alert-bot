from datetime import timedelta
from app.config import ATR_MULTIPLIER, MAX_SL_PERCENT, DEFAULT_RR_RATIO, ATR_PERIOD
import pandas as pd

def detect_death_cross(df, ticker, latest_only=True, max_signal_age_days=7, rr_ratio=DEFAULT_RR_RATIO, max_sl_pct=MAX_SL_PERCENT, atr_period=ATR_PERIOD):
    signals = []
    df["MA21"] = df["Close"].rolling(21).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["ATR"] = abs(df["High"] - df["Low"]).rolling(atr_period).mean()

    if len(df) < 51:
        return signals

    last_date = df.index[-1]
    cutoff_date = last_date - timedelta(days=max_signal_age_days)
    latest_price = df["Close"].iloc[-1]

    for i in range(50, len(df)):
        signal_date = df.index[i]

        if df["MA21"].iloc[i - 1] > df["MA50"].iloc[i - 1] and df["MA21"].iloc[i] < df["MA50"].iloc[i]:
            if not latest_only or signal_date >= cutoff_date:
                entry_price = df["Close"].iloc[i]
                atr = df["ATR"].iloc[i]
                sl_value = ATR_MULTIPLIER * atr
                stop_loss = entry_price + sl_value
                sl_pct = (stop_loss - entry_price) / entry_price * 100

                if sl_pct > max_sl_pct:
                    continue

                take_profit = entry_price - sl_value * rr_ratio
                distance_from_signal = abs((latest_price - entry_price) / entry_price)

                signals.append({
                    "ticker": ticker,
                    "signal_date": signal_date.strftime("%Y-%m-%d"),
                    "entry_price": round(entry_price, 2),
                    "stop_loss": round(stop_loss, 2),
                    "take_profit": round(take_profit, 2),
                    "rr_ratio": rr_ratio,
                    "strategy": "death_cross_21_50",
                    "event": "21MA crossed below 50MA",
                    "atr":round(atr, 2),
                    "distance_from_signal": round(distance_from_signal * 100, 2)  # percentage
                })

    return signals
