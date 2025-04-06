import pandas as pd
from app.config import ATR_MULTIPLIER, MAX_SL_PERCENT, DEFAULT_RR_RATIO, ATR_PERIOD

def detect_consolidation_and_breakout(df, ticker, window=10, tolerance=0.03, rr_ratios=[DEFAULT_RR_RATIO], max_sl_pct=MAX_SL_PERCENT, atr_period=ATR_PERIOD, latest_only=True):
    """
    Detects tight consolidation followed by breakout.
    Uses ATR-based SL and filters if SL > max_sl_pct.
    """
    trade_ideas = []

    if len(df) < window + atr_period + 2:
        return trade_ideas

    df["ATR"] = abs(df["High"] - df["Low"]).rolling(atr_period).mean()
    recent_data = df[-(window + 1):-1]
    breakout_day = df.iloc[-1]
    breakout_date = breakout_day.name

    high = recent_data["High"].max()
    low = recent_data["Low"].min()
    range_pct = (high - low) / low

    if range_pct <= tolerance and breakout_day["Close"] > high:
        entry_price = breakout_day["Close"]
        atr = df["ATR"].iloc[-1]
        sl_value = ATR_MULTIPLIER * atr
        stop_loss = entry_price - sl_value
        sl_pct = (entry_price - stop_loss) / entry_price * 100

        if sl_pct > max_sl_pct:
            return []

        for rr in rr_ratios:
            take_profit = entry_price + sl_value * rr
            signal = {
                "ticker": ticker,
                "breakout_date": breakout_date.strftime("%Y-%m-%d"),
                "entry_price": round(entry_price, 2),
                "stop_loss": round(stop_loss, 2),
                "take_profit": round(take_profit, 2),
                "rr_ratio": rr,
                "strategy": "consolidation_breakout",
                "event": "Breakout from consolidation",
                "atr": round(atr, 2)
            }

            if not latest_only or breakout_date == df.index[-1]:
                trade_ideas.append(signal)

    return trade_ideas
