import pandas as pd
from app.config import ATR_MULTIPLIER, MAX_SL_PERCENT, DEFAULT_RR_RATIO, ATR_PERIOD

def detect_breakout_signals(df, ticker, window=10, rr_ratio=DEFAULT_RR_RATIO, max_sl_pct=MAX_SL_PERCENT, atr_period=ATR_PERIOD, latest_only=True):
    """
    Detect breakout above recent high with ATR-based SL.
    Valid only if SL is within max_sl_pct (e.g., 8%).
    """
    signals = []

    if df.empty or "Close" not in df.columns or "High" not in df.columns:
        return signals

    df["ATR"] = abs(df["High"] - df["Low"]).rolling(atr_period).mean()

    if len(df) < window + atr_period + 2:
        return signals

    for i in range(window + atr_period, len(df) - 1):
        try:
            recent_high = df.iloc[i - window:i]["High"].max()
            breakout_close = df.iloc[i]["Close"]
            breakout_date = df.index[i]

            if breakout_close > recent_high:
                atr = df["ATR"].iloc[i]
                sl_value = ATR_MULTIPLIER  * atr
                stop_loss = breakout_close - sl_value
                sl_pct = (breakout_close - stop_loss) / breakout_close * 100

                if sl_pct > max_sl_pct:
                    continue

                take_profit = breakout_close + (sl_value * rr_ratio)

                signal = {
                    "ticker": ticker,
                    "breakout_date": breakout_date.strftime("%Y-%m-%d"),
                    "entry_price": round(breakout_close, 2),
                    "stop_loss": round(stop_loss, 2),
                    "take_profit": round(take_profit, 2),
                    "rr_ratio": rr_ratio,
                    "strategy": "breakout_risk",
                    "event": "Breakout above resistance",
                    "atr": round(atr, 2)
                }

                if not latest_only or breakout_date == df.index[-1]:
                    signals.append(signal)

        except Exception as e:
            print(f"❌ Error at row {i} for {ticker}: {e}")
            break

    return signals
