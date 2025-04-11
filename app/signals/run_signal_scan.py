import pandas as pd
from app.strategy.breakout_risk import detect_breakout_signals
from app.strategy.golden_cross import detect_golden_cross
from app.strategy.death_cross import detect_death_cross
from app.strategy.golden_cross_fast import detect_golden_cross as detect_golden_cross_fast
from app.strategy.death_cross_fast import detect_death_cross as detect_death_cross_fast
from app.strategy.consolidation import detect_consolidation_and_breakout
from app.strategy.short_breakdown import detect_short_breakdown_signal
from app.signals.log_signal import save_signals_to_log
from app.utils.telegram_utils import send_telegram_alert
import traceback
import time

def run_all_strategies_on_ticker(df, ticker):
    strategies = [
        # detect_golden_cross,
        # detect_death_cross,
        # detect_golden_cross_fast,
        detect_death_cross_fast,
        # detect_breakout_signals,
        # detect_consolidation_and_breakout,
        detect_short_breakdown_signal
    ]
    all_signals = []

    for strategy in strategies:
        try:
            signals = strategy(df.copy(), ticker)
            all_signals.extend(signals)
            
            if signals:  # ✅ Save per-strategy log only if signals exist
                strategy_name = strategy.__name__.replace("detect_", "")
                save_signals_to_log(signals, strategy=strategy_name)
            
        except Exception as e:
            print(f"Error running {strategy.__name__} on {ticker}: {e}")
            traceback.print_exc()
            continue

    
    return all_signals


def scan_all_tickers(ticker_list, data_dir="data/tickers"):
    all_signals = []
    for ticker in ticker_list:
        try:
            path = f"{data_dir}/{ticker}_12mo.csv"
            df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
            
            signals = run_all_strategies_on_ticker(df, ticker)
            all_signals.extend(signals)  # Collect each ticker's signals
                
        except Exception as e:
            print(f"⚠️ Error scanning {ticker}: {e}")
            traceback.print_exc()
            continue
        
    send_signals_to_telegram(all_signals)
    return all_signals

def format_signal_message(signal):
    entry = signal["entry_price"]
    stop = signal.get("stop_loss")
    target = signal.get("take_profit")
    rr = signal.get("rr_ratio", "-")
    atr = signal.get("atr", "-")

    # Calculate SL and TP percentages
    sl_pct = f"{round(abs((entry - stop) / entry * 100), 2)}%" if stop else "-"
    tp_pct = f"{round(abs((target - entry) / entry * 100), 2)}%" if target else "-"

    return (
        f"📈 *{signal['strategy'].replace('_', ' ').title()}* on *{signal['ticker']}*\n"
        f"🗓️ {signal.get('event', 'Signal generated')} on {signal.get('signal_date', signal.get('breakout_date'))}\n"
        f"💵 Entry: ${entry}\n"
        f"⛔ Stop Loss: ${stop} ({sl_pct})\n"
        f"🎯 Take Profit: ${target} ({tp_pct})\n"
        f"📊 ATR: {atr}\n"
        f"🔁 R:R: {rr}"
    )


def send_signals_to_telegram(signals, max_signals=10, delay_seconds=5):
    for i, signal in enumerate(signals[:max_signals]):
        msg = format_signal_message(signal)  # You must define this function
        print(f"Sending {i+1}/{min(len(signals), max_signals)}: {signal['ticker']}")
        send_telegram_alert(msg, chat_id=-1002555638969)
        if i < max_signals - 1:
            time.sleep(delay_seconds)  # ⏳ wait between sends