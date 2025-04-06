import pandas as pd
from app.strategy.breakout_risk import detect_breakout_signals
from app.strategy.golden_cross import detect_golden_cross
from app.strategy.death_cross import detect_death_cross
from app.strategy.golden_cross_fast import detect_golden_cross as detect_golden_cross_fast
from app.strategy.death_cross_fast import detect_death_cross as detect_death_cross_fast
from app.strategy.consolidation import detect_consolidation_and_breakout
from app.signals.log_signal import log_trade_signal
from app.utils import send_telegram_alert
import traceback

def run_all_strategies_on_ticker(df, ticker):
    strategies = [
        detect_golden_cross,
        detect_death_cross,
        detect_golden_cross_fast,
        detect_death_cross_fast,
        detect_breakout_signals,
        detect_consolidation_and_breakout
    ]
    all_signals = []

    for strategy in strategies:
        try:
            signals = strategy(df.copy(), ticker)
            all_signals.extend(signals)
        except Exception as e:
            print(f"Error running {strategy.__name__} on {ticker}: {e}")

    return all_signals


def scan_all_tickers(ticker_list, data_dir="data/tickers"):
    for ticker in ticker_list[:10]:
        try:
            path = f"{data_dir}/{ticker}_12mo.csv"
            df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
            signals = run_all_strategies_on_ticker(df, ticker)

            for signal in signals:
                log_trade_signal(signal)
                message = format_signal_message(signal)
                print(message)
                #send_telegram_alert(message, chat_id=849242284)
                
        except Exception as e:
            print(f"⚠️ Error scanning {ticker}: {e}")
            traceback.print_exc()
            break

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


