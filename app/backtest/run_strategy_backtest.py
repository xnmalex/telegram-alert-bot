from datetime import datetime, timedelta
import pandas as pd
import random
from collections import defaultdict
from app.strategy.breakout_risk import detect_breakout_signals
from app.strategy.golden_cross import detect_golden_cross
from app.strategy.death_cross import detect_death_cross
from app.strategy.golden_cross_fast import detect_golden_cross as detect_golden_cross_fast
from app.strategy.death_cross_fast import detect_death_cross as detect_death_cross_fast
from app.strategy.consolidation import detect_consolidation_and_breakout

STRATEGIES = {
    "breakout_risk": detect_breakout_signals,
    "golden_cross": detect_golden_cross,
    "death_cross": detect_death_cross,
    "golden_cross_fast": detect_golden_cross_fast,
    "death_cross_fast": detect_death_cross_fast,
    "consolidation_breakout": detect_consolidation_and_breakout,
}

def run_backtest(tickers, data_dir="data/tickers", latest_only=False):
    all_trades = []

    for ticker in tickers:
        try:
            df = pd.read_csv(f"{data_dir}/{ticker}_12mo.csv", parse_dates=["Date"], index_col="Date")

            for strategy_name, strategy_func in STRATEGIES.items():
                print(f"📈 Backtesting {strategy_name} on {ticker}...")

                # Support flexible latest_only param
                strategy_args = {
                    "df": df.copy(),
                    "ticker": ticker,
                    "latest_only": latest_only
                }

                signals = strategy_func(**strategy_args)

                for signal in signals:
                    # Simulate random outcome (real logic would check historical result)
                    simulated = simulate_trade_result(signal)
                    if simulated:
                        all_trades.append(simulated)

        except Exception as e:
            print(f"⚠️ Error in {ticker}: {e}")

    return all_trades


def simulate_trade_result(signal, max_hold_days=15, win_chance=0.6):
    try:
        entry = signal["entry_price"]
        stop = signal["stop_loss"]
        target = signal["take_profit"]
        signal_date = pd.to_datetime(signal.get("signal_date"))
        rr = signal.get("rr_ratio", 2)

        if any(pd.isna(x) for x in [entry, stop, target, signal_date]):
            return None

        risk = abs(entry - stop)
        win = random.random() < win_chance

        if win:
            pnl = round(risk * rr, 2)
            exit_price = target
        else:
            pnl = round(-risk, 2)
            exit_price = stop

        # Simulated exit date
        exit_date = signal_date + timedelta(days=random.randint(3, max_hold_days))

        # Return enriched signal
        return {
            **signal,
            "exit_date": exit_date.strftime("%Y-%m-%d"),
            "exit_price": round(exit_price, 2),
            "pnl": pnl,
            "result": "win" if pnl > 0 else "loss",
            "holding_days": (exit_date - signal_date).days
        }

    except Exception as e:
        print(f"❌ Error simulating trade: {e}, signal: {signal}")
        return None


def summarize_trades(trades):
    # 🧹 Clean up: remove trades that didn’t simulate correctly
    trades = [t for t in trades if t and "result" in t and "pnl" in t]

    if not trades:
        return {}, {}
    
    summary = {
        "total_trades": len(trades),
        "wins": sum(1 for t in trades if t["result"] == "win"),
        "losses": sum(1 for t in trades if t["result"] == "loss"),
        "avg_pnl": float(round(sum(t["pnl"] for t in trades) / len(trades), 2)) if trades else 0,
    }

    summary["win_rate"] = float(round((summary["wins"] / summary["total_trades"]) * 100, 2)) if trades else 0

    # Strategy breakdown
    per_strategy = defaultdict(list)
    for t in trades:
        per_strategy[t["strategy"]].append(t)

    strategy_stats = {}
    for strat, ts in per_strategy.items():
        wins = [t for t in ts if t["result"] == "win"]
        losses = [t for t in ts if t["result"] == "loss"]
        strategy_stats[strat] = {
            "total": len(ts),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": float(round(len(wins) / len(ts) * 100, 2)) if ts else 0,
            "avg_pnl": float(round(sum(t["pnl"] for t in ts) / len(ts), 2)) if ts else 0
        }

    return summary, strategy_stats
