import os
import json
import pandas as pd
from datetime import datetime

DATA_FOLDER = "data/tickers/"
CACHE_FILE = os.path.join("data", "large_cap_tickers.json")
INITIAL_PORTFOLIO = 100_000
RISK_PER_TRADE_PCT = 1
MA_HEALTH_STATUS = "Moderate"

# === BACKTEST STRATEGY ===

def portfolio_backtest(df, ticker, initial_portfolio, risk_per_trade_pct, ma_health_status):
    trades = []
    portfolio_size = initial_portfolio
    position = None

    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["EMA12"] = df["Close"].ewm(span=12).mean()
    df["EMA26"] = df["Close"].ewm(span=26).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["EMA21"] = df["Close"].ewm(span=21).mean()

    for i in range(30, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 5]

        price_now = row["Close"]
        price_then = prev["Close"]
        macd_now = row["MACD"]
        macd_then = prev["MACD"]

        if pd.isna(row["SMA50"]) or pd.isna(row["SMA200"]) or pd.isna(macd_now) or pd.isna(macd_then):
            continue

        macd_slope = (macd_now - macd_then) / abs(macd_then) * 100 if macd_then != 0 else 0

        # Entry
        if (
            not position
            and price_now < price_then
            and macd_now > macd_then
            and price_now > row["SMA50"]
            and price_now > row["SMA200"]
            and macd_slope > 100
        ):
            entry_price = price_now
            stop_loss = entry_price * 0.92
            target_price = entry_price * 1.16
            risk_per_trade = portfolio_size * (risk_per_trade_pct / 100)
            risk_per_share = entry_price - stop_loss

            if risk_per_share <= 0:
                continue

            shares = int(risk_per_trade / risk_per_share)
            position_size = shares * entry_price

            if shares < 1 or position_size > portfolio_size:
                continue

            position = {
                "entry_price": entry_price,
                "entry_date": row.name,
                "stop_loss": stop_loss,
                "target_price": target_price,
                "shares": shares,
                "capital_used": position_size,
                "target_reached": False,
            }

            portfolio_size -= position_size

        elif position:
            exit_price = price_now
            pnl = (exit_price - position["entry_price"]) * position["shares"]
            below_ema21 = exit_price < row["EMA21"]

            exit = False
            exit_reason = ""

            if exit_price <= position["stop_loss"]:
                exit = True
                exit_reason = "Stop loss"

            elif not position["target_reached"] and exit_price >= position["target_price"]:
                position["target_reached"] = True

            elif position["target_reached"] and below_ema21:
                ema_gap = (row["EMA21"] - position["entry_price"]) / position["entry_price"] * 100
                if ema_gap >= -8:
                    exit = True
                    exit_reason = "Broke EMA21 after TP"

            if exit:
                portfolio_size += position["capital_used"] + pnl
                trades.append({
                    "ticker": ticker,
                    "entry_date": position["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": row.name.strftime("%Y-%m-%d"),
                    "entry_price": round(position["entry_price"], 2),
                    "exit_price": round(exit_price, 2),
                    "shares": position["shares"],
                    "pnl": round(pnl, 2),
                    "pct_gain": round((exit_price - position["entry_price"]) / position["entry_price"] * 100, 2),
                    "exit_reason": exit_reason,
                    "capital_used": round(position["capital_used"], 2),
                })
                position = None

    return trades, round(portfolio_size, 2)

# === RUN MULTIPLE TICKERS ===
def load_csv_data(ticker):
    path = os.path.join(DATA_FOLDER, f"{ticker}_12mo.csv")
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        return None
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
    return df

def run_multi_backtest():
    all_trades = []
    portfolio = INITIAL_PORTFOLIO
    
    with open("data/large_cap_tickers.json") as f:
            ticker_list = json.load(f)

    for ticker in ticker_list:
        df = load_csv_data(ticker)
        if df is None:
            continue

        trades, portfolio = portfolio_backtest(df, ticker, portfolio, RISK_PER_TRADE_PCT, MA_HEALTH_STATUS)
        all_trades.extend(trades)

    print("\nFinal Portfolio Value:", portfolio)
    print("\nClosed Trades:")
    for t in all_trades:
        print(t)

    # Optional: save to CSV
    # if all_trades:
    #     pd.DataFrame(all_trades).to_csv("trade_results.csv", index=False)
    #     print("\nSaved to trade_results.csv")
        
    # save to JSON
    if all_trades:
        with open("trade_results.json", "w") as f:
            json.dump(all_trades, f, indent=2, default=str)

        print("\nSaved to trade_results.json")


def summarize_trade_results(json_path="trade_results.json"):
    try:
        with open(json_path, "r") as f:
            trades = json.load(f)

        if not trades:
            print("No trades to summarize.")
            return

        df = pd.DataFrame(trades)
 
        # === Per-Ticker Summary ===
        print("=== Per Ticker Summary ===")
        grouped = df.groupby("ticker")

        for ticker, group in grouped:
            total = len(group)
            wins = group[group["pnl"] > 0]
            win_rate = len(wins) / total * 100
            avg_pnl = group["pnl"].mean()
            avg_pct = group["pct_gain"].mean()
            total_profit = group["pnl"].sum()

            print(f"\nTicker: {ticker}")
            print(f"  Trades       : {total}")
            print(f"  Win rate     : {win_rate:.2f}%")
            print(f"  Avg PnL      : ${avg_pnl:.2f}")
            print(f"  Avg % gain   : {avg_pct:.2f}%")
            print(f"  Total profit : ${total_profit:.2f}")
            
        total_trades = len(df)
        wins = df[df["pnl"] > 0]
        losses = df[df["pnl"] <= 0]
        win_rate = len(wins) / total_trades * 100
        avg_pnl = df["pnl"].mean()
        avg_pct_gain = df["pct_gain"].mean()
        net_profit = df["pnl"].sum()

        print("\n\n=== Trade Summary ===")
        print(f"Total trades   : {total_trades}")
        print(f"Win rate       : {win_rate:.2f}%")
        print(f"Average PnL    : ${avg_pnl:.2f}")
        print(f"Average % gain : {avg_pct_gain:.2f}%")
        print(f"Net profit     : ${net_profit:.2f}")

    except Exception as e:
        print(f"Error summarizing results: {e}")
        
        
def backtest_cross_strategy(df, ticker, initial_portfolio=100_000, risk_per_trade_pct=1.0):
    trades = []
    portfolio_size = initial_portfolio
    position = None
    position_type = None  # 'long' or 'short'

    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["EMA20"] = df["Close"].ewm(span=20).mean()

    for i in range(51, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        price_now = row["Close"]
        sma50_now = row["SMA50"]
        sma200_now = row["SMA200"]
        ema20_now = row["EMA20"]

        sma50_prev = prev_row["SMA50"]
        sma200_prev = prev_row["SMA200"]

        if pd.isna(sma50_now) or pd.isna(sma200_now) or pd.isna(ema20_now):
            continue

        # Entry logic
        if not position:
            # Golden cross (Long)
            if sma50_prev <= sma200_prev and sma50_now > sma200_now:
                entry_price = price_now
                stop_loss = entry_price * 0.92
                target_price = entry_price * 1.08
                risk_per_trade = portfolio_size * (risk_per_trade_pct / 100)
                risk_per_share = entry_price - stop_loss
                if risk_per_share <= 0:
                    continue
                shares = int(risk_per_trade / risk_per_share)
                position_size = shares * entry_price
                if shares < 1 or position_size > portfolio_size:
                    continue

                position = {
                    "entry_price": entry_price,
                    "entry_date": row.name,
                    "stop_loss": stop_loss,
                    "target_price": target_price,
                    "shares": shares,
                    "capital_used": position_size,
                    "ema_exit_level": ema20_now
                }
                position_type = "long"
                portfolio_size -= position_size

            # Death cross (Short)
            elif sma50_prev >= sma200_prev and sma50_now < sma200_now:
                entry_price = price_now
                stop_loss = entry_price * 1.08
                target_price = entry_price * 0.92
                risk_per_trade = portfolio_size * (risk_per_trade_pct / 100)
                risk_per_share = stop_loss - entry_price
                if risk_per_share <= 0:
                    continue
                shares = int(risk_per_trade / risk_per_share)
                position_size = shares * entry_price
                if shares < 1 or position_size > portfolio_size:
                    continue

                position = {
                    "entry_price": entry_price,
                    "entry_date": row.name,
                    "stop_loss": stop_loss,
                    "target_price": target_price,
                    "shares": shares,
                    "capital_used": position_size,
                    "ema_exit_level": ema20_now
                }
                position_type = "short"
                portfolio_size -= position_size

        # Exit logic
        elif position:
            exit = False
            exit_reason = ""
            exit_price = price_now
            shares = position["shares"]

            if position_type == "long":
                pnl = (exit_price - position["entry_price"]) * shares
                gain_pct = (exit_price - position["entry_price"]) / position["entry_price"] * 100

                if exit_price <= position["stop_loss"]:
                    exit = True
                    exit_reason = "Stop loss"
                elif exit_price >= position["target_price"]:
                    exit = True
                    exit_reason = "Target hit"
                elif exit_price < row["EMA20"]:
                    exit = True
                    exit_reason = "Broke EMA20"

            elif position_type == "short":
                pnl = (position["entry_price"] - exit_price) * shares
                gain_pct = (position["entry_price"] - exit_price) / position["entry_price"] * 100

                if exit_price >= position["stop_loss"]:
                    exit = True
                    exit_reason = "Stop loss"
                elif exit_price <= position["target_price"]:
                    exit = True
                    exit_reason = "Target hit"
                elif exit_price > row["EMA20"]:
                    exit = True
                    exit_reason = "Broke EMA20"

            if exit:
                portfolio_size += position["capital_used"] + pnl
                trades.append({
                    "ticker": ticker,
                    "entry_date": position["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": row.name.strftime("%Y-%m-%d"),
                    "entry_price": round(position["entry_price"], 2),
                    "exit_price": round(exit_price, 2),
                    "shares": shares,
                    "pnl": round(pnl, 2),
                    "pct_gain": round(gain_pct, 2),
                    "exit_reason": exit_reason,
                    "capital_used": round(position["capital_used"], 2),
                    "position_type": position_type,
                    "portfolio_after_trade": round(portfolio_size, 2),
                    "strategy": "golden_cross" if position_type == "long" else "death_cross"
                })
                position = None
                position_type = None

    return trades, round(portfolio_size, 2)


def run_multi_cross_backtest():
    all_cross_trades = []
    portfolio = INITIAL_PORTFOLIO
    
    with open("data/large_cap_tickers.json") as f:
            ticker_list = json.load(f)

    for ticker in ticker_list:
        df = load_csv_data(ticker)
        if df is None:
            continue

        print(f"▶️ Running cross strategy for {ticker}...")
        trades, portfolio = backtest_cross_strategy(
            df.copy(),
            ticker=ticker,
            initial_portfolio=portfolio,
            risk_per_trade_pct=RISK_PER_TRADE_PCT
        )
        all_cross_trades.extend(trades)

    # Save results
    if all_cross_trades:
        with open("cross_strategy_results.json", "w") as f:
            json.dump(all_cross_trades, f, indent=2, default=str)
        print("✅ Saved cross trades to cross_strategy_results.json")

        # Optionally summarize
        #summarize_cross_trades("cross_strategy_results.json")
    else:
        print("⚠️ No cross trades triggered across tickers.")
        
def backtest_breakout_with_risk_manager(
    df,
    ticker,
    risk_manager,
    position_pct=25,
    stop_loss_pct=8,
    rr_ratio=2.0
):
    trades = []
    previous_high_window = 10

    for i in range(previous_high_window, len(df)):
        row = df.iloc[i]
        current_price = row["Close"]
        date = row.name

        recent_high = df.iloc[i - previous_high_window:i]["High"].max()

        if current_price > recent_high:
            if not risk_manager.can_take_trade(position_pct, stop_loss_pct):
                continue

            position_size = risk_manager.portfolio_value * (position_pct / 100)
            stop_loss = current_price * (1 - stop_loss_pct / 100)
            risk_amount = current_price - stop_loss
            take_profit = current_price + (risk_amount * rr_ratio)

            next_row = df.iloc[i + 1] if i + 1 < len(df) else row
            outcome_price = next_row["Close"]
            result = {}

            if outcome_price <= stop_loss:
                result = risk_manager.take_trade(position_pct, stop_loss_pct, hit_stop=True)
                trades.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "entry": current_price,
                    "exit": outcome_price,
                    "pnl": -risk_amount * position_size / current_price,
                    "result": "loss",
                    "portfolio": result["portfolio_value"]
                })
            elif outcome_price >= take_profit:
                profit = risk_amount * rr_ratio * position_size / current_price
                risk_manager.take_trade(position_pct, stop_loss_pct, hit_stop=False)
                trades.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "entry": current_price,
                    "exit": outcome_price,
                    "pnl": profit,
                    "result": "win",
                    "portfolio": risk_manager.portfolio_value
                })

    return trades
