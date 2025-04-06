import json
import os

SIGNAL_LOG_PATH = "logs/trade_signals.json"

def log_trade_signal(signal):
    """
    Append new signal to log file. Handles corrupted or empty file.
    """
    if not signal:
        return

    # Create logs folder if not exists
    os.makedirs(os.path.dirname(SIGNAL_LOG_PATH), exist_ok=True)

    existing = []

    if os.path.exists(SIGNAL_LOG_PATH):
        try:
            with open(SIGNAL_LOG_PATH, "r") as f:
                content = f.read().strip()
                if content:
                    existing = json.loads(content)
        except Exception as e:
            print(f"⚠️ Could not load existing signals, resetting file: {e}")
            existing = []

    existing.append(signal)

    with open(SIGNAL_LOG_PATH, "w") as f:
        json.dump(existing, f, indent=2)
