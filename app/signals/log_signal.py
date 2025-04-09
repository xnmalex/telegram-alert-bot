import os
import json
from datetime import datetime

def save_signals_to_log(signals, strategy, log_dir="logs"):
    from datetime import datetime
    import os, json

    today = datetime.now().strftime("%Y-%m-%d")
    folder = os.path.join(log_dir, strategy)
    os.makedirs(folder, exist_ok=True)

    path = os.path.join(folder, f"{today}.json")

    # Load existing data
    if os.path.exists(path):
        with open(path, "r") as f:
            existing = json.load(f)
    else:
        existing = []

    existing.extend(signals)

    with open(path, "w") as f:
        json.dump(existing, f, indent=2, default=str)


    # print(f"✅ Logged {len(signals)} signals to {file_path}")
