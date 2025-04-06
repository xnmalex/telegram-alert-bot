import json

def load_tickers_from_json(path="data/large_cap_tickers.json"):
    """
    Load tickers from a JSON file.
    Format: ["AAPL", "MSFT", "GOOGL", ...]
    """
    try:
        with open(path, "r") as f:
            tickers = json.load(f)
        return tickers
    except Exception as e:
        print(f"Error loading tickers from {path}: {e}")
        return []
