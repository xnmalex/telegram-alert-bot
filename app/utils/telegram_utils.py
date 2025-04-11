import requests
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEFAULT_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_alert(message, chat_id=None):
    target_chat = chat_id if chat_id else DEFAULT_CHAT_ID
    logging.info(f"chatid: {target_chat}")
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": target_chat,
        "text": message,
        "parse_mode": "Markdown"
    }
    requests.post(url, data=payload)
    
def get_strength_emoji(value, ma_type="default"):
    """
    Returns strength label based on MA type:
    - For 50MA and 'both': <40 Weak, 40–60 Moderate, >60 Strong
    - For 200MA: <50 Weak, 50–70 Moderate, >70 Strong
    """
    if ma_type == "200ma":
        if value < 50:
            return "🟥 Weak"
        elif value < 70:
            return "🟧 Moderate"
        else:
            return "🟩 Strong"
    else:
        if value < 40:
            return "🟥 Weak"
        elif value < 60:
            return "🟧 Moderate"
        else:
            return "🟩 Strong"
  
def format_market_breadth_telegram_message(stats):
    """
    Formats the market breadth message for Telegram.
    Uses custom thresholds per MA type, but shows a clean, unified legend.
    
    stats: dict with 'above_50ma', 'above_200ma', 'above_both', 'date' (optional)
    """
    
    # Get today's date in YYYY-MM-DD format
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    msg = []
    msg.append(f"📊 Market Breadth Overview (as of {today_str})\n")

    msg.append(f"% Above 50MA   : {stats['above_50ma']}%  → {get_strength_emoji(stats['above_50ma'], '50ma')}")
    msg.append(f"% Above 200MA  : {stats['above_200ma']}%  → {get_strength_emoji(stats['above_200ma'], '200ma')}")
    msg.append(f"% Above Both   : {stats['above_both']}%  → {get_strength_emoji(stats['above_both'], 'both')}")

    msg.append("\n📘 Legend:")
    msg.append("🟩 Strong")
    msg.append("🟧 Moderate")
    msg.append("🟥 Weak")

    return "\n".join(msg)