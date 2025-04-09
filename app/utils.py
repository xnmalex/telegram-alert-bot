import requests
import os
import logging
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


