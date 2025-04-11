from flask import Blueprint, request
from .bot import handle_command, analyze_new_highs,find_golden_cross_tickers, find_death_cross_tickers,scan_divergence_bulk,ma_health_alert
from app.utils.telegram_utils import send_telegram_alert
from .signals.update_tickers import download_and_save_tickers
import logging

webhook = Blueprint('webhook', __name__)
screener = Blueprint('screener', __name__)

@webhook.route('/webhook', methods=['POST'])
def telegram_webhook():
    data = request.get_json()
    logging.info(f"response: {data}")
    if 'message' in data and 'text' in data['message']:
        message = data['message']
        text = message['text']
        chat_id = message['chat']['id']
        reply = handle_command(text)
        if reply:
            send_telegram_alert(reply, chat_id=chat_id)
    return {"ok": True}, 200

@screener.route('/new-high', methods=['GET'])
def new_high():
   return analyze_new_highs()

@screener.route('/golden-cross', methods=['GET'])
def golden_cross():
   return find_golden_cross_tickers()

@screener.route('/death-cross', methods=['GET'])
def death_cross():
   return find_death_cross_tickers()

@screener.route('/scan-divergence', methods=['GET'])
def divergence_scan():
    return scan_divergence_bulk(indicator = request.args.get("indicator", "macd"), signal_filter=request.args.get("signal"))
    
@screener.route("/ma-health", methods=["GET"])
def ma_health():
    return ma_health_alert()

@screener.route("/ma-health-alert", methods=["GET"])
def alert_ma_health():
    reply = ma_health_alert()
    send_telegram_alert(message=reply, chat_id=request.args.get("chat-id"))
    return {"status":"ok"},200

@screener.route("/update-tickers", methods=["GET"])
def update_ticker_scheduler():
    return download_and_save_tickers()