from flask import Blueprint, request
from .bot import handle_command
from .utils import send_telegram_alert

webhook = Blueprint('webhook', __name__)

@webhook.route('/webhook', methods=['POST'])
def telegram_webhook():
    data = request.get_json()
    if 'message' in data and 'text' in data['message']:
        message = data['message']
        text = message['text']
        chat_id = message['chat']['id']
        reply = handle_command(text)
        if reply:
            send_telegram_alert(reply, chat_id=chat_id)
    return {"ok": True}, 200
