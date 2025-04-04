# Telegram Alert Bot for Stock Technical Signals 📈

This Python bot monitors a list of stocks or ETFs (e.g., SPY, QQQ) and sends alerts to a Telegram group when key technical indicators are triggered, such as:
- **Golden Cross** (50-day MA crosses above 200-day MA)
- **RSI near oversold** (RSI < 35)

## 🔧 Features
- Automatic technical analysis using `yfinance`
- Sends real-time alerts to Telegram using the Telegram Bot API
- Easily extendable to support more signals or stocks

---

## 🚀 Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/xnmalex/telegram-alert-bot.git
cd telegram-alert-bot
