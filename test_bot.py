# test_bot.py

from app.bot import generate_summary_message

if __name__ == '__main__':
    print("🔍 Telegram Bot Function Tester")
    while True:
        user_input = input("Enter a command (e.g., /summary AAPL or 'q' to quit): ").strip()
        if user_input.lower() == 'q':
            break
        response = generate_summary_message(user_input.split()[1]) if user_input.lower().startswith('/summary') else "Unknown command."
        print("\n📬 Response:\n", response, "\n")
