# test_bot.py

import unittest
from app.bot import handle_command

def fake_summary(ticker):
    return f"Summary for {ticker}"

def fake_finviz_summary():
    return "✅ *AAPL* – $108.56 (Apple Inc)\n✅ *MSFT* – $340.00 (Microsoft Corp)"

class TestHandleCommand(unittest.TestCase):
    def setUp(self):
        # Patch the summary functions for testing
        self.original_summary = handle_command.__globals__.get("summarize_ticker")
        self.original_new_highs = handle_command.__globals__.get("format_finviz_new_highs")

        handle_command.__globals__["summarize_ticker"] = fake_summary
        handle_command.__globals__["format_finviz_new_highs"] = fake_finviz_summary

    def tearDown(self):
        # Restore original functions
        handle_command.__globals__["summarize_ticker"] = self.original_summary
        handle_command.__globals__["format_finviz_new_highs"] = self.original_new_highs

    def test_summary_command(self):
        result = handle_command("/summary AAPL")
        print(result)
        self.assertTrue(result.startswith("\nSummary for"))

    def test_newhighs_command(self):
        result = handle_command("/newhigh")
        self.assertTrue(len(result)>0)

if __name__ == '__main__':
    unittest.main()
