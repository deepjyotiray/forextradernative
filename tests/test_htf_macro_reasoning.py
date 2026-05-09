import unittest

from engine.htf_bias_engine import evaluate as evaluate_long
from engine.htf_short_bias_engine import evaluate as evaluate_short
from engine.strategies.htf_long_strategy import HTFLongStrategy
from engine.strategies.htf_short_strategy import HTFShortStrategy


def _base_engine_data():
    return {
        "price": 4700.0,
        "ohlc": {},
        "weekly_range": {"high_7d": 4800.0, "low_7d": 4600.0, "mid_7d": 4700.0},
        "structure": {"D1": "UP", "H4": "UP", "H1": "UP"},
        "momentum": {"D1_body_ratio": 0.7, "H4_body_ratio": 0.7},
        "news": {"high_impact_soon": False},
        "pullback": {"depth_pct": 0.20},
        "account": {"balance": 10000.0},
    }


def _base_strategy_data():
    return {
        "tick": {"ask": 4700.0, "bid": 4699.8, "spread": 0.2},
        "account": {"balance": 10000.0},
        "market_state": {
            "structure": {"D1": "UP", "H4": "UP", "H1": "UP"},
            "body_ratio": {"D1": 0.7, "H4": 0.7},
            "weekly_range": {"high_7d": 4800.0, "low_7d": 4600.0, "mid_7d": 4700.0},
            "pullback_depth": 0.20,
            "ohlc": {},
        },
        "calendar": {"blocked": False},
    }


class HTFMacroReasoningTests(unittest.TestCase):
    def test_long_engine_calls_out_neutral_macro_instead_of_bearish(self):
        data = _base_engine_data()
        data["macro"] = {"dxy_trend": "NEUTRAL", "us10y_trend": "NEUTRAL"}

        result = evaluate_long(data)

        self.assertEqual(result["decision"], "NO_TRADE")
        self.assertIn("Macro not supportive for long", result["reason"])
        self.assertIn("(neutral:", result["reason"])
        self.assertNotIn("strongly bearish", result["reason"].lower())

    def test_short_engine_calls_out_neutral_macro_instead_of_bullish(self):
        data = _base_engine_data()
        data["structure"] = {"D1": "DOWN", "H4": "DOWN", "H1": "DOWN"}
        data["macro"] = {"dxy_trend": "NEUTRAL", "us10y_trend": "NEUTRAL"}

        result = evaluate_short(data)

        self.assertEqual(result["decision"], "NO_TRADE")
        self.assertIn("Macro not supportive for short", result["reason"])
        self.assertIn("(neutral:", result["reason"])
        self.assertNotIn("strongly bullish", result["reason"].lower())

    def test_long_strategy_no_trade_reason_includes_macro_inputs(self):
        data = _base_strategy_data()
        data["correlation"] = {
            "dxy": {"trend": "NEUTRAL"},
            "us10y": {"trend": "NEUTRAL"},
        }

        signal = HTFLongStrategy().generate_signal(data)

        self.assertEqual(signal["signal"], "NO_TRADE")
        self.assertIn("DXY=NEUTRAL", signal["reason"])
        self.assertIn("US10Y=NEUTRAL", signal["reason"])

    def test_short_strategy_no_trade_reason_includes_macro_inputs(self):
        data = _base_strategy_data()
        data["tick"] = {"ask": 4700.0, "bid": 4700.0, "spread": 0.2}
        data["market_state"]["structure"] = {"D1": "DOWN", "H4": "DOWN", "H1": "DOWN"}
        data["correlation"] = {
            "dxy": {"trend": "NEUTRAL"},
            "us10y": {"trend": "NEUTRAL"},
        }

        signal = HTFShortStrategy().generate_signal(data)

        self.assertEqual(signal["signal"], "NO_TRADE")
        self.assertIn("DXY=NEUTRAL", signal["reason"])
        self.assertIn("US10Y=NEUTRAL", signal["reason"])

    def test_mixed_macro_is_called_mixed_for_blocked_longs(self):
        data = _base_engine_data()
        data["macro"] = {"dxy_trend": "UP", "us10y_trend": "NEUTRAL"}

        result = evaluate_long(data)

        self.assertEqual(result["decision"], "NO_TRADE")
        self.assertIn("(mixed:", result["reason"])

    def test_mixed_macro_is_called_mixed_for_blocked_shorts(self):
        data = _base_engine_data()
        data["structure"] = {"D1": "DOWN", "H4": "DOWN", "H1": "DOWN"}
        data["macro"] = {"dxy_trend": "DOWN", "us10y_trend": "NEUTRAL"}

        result = evaluate_short(data)

        self.assertEqual(result["decision"], "NO_TRADE")
        self.assertIn("(mixed:", result["reason"])


if __name__ == "__main__":
    unittest.main()
