import unittest

from engine.ai_manual_trade_ideas import (
    build_manual_trade_context,
    infer_pending_order_type,
    normalize_manual_trade_idea,
)


class AIManualTradeIdeaTests(unittest.TestCase):
    def test_infer_pending_order_type_buy_limit_and_sell_stop(self):
        buy_type = infer_pending_order_type("BUY", 2299.8, {"bid": 2300.0, "ask": 2300.2})
        sell_type = infer_pending_order_type("SELL", 2299.5, {"bid": 2300.0, "ask": 2300.2})

        self.assertEqual(buy_type, "BUY_LIMIT")
        self.assertEqual(sell_type, "SELL_STOP")

    def test_normalize_manual_trade_idea_keeps_valid_buy_setup(self):
        idea = normalize_manual_trade_idea(
            {
                "action": "BUY",
                "confidence": 0.74,
                "entry": 2300.4,
                "sl": 2299.6,
                "tp": 2302.0,
                "rr": 2.0,
                "setup_type": "LIMIT",
                "reasoning": "Pullback into support with bullish pressure.",
                "invalidation": "Lose support",
                "checklist": ["Bias aligned", "Spread stable"],
            },
            {"bid": 2300.1, "ask": 2300.3},
        )

        self.assertEqual(idea["action"], "BUY")
        self.assertTrue(idea["actionable"])
        self.assertEqual(idea["pending_order_type"], "BUY_STOP")
        self.assertAlmostEqual(idea["current_price"], 2300.3)

    def test_normalize_manual_trade_idea_downgrades_invalid_setup_to_wait(self):
        idea = normalize_manual_trade_idea(
            {
                "action": "SELL",
                "confidence": 0.81,
                "entry": 2300.0,
                "sl": 2299.5,
                "tp": 2301.0,
                "setup_type": "STOP",
            },
            {"bid": 2299.9, "ask": 2300.1},
        )

        self.assertEqual(idea["action"], "WAIT")
        self.assertFalse(idea["actionable"])
        self.assertEqual(idea["pending_order_type"], "NONE")

    def test_build_manual_trade_context_stays_compact(self):
        context = build_manual_trade_context(
            symbol="XAUUSD",
            timeframe="M5",
            tick={"bid": 2300.1, "ask": 2300.3, "spread": 0.2},
            session="LONDON",
            market_open=True,
            indicators={"ema9": 2299.8, "ema15": 2299.9, "atr14": 1.2, "rsi": 56.0},
            regime={"state": "TRENDING", "direction": "LONG", "trade_allowed": True},
            bias={"direction": "LONG", "confidence": 0.71},
            liquidity={"key_levels": {"session_high": 2302.4, "session_low": 2297.8}},
            tick_pressure={"ready": True, "directional_bias": "LONG", "pressure_score": 0.31},
            xgb_live_prediction={"available": True, "buy_probability": 0.68, "sell_probability": 0.32},
            account={"balance": 1000, "equity": 1005},
            candles=[{"open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 12}] * 30,
            support_zones=[{"zone_low": 2298.0, "zone_high": 2298.6, "strength": 0.7}],
            resistance_zones=[{"zone_low": 2302.0, "zone_high": 2302.5, "strength": 0.66}],
        )

        self.assertEqual(context["symbol"], "XAUUSD")
        self.assertEqual(context["timeframe"], "M5")
        self.assertEqual(len(context["recent_candles"]), 24)
        self.assertEqual(context["levels"]["support_zones"][0]["low"], 2298.0)


if __name__ == "__main__":
    unittest.main()
