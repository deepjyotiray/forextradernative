import unittest
from unittest.mock import Mock, patch

from engine.ai_trade_advisor import AITradeAdvisorService, build_trade_review_context
from engine.decision_logger import get_recent_ai_reviews


class AITradeAdvisorServiceTests(unittest.TestCase):
    def test_build_trade_review_context_keeps_payload_compact(self):
        context = build_trade_review_context(
            "SMC_CONFLUENCE",
            {
                "signal": "BUY",
                "confidence": 0.68,
                "entry": 2350.4,
                "sl": 2349.8,
                "tp": 2351.8,
                "rr": 2.3,
                "reason": "M15 rejection with bias support",
                "_entry_volume_ratio": 1.15,
            },
            {
                "symbol": "XAUUSD",
                "tick": {"ask": 2350.42, "spread": 0.18},
                "indicators": {"atr": 1.4, "rsi": 54.2},
                "bias": {"direction": "LONG", "confidence": 0.74},
                "regime": {"state": "TRENDING"},
                "positions": [1],
                "account": {"balance": 1000, "equity": 1005},
            },
            {
                "score": 78,
                "required_score": 75,
                "confirmation_count": 3,
                "required_rr": 1.4,
                "session_state": {"session": "LONDON", "reason": "SESSION_PASS"},
                "m15_context": {"reason": "M15_CONTEXT_PASS"},
            },
        )

        self.assertEqual(context["setup"]["action"], "BUY")
        self.assertEqual(context["gate"]["score"], 78)
        self.assertEqual(context["market"]["session"], "LONDON")
        self.assertEqual(context["market"]["open_positions"], 1)

    def test_review_trade_returns_disabled_without_api_key(self):
        service = AITradeAdvisorService(api_key="", enabled=True)

        result = service.review_trade(
            strategy="SMC_CONFLUENCE",
            signal={"signal": "BUY", "confidence": 0.6},
            market_state={},
            gate_result={"score": 76, "required_score": 75},
        )

        self.assertEqual(result["status"], "disabled")
        self.assertFalse(result["used"])
        self.assertIn("NVIDIA_API_KEY", result["reason"])

    def test_review_trade_skips_clear_high_quality_setup(self):
        service = AITradeAdvisorService(
            api_key="test-key",
            enabled=True,
            review_max_confidence=0.72,
            review_score_buffer=5,
        )

        result = service.review_trade(
            strategy="SMC_CONFLUENCE",
            signal={"signal": "BUY", "confidence": 0.91},
            market_state={},
            gate_result={"score": 90, "required_score": 75, "counter_trend": False},
        )

        self.assertEqual(result["status"], "skipped")
        self.assertFalse(result["used"])
        self.assertEqual(result["reason"], "clear_high_quality_setup")

    @patch("engine.ai_trade_advisor.requests.post")
    def test_review_trade_blocks_on_confident_reject(self, post_mock):
        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "```json\n"
                            "{\"decision\":\"REJECT\",\"confidence\":0.86,"
                            "\"reason\":\"Counter-trend with weak pressure alignment\","
                            "\"risk_flags\":[\"counter_trend\",\"weak_pressure\"]}\n"
                            "```"
                        )
                    }
                }
            ]
        }
        post_mock.return_value = response

        service = AITradeAdvisorService(
            api_key="test-key",
            enabled=True,
            min_seconds_between_calls=0,
            review_max_confidence=0.95,
            reject_confidence_min=0.70,
        )
        result = service.review_trade(
            strategy="SMC_CONFLUENCE",
            signal={"signal": "SELL", "confidence": 0.69},
            market_state={"tick": {"bid": 2300.1, "spread": 0.2}},
            gate_result={"score": 77, "required_score": 75, "counter_trend": True},
        )

        self.assertEqual(result["status"], "ready")
        self.assertTrue(result["used"])
        self.assertTrue(result["should_block"])
        self.assertEqual(result["decision"], "REJECT")
        self.assertGreaterEqual(result["confidence"], 0.86)
        self.assertIn("counter_trend", result["risk_flags"])
        self.assertIn("Review this pre-approved trade candidate", result["request_text"])
        self.assertIn("\"decision\":\"REJECT\"", result["response_text"])

    @patch("engine.decision_logger.get_recent_decisions")
    def test_recent_ai_reviews_extracts_transcripts(self, recent_decisions_mock):
        recent_decisions_mock.return_value = [
            {
                "timestamp": "2026-04-30T10:00:00+05:30",
                "strategy": "SMC_CONFLUENCE",
                "decision": "TRADE_SKIPPED",
                "reason": "AI_BLOCK: weak setup",
                "direction": "BUY",
                "signal_confidence": 0.66,
                "validation_result": {
                    "ai_trade_review": {
                        "status": "ready",
                        "used": True,
                        "decision": "REJECT",
                        "confidence": 0.81,
                        "reason": "weak setup",
                        "review_trigger": "counter_trend_setup",
                        "model": "mistralai/mistral-medium-3.5-128b",
                        "request_text": "prompt body",
                        "response_text": "{\"decision\":\"REJECT\"}",
                        "should_block": True,
                    }
                },
            }
        ]

        result = get_recent_ai_reviews(limit=5)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["ai_review_decision"], "REJECT")
        self.assertEqual(result[0]["request_text"], "prompt body")
        self.assertEqual(result[0]["response_text"], "{\"decision\":\"REJECT\"}")


if __name__ == "__main__":
    unittest.main()
