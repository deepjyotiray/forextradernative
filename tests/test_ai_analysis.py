import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from engine.ai_analysis import AIAnalysisService, build_analysis_snapshot, build_analysis_snapshot_for_date
from engine.analytics_api import router as analytics_router


class AIAnalysisServiceTests(unittest.TestCase):
    @patch("engine.ai_analysis.generate_comprehensive_report")
    @patch("engine.ai_analysis.get_decision_summary")
    @patch("engine.ai_analysis.OrderDatabase")
    @patch("engine.ai_analysis._should_use_lightweight_snapshot", return_value=False)
    def test_build_analysis_snapshot_uses_existing_reports(
        self,
        _lightweight_mock,
        order_database_cls,
        get_decision_summary_mock,
        generate_report_mock,
    ):
        generate_report_mock.return_value = {
            "performance_summary": {"total_trades": 12, "win_rate": 0.58, "total_pnl": 18.4},
            "directional_analysis": {"better_direction": "LONG", "LONG": {"trade_count": 7}, "SHORT": {"trade_count": 5}},
            "session_analysis": {"best_session": {"name": "LONDON"}, "worst_session": {"name": "NY"}},
            "quality_analysis": {"high_quality": {"trade_count": 4}},
            "risk_analysis": {"max_consecutive_losses": 2},
            "recommendations": [{"message": "Review breakeven exits."}],
        }
        get_decision_summary_mock.return_value = {
            "total_decisions": 20,
            "trades_taken": 12,
            "trades_skipped": 8,
            "completion_rate": 0.91,
            "conversion_rate": 0.6,
        }
        order_database_cls.return_value.get_trade_outcome_review.return_value = {
            "summary": {"trades": 12, "wins": 7, "losses": 4, "breakeven": 1},
            "close_reason_categories": [{"close_reason_category": "tp", "trades": 5}],
            "strategies": [{"strategy": "SMC_CONFLUENCE", "trades": 6}],
            "profiles": [{"profile_name": "swing_fast", "trades": 8}],
            "volume_ratio_buckets": [{"volume_ratio_bucket": "strong(1.2-1.5x)", "trades": 4}],
            "pressure_score_buckets": [{"pressure_score_bucket": "strong_long(>=0.35)", "trades": 3}],
            "breakeven_review": {"count": 1},
        }

        snapshot = build_analysis_snapshot(14)

        self.assertEqual(snapshot["period_days"], 14)
        self.assertEqual(snapshot["performance_summary"]["total_trades"], 12)
        self.assertEqual(snapshot["decision_summary"]["total_decisions"], 20)
        self.assertEqual(snapshot["directional_analysis"]["better_direction"], "LONG")
        self.assertEqual(snapshot["order_review"]["breakeven_review"]["count"], 1)
        self.assertEqual(snapshot["recommendations"], ["Review breakeven exits."])

    @patch("engine.ai_analysis.get_decision_summary")
    @patch("engine.ai_analysis.OrderDatabase")
    @patch("engine.ai_analysis._should_use_lightweight_snapshot", return_value=True)
    def test_build_analysis_snapshot_skips_heavy_report_when_forced(
        self,
        _lightweight_mock,
        order_database_cls,
        get_decision_summary_mock,
    ):
        get_decision_summary_mock.return_value = {
            "total_decisions": 5,
            "trades_taken": 2,
            "trades_skipped": 3,
        }
        order_database_cls.return_value.get_trade_outcome_review.return_value = {
            "summary": {"trades": 2, "win_rate": 0.5, "total_pnl": 4.0},
            "close_reason_categories": [],
            "strategies": [],
            "profiles": [],
            "volume_ratio_buckets": [],
            "pressure_score_buckets": [],
            "breakeven_review": {},
        }

        snapshot = build_analysis_snapshot(7)

        self.assertEqual(snapshot["performance_summary"]["total_trades"], 2)
        self.assertEqual(snapshot["decision_summary"]["total_decisions"], 5)
        self.assertIn("Skipped full attribution-backed report", snapshot["report_error"])

    @patch("engine.ai_analysis.get_recent_attributions")
    @patch("engine.ai_analysis.OrderDatabase")
    def test_build_analysis_snapshot_for_date_filters_exact_day(
        self,
        order_database_cls,
        recent_attributions_mock,
    ):
        order_database_cls.return_value.get_trade_outcome_review_for_ist_date.return_value = {
            "summary": {"trades": 2, "wins": 1, "losses": 1, "breakeven": 0, "total_pnl": 1.5},
            "close_reason_categories": [{"close_reason_category": "tp", "trades": 1}],
            "strategies": [{"strategy": "SMC_CONFLUENCE", "trades": 2, "total_pnl": 1.5}],
            "profiles": [],
            "volume_ratio_buckets": [],
            "pressure_score_buckets": [],
            "breakeven_review": {"count": 0},
        }
        recent_attributions_mock.return_value = [
            {
                "timestamp": "2026-05-22T10:15:00+05:30",
                "unix_time": 1,
                "trade_id": "t1",
                "decision": "TRADE_TAKEN",
                "trade_completed": True,
                "setup_direction": "LONG",
                "session": "LONDON",
                "quality_score": 0.81,
                "pnl": 3.0,
                "outcome": "WIN",
                "trade_duration": 120,
            },
            {
                "timestamp": "2026-05-22T11:20:00+05:30",
                "unix_time": 2,
                "trade_id": "t2",
                "decision": "TRADE_TAKEN",
                "trade_completed": True,
                "setup_direction": "SHORT",
                "session": "NY",
                "quality_score": 0.62,
                "pnl": -1.5,
                "outcome": "LOSS",
                "trade_duration": 180,
            },
            {
                "timestamp": "2026-05-22T12:00:00+05:30",
                "unix_time": 3,
                "decision": "TRADE_SKIPPED",
                "reason": "spread too high",
            },
            {
                "timestamp": "2026-05-23T09:00:00+05:30",
                "unix_time": 4,
                "decision": "TRADE_SKIPPED",
                "reason": "different day",
            },
        ]

        snapshot = build_analysis_snapshot_for_date("2026-05-22")

        self.assertEqual(snapshot["target_ist_date"], "2026-05-22")
        self.assertEqual(snapshot["performance_summary"]["total_trades"], 2)
        self.assertEqual(snapshot["decision_summary"]["total_decisions"], 3)
        self.assertEqual(snapshot["decision_summary"]["trades_skipped"], 1)
        self.assertEqual(snapshot["directional_analysis"]["better_direction"], "LONG")

    def test_generate_summary_returns_disabled_without_api_key(self):
        service = AIAnalysisService(api_key="", enabled=True)

        result = service.generate_summary(days=7)

        self.assertFalse(result["enabled"])
        self.assertEqual(result["status"], "disabled")
        self.assertIn("OPENAI_API_KEY", result["message"])
        self.assertIn("NVIDIA_API_KEY", result["message"])

    @patch("engine.ai_analysis.build_analysis_snapshot")
    @patch.object(AIAnalysisService, "_request_openai_analysis")
    def test_generate_summary_returns_ready_payload(self, request_mock, build_snapshot_mock):
        build_snapshot_mock.return_value = {
            "period_days": 7,
            "performance_summary": {"total_trades": 4, "win_rate": 0.5},
        }
        request_mock.return_value = "- Stable sample output\nPriority: review low-sample bias."
        service = AIAnalysisService(api_key="test-key", enabled=True, model="gpt-5.4-mini")

        result = service.generate_summary(days=7, refresh=True)

        self.assertTrue(result["enabled"])
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["model"], "gpt-5.4-mini")
        self.assertIn("Priority:", result["analysis"])
        self.assertEqual(result["snapshot"]["period_days"], 7)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "", "NVIDIA_API_KEY": "nv-test-key"}, clear=False)
    @patch("engine.ai_analysis.build_analysis_snapshot")
    @patch.object(AIAnalysisService, "_request_nvidia_analysis")
    def test_generate_summary_uses_nvidia_fallback(self, request_mock, build_snapshot_mock):
        build_snapshot_mock.return_value = {
            "period_days": 7,
            "performance_summary": {"total_trades": 4, "win_rate": 0.5},
        }
        request_mock.return_value = "- NVIDIA sample output\nPriority: review low-sample bias."
        service = AIAnalysisService(api_key=None, enabled=True, model="mistralai/mistral-medium-3.5-128b")

        result = service.generate_summary(days=7, refresh=True)

        self.assertTrue(result["enabled"])
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["provider"], "nvidia")
        self.assertEqual(result["model"], "mistralai/mistral-medium-3.5-128b")
        self.assertIn("Priority:", result["analysis"])
        request_mock.assert_called_once()

    @patch("engine.ai_analysis.build_analysis_snapshot_for_date")
    @patch.object(AIAnalysisService, "_request_openai_analysis")
    def test_generate_day_summary_returns_ready_payload(self, request_mock, build_snapshot_mock):
        build_snapshot_mock.return_value = {
            "target_ist_date": "2026-05-22",
            "performance_summary": {"total_trades": 3, "win_rate": 0.667},
        }
        request_mock.return_value = "- Day looks stable\nPriority: review the losing setup."
        service = AIAnalysisService(api_key="test-key", enabled=True, model="gpt-5.4-mini")

        result = service.generate_day_summary("2026-05-22", refresh=True)

        self.assertTrue(result["enabled"])
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["target_ist_date"], "2026-05-22")
        self.assertIn("Priority:", result["analysis"])


class AnalyticsAISummaryApiTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(analytics_router)
        self.client = TestClient(app)

    def test_ai_summary_endpoint_returns_service_payload(self):
        payload = {
            "enabled": True,
            "status": "ready",
            "days": 30,
            "model": "gpt-5.4-mini",
            "generated_at": "2026-04-29T00:00:00+00:00",
            "cached": False,
            "analysis": "- Snapshot looks stable.\nPriority: validate sample size.",
        }
        with patch("engine.analytics_api.ai_analysis_service.generate_summary", return_value=payload) as generate_summary_mock:
            response = self.client.get("/analytics/api/ai-summary?days=30&refresh=1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ready")
        generate_summary_mock.assert_called_once_with(days=30, refresh=True)

    def test_day_summary_endpoint_returns_service_payload(self):
        payload = {
            "enabled": True,
            "status": "ready",
            "target_ist_date": "2026-05-22",
            "model": "gpt-5.4-mini",
            "generated_at": "2026-05-23T00:00:00+00:00",
            "cached": False,
            "analysis": "- Day-specific snapshot looks stable.\nPriority: compare the losing setup.",
        }
        with patch("engine.analytics_api.ai_analysis_service.generate_day_summary", return_value=payload) as generate_summary_mock:
            response = self.client.get("/analytics/api/day-summary?date=2026-05-22&refresh=1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["target_ist_date"], "2026-05-22")
        generate_summary_mock.assert_called_once_with(ist_date="2026-05-22", refresh=True)


if __name__ == "__main__":
    unittest.main()
