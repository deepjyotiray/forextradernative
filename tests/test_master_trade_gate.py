import time
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import config as cfg
from engine.decision_logger import _get_session
from engine.master_trade_gate import evaluate_auto_relax, master_trade_gate
from engine.master_control import MasterControlSystem
from engine.risk_manager import RiskManager
from engine.session_risk_control import SessionRiskController
from engine.session_filter import get_session_state_at
from engine.xgb_model import apply_xgb_filter


class _FakeRiskManager:
    def __init__(self, allowed=True, reason="RISK_PASS"):
        self._allowed = allowed
        self._reason = reason
        self._daily_status = {"consecutive_losses": 0, "pnl": 0.0}
        self._last_trade_time = time.time() - 7200
        self._last_trade_outcome = ""
        self._floating_pnl = 0.0

    def can_trade(self, account, open_count):
        return self._allowed, self._reason

    @property
    def daily_status(self):
        return dict(self._daily_status)

    @property
    def last_trade_time(self):
        return self._last_trade_time

    @property
    def last_trade_outcome(self):
        return self._last_trade_outcome


class _FakeM15Context:
    def __init__(self, allowed=True, reason="M15_CONTEXT_PASS", candle_confirmation=True, spike_allowed=True):
        self.allowed = allowed
        self.reason = reason
        self.candle_confirmation = candle_confirmation
        self.spike_allowed = spike_allowed

    def evaluate_trade_context(self, data, direction, signal=None):
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "distance_atr": 0.12,
            "candle_confirmation": self.candle_confirmation,
        }

    def evaluate_spike_context(self, data):
        return {
            "allowed": self.spike_allowed,
            "reason": "SPIKE_PASS" if self.spike_allowed else "SPIKE_BLOCK: recent spike",
            "spike_detected": not self.spike_allowed,
        }


class MasterTradeGateTests(unittest.TestCase):
    def setUp(self):
        self._original = {
            "TRADE_SCORE_MIN": cfg.TRADE_SCORE_MIN,
            "TRADE_SCORE_PREMIUM": cfg.TRADE_SCORE_PREMIUM,
            "TRADE_SCORE_STRICT_AFTER_LOSS": cfg.TRADE_SCORE_STRICT_AFTER_LOSS,
            "AUTO_RELAX_ENABLED": cfg.AUTO_RELAX_ENABLED,
            "AUTO_RELAX_ONLY_GOOD_SESSION": cfg.AUTO_RELAX_ONLY_GOOD_SESSION,
            "AUTO_RELAX_BLOCK_AFTER_LOSS": cfg.AUTO_RELAX_BLOCK_AFTER_LOSS,
            "AUTO_RELAX_BLOCK_DRAWDOWN": cfg.AUTO_RELAX_BLOCK_DRAWDOWN,
            "AUTO_RELAX_REQUIRE_STABLE_SPREAD": cfg.AUTO_RELAX_REQUIRE_STABLE_SPREAD,
            "TIER1_SPREAD_RECHECK_MAX_DELTA": cfg.TIER1_SPREAD_RECHECK_MAX_DELTA,
            "TIER1_MAX_ENTRY_DELAY_MS": cfg.TIER1_MAX_ENTRY_DELAY_MS,
            "STOP_AFTER_CONSECUTIVE_LOSSES": cfg.STOP_AFTER_CONSECUTIVE_LOSSES,
            "DAILY_PROFIT_LOCK": cfg.DAILY_PROFIT_LOCK,
            "DAILY_PROFIT_LOCK_AFTER": cfg.DAILY_PROFIT_LOCK_AFTER,
            "DAILY_MAX_LOSS": cfg.DAILY_MAX_LOSS,
            "MAX_OPEN_TRADES": cfg.MAX_OPEN_TRADES,
            "MAX_TRADES_PER_DAY": cfg.MAX_TRADES_PER_DAY,
            "XGB_BYPASS_ENABLED": cfg.XGB_BYPASS_ENABLED,
            "XGB_TRAINING_ENABLED": cfg.XGB_TRAINING_ENABLED,
            "XGB_BLOCKING_ENABLED": cfg.XGB_BLOCKING_ENABLED,
            "XGB_BLOCK_THRESHOLD": cfg.XGB_BLOCK_THRESHOLD,
            "XGB_BLEND_CONFIDENCE_ENABLED": cfg.XGB_BLEND_CONFIDENCE_ENABLED,
            "XGB_LOG_CONFIDENCE_ENABLED": cfg.XGB_LOG_CONFIDENCE_ENABLED,
        }
        cfg.TRADE_SCORE_MIN = 75
        cfg.TRADE_SCORE_PREMIUM = 85
        cfg.TRADE_SCORE_STRICT_AFTER_LOSS = 85
        cfg.AUTO_RELAX_ENABLED = True
        cfg.AUTO_RELAX_ONLY_GOOD_SESSION = True
        cfg.AUTO_RELAX_BLOCK_AFTER_LOSS = True
        cfg.AUTO_RELAX_BLOCK_DRAWDOWN = True
        cfg.AUTO_RELAX_REQUIRE_STABLE_SPREAD = True
        cfg.TIER1_SPREAD_RECHECK_MAX_DELTA = 0.03
        cfg.TIER1_MAX_ENTRY_DELAY_MS = 800
        cfg.STOP_AFTER_CONSECUTIVE_LOSSES = 2
        cfg.DAILY_PROFIT_LOCK = True
        cfg.DAILY_PROFIT_LOCK_AFTER = 10.0
        cfg.DAILY_MAX_LOSS = -8.0
        cfg.MAX_OPEN_TRADES = 1
        cfg.MAX_TRADES_PER_DAY = 3
        cfg.XGB_BYPASS_ENABLED = False
        cfg.XGB_TRAINING_ENABLED = True
        cfg.XGB_BLOCKING_ENABLED = False
        cfg.XGB_BLOCK_THRESHOLD = 0.35
        cfg.XGB_BLEND_CONFIDENCE_ENABLED = False
        cfg.XGB_LOG_CONFIDENCE_ENABLED = True

    def tearDown(self):
        for key, value in self._original.items():
            setattr(cfg, key, value)

    def _market_state(self, hour=8, spread=0.18, spread_stable=True, bias="LONG"):
        now_utc = datetime(2026, 4, 1, hour, 0, tzinfo=timezone.utc)
        return {
            "now_utc": now_utc,
            "tick": {"bid": 4700.0, "ask": 4700.18, "spread": spread},
            "tick_snapshot": {"spread_stable": spread_stable, "spread_mean": spread, "ready": True},
            "indicators": {"atr14": 1.2, "atr_ratio": 1.0, "body_ratio": 0.72},
            "bias": {"direction": bias},
            "account": {"balance": 1000.0, "equity": 1000.0},
            "positions": [],
        }

    def _signal(self, bias="LONG", entry_spread=0.15):
        return {
            "signal": "BUY",
            "_signal_family": "SMC",
            "_sweep_confirmed": True,
            "_candle_confirmation": True,
            "_ema_aligned": True,
            "_body_ratio": 0.72,
            "_entry_spread": entry_spread,
            "_signal_generated_ts_ms": time.time() * 1000.0,
            "rr": 1.6,
            "bias": {"direction": bias},
        }

    def test_trade_blocked_when_spread_widens_after_signal(self):
        signal = self._signal(entry_spread=0.10)
        result = master_trade_gate(
            signal,
            self._market_state(spread=0.15),
            risk_manager=_FakeRiskManager(),
            m15_context_provider=_FakeM15Context(),
        )
        self.assertFalse(result["allowed"])
        self.assertEqual(result["reason"], "TIER1_BLOCK: Spread widened since signal")

    def test_trade_blocked_when_entry_delay_exceeds_max(self):
        signal = self._signal()
        signal["_signal_generated_ts_ms"] = time.time() * 1000.0 - 1200.0
        result = master_trade_gate(
            signal,
            self._market_state(),
            risk_manager=_FakeRiskManager(),
            m15_context_provider=_FakeM15Context(),
        )
        self.assertFalse(result["allowed"])
        self.assertEqual(result["reason"], "TIER1_BLOCK: Entry delay exceeded")

    def test_trade_blocked_when_m15_context_disagrees(self):
        result = master_trade_gate(
            self._signal(),
            self._market_state(),
            risk_manager=_FakeRiskManager(),
            m15_context_provider=_FakeM15Context(allowed=False, reason="M15_CONTEXT_BLOCK: direction not aligned"),
        )
        self.assertFalse(result["allowed"])
        self.assertEqual(result["reason"], "M15_CONTEXT_BLOCK: direction not aligned")

    def test_trade_blocked_when_score_below_minimum(self):
        cfg.AUTO_RELAX_ENABLED = False
        signal = self._signal()
        signal["_sweep_confirmed"] = False
        signal["_ema_aligned"] = False
        signal["_body_ratio"] = 0.30
        result = master_trade_gate(
            signal,
            self._market_state(),
            risk_manager=_FakeRiskManager(),
            m15_context_provider=_FakeM15Context(candle_confirmation=True),
        )
        self.assertFalse(result["allowed"])
        self.assertEqual(result["reason"], "TRADE_SCORE_BLOCK: score=50 min=75")

    def test_trade_allowed_when_score_and_confirmation_pass(self):
        result = master_trade_gate(
            self._signal(),
            self._market_state(),
            risk_manager=_FakeRiskManager(),
            m15_context_provider=_FakeM15Context(),
        )
        self.assertTrue(result["allowed"])
        self.assertGreaterEqual(result["score"], 75)
        self.assertEqual(result["confirmation_count"], 4)

    def test_counter_trend_trade_blocked_without_premium_conditions(self):
        signal = self._signal(bias="SHORT")
        signal["_ema_aligned"] = False
        signal["_body_ratio"] = 0.40
        result = master_trade_gate(
            signal,
            self._market_state(bias="SHORT"),
            risk_manager=_FakeRiskManager(),
            m15_context_provider=_FakeM15Context(),
        )
        self.assertFalse(result["allowed"])
        self.assertEqual(result["reason"], "COUNTER_TREND_BLOCK: weak counter-trend setup")

    def test_bot_stops_after_two_consecutive_losses(self):
        risk = RiskManager()
        risk.set_start_balance(1000.0)
        risk._consecutive_losses = 2
        allowed, reason = risk.can_trade({"balance": 1000.0, "equity": 1000.0}, 0)
        self.assertFalse(allowed)
        self.assertEqual(reason, "RISK_BLOCK: consecutive losses reached")

    def test_bot_locks_after_daily_profit_target(self):
        risk = RiskManager()
        risk.set_start_balance(1000.0)
        risk._daily_pnl = 11.0
        allowed, reason = risk.can_trade({"balance": 1000.0, "equity": 1000.0}, 0)
        self.assertFalse(allowed)
        self.assertEqual(reason, "RISK_LOCK: daily profit target reached")

    def test_bot_stops_after_daily_max_loss(self):
        risk = RiskManager()
        risk.set_start_balance(1000.0)
        risk._daily_pnl = -9.0
        allowed, reason = risk.can_trade({"balance": 1000.0, "equity": 1000.0}, 0)
        self.assertFalse(allowed)
        self.assertEqual(reason, "RISK_BLOCK: daily max loss reached")

    def test_daily_trade_count_limit_can_be_disabled(self):
        cfg.MAX_TRADES_PER_DAY = 0
        risk = RiskManager()
        risk.set_start_balance(1000.0)
        risk._daily_trades = 99
        allowed, reason = risk.can_trade({"balance": 1000.0, "equity": 1000.0}, 0)
        self.assertTrue(allowed)
        self.assertEqual(reason, "OK")

    def test_session_trade_count_limit_can_be_disabled(self):
        cfg.SESSION_MAX_TRADES = 0
        controller = SessionRiskController()
        controller._session_trades = 99
        controller._current_session = "LONDON"
        controller._current_date = "2026-04-01"
        fixed_now = datetime(2026, 4, 1, 8, 0, tzinfo=timezone.utc)
        with patch("engine.session_risk_control._now_utc", return_value=fixed_now), patch(
            "engine.session_risk_control.is_market_open", return_value=True
        ), patch("engine.session_risk_control.get_session", return_value="LONDON"):
            result = controller.check_trade_allowed()
        self.assertTrue(result["allowed"])

    def test_xgb_filter_bypass_returns_original_signal(self):
        cfg.XGB_BYPASS_ENABLED = True
        signal = {"signal": "BUY", "confidence": 0.61}
        filtered = apply_xgb_filter(signal, {}, {}, {}, {"spread": 0.12})
        self.assertEqual(filtered["signal"], "BUY")
        self.assertTrue(filtered.get("_xgb_bypassed"))

    def test_xgb_filter_logs_confidence_without_blocking(self):
        cfg.XGB_BYPASS_ENABLED = False
        cfg.XGB_BLOCKING_ENABLED = False
        signal = {"signal": "BUY", "confidence": 0.61}
        with patch("engine.xgb_model.xgb_model.is_trained", True), patch(
            "engine.xgb_model.xgb_model.predict_win_prob", return_value=0.22
        ):
            filtered = apply_xgb_filter(signal, {}, {}, {}, {"spread": 0.12})
        self.assertEqual(filtered["signal"], "BUY")
        self.assertEqual(filtered.get("xgb_prob"), 0.22)
        self.assertEqual(filtered.get("_xgb_prob"), 0.22)
        self.assertTrue(filtered.get("_xgb_trained"))

    def test_pre_trade_validation_skips_xgb_when_bypassed(self):
        cfg.XGB_BYPASS_ENABLED = True
        signal = {"signal": "BUY", "confidence": 0.7}
        system = MasterControlSystem()
        with patch("engine.master_control.check_trade_allowed", return_value={"allowed": True, "risk_status": {}}), patch(
            "engine.master_control.check_pacing_allowed", return_value={"allowed": True}
        ), patch("engine.master_control.get_current_parameters", return_value={}), patch(
            "engine.master_control.get_active_relaxations", return_value={}
        ), patch("engine.master_control.apply_xgb_filter") as mock_apply:
            allowed, validation = system.pre_trade_validation(signal, "SMC_CONFLUENCE", {"tick": {}, "indicators": {}, "regime": {}, "bias": {}})
        self.assertTrue(allowed)
        self.assertEqual(validation["blocks"], [])
        mock_apply.assert_not_called()

    def test_auto_relax_does_not_activate_after_loss(self):
        risk = _FakeRiskManager()
        risk._last_trade_outcome = "LOSS"
        state = evaluate_auto_relax(
            self._market_state(),
            risk,
            {"spread_stable": True},
            get_session_state_at(datetime(2026, 4, 1, 8, 0, tzinfo=timezone.utc)),
            {"allowed": True, "spike_detected": False},
        )
        self.assertFalse(state["active"])
        self.assertEqual(state["reason"], "AUTO_RELAX_BLOCK: last trade was a loss")

    def test_auto_relax_activates_only_in_stable_london_or_new_york(self):
        risk = _FakeRiskManager()
        london_state = evaluate_auto_relax(
            self._market_state(hour=8),
            risk,
            {"spread_stable": True},
            get_session_state_at(datetime(2026, 4, 1, 8, 0, tzinfo=timezone.utc)),
            {"allowed": True, "spike_detected": False},
        )
        asia_state = evaluate_auto_relax(
            self._market_state(hour=2),
            risk,
            {"spread_stable": True},
            get_session_state_at(datetime(2026, 4, 1, 2, 0, tzinfo=timezone.utc)),
            {"allowed": True, "spike_detected": False},
        )
        self.assertTrue(london_state["active"])
        self.assertFalse(asia_state["active"])

    def test_session_detector_is_consistent_for_logging(self):
        now_utc = datetime(2026, 4, 1, 14, 0, tzinfo=timezone.utc)
        session_state = get_session_state_at(now_utc)
        self.assertEqual(_get_session(now_utc), session_state["label"])
        self.assertEqual(session_state["session"], "NEW_YORK")


if __name__ == "__main__":
    unittest.main()
