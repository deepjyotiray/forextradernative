import time
import unittest
from unittest.mock import Mock

import pandas as pd

import config as cfg

from engine.trade_manager import TradeManager, TradeRecord


class _FakeBridge:
    def __init__(self):
        self.modify_trade = Mock(return_value={"success": True})
        self.close_trade = Mock(return_value={"success": True, "close_price": 4579.04})


class TradeManagerTests(unittest.TestCase):
    def _build_manager(self):
        manager = TradeManager.__new__(TradeManager)
        manager.bridge = _FakeBridge()
        manager.open_trades = {}
        manager.closed_trades = []
        manager._closing_tickets = set()
        manager._pending_close_reasons = {}
        manager._strategy_lockouts = {}
        manager.order_db = Mock()
        manager.pnl_validator = Mock()
        manager._save_state = Mock()
        manager._log_closed_trade = Mock()
        return manager

    def _scalp_features(self):
        return {
            "profile_name": "scalp",
            "be_trigger_r": 0.22,
            "breakeven_min_hold_seconds": 15,
            "breakeven_volume_hold_ratio": 1.30,
            "min_hold_seconds": 20,
            "early_fail_points": 0.20,
            "early_fail_min_ticks": 3,
            "early_fail_max_ticks": 15,
            "reversal_arm_r": 1.0,
            "reversal_drawdown_pct": 0.65,
            "reversal_floor_r": 0.25,
            "profit_lock_1_arm_r": 0.55,
            "profit_lock_1_r": 0.12,
            "profit_lock_2_arm_r": 0.80,
            "profit_lock_2_r": 0.22,
            "timeout_seconds": 90,
            "timeout_min_progress_r": 0.20,
            "trail_activate_r": 1.20,
            "trail_lock_r": 0.40,
            "velocity_drop_enabled": True,
        }

    def _m15_zone_features(self):
        return {
            "profile_name": "m15_zone_scalp",
            "be_trigger_r": 0.12,
            "breakeven_min_hold_seconds": 8,
            "breakeven_volume_hold_ratio": 0.0,
            "min_hold_seconds": 15,
            "early_fail_points": 0.12,
            "early_fail_min_ticks": 2,
            "early_fail_max_ticks": 12,
            "reversal_arm_r": 0.80,
            "reversal_drawdown_pct": 0.60,
            "reversal_floor_r": 0.10,
            "profit_lock_1_arm_r": 0.55,
            "profit_lock_1_r": 0.12,
            "profit_lock_2_arm_r": 0.80,
            "profit_lock_2_r": 0.22,
            "timeout_seconds": 3600,
            "timeout_min_progress_r": 0.05,
            "trail_activate_r": 0.90,
            "trail_lock_r": 0.20,
            "velocity_drop_enabled": True,
        }

    def _swing_features(self):
        return {
            "profile_name": "swing_fast",
            "be_trigger_r": 0.50,
            "breakeven_min_hold_seconds": 20,
            "breakeven_volume_hold_ratio": 1.20,
            "min_hold_seconds": 30,
            "early_fail_points": 0.25,
            "early_fail_min_ticks": 4,
            "early_fail_max_ticks": 9,
            "reversal_arm_r": 1.20,
            "reversal_drawdown_pct": 0.70,
            "reversal_floor_r": 0.35,
            "timeout_seconds": 120,
            "timeout_min_progress_r": 0.25,
            "trail_activate_r": 1.50,
            "trail_lock_r": 0.60,
            "velocity_drop_enabled": False,
        }

    def test_register_trade_persists_profile_metadata(self):
        manager = self._build_manager()

        manager.register_trade(
            12345, "SELL", 0.02, 4576.38, 4577.96, 4573.20, 1.58,
            strategy="SMC_CONFLUENCE",
            confidence=0.87,
            reason="profile regression test",
            scalp=False,
            be_trigger=0.30,
            timeout=60,
            early_fail=0.25,
            features=self._swing_features(),
            tier1_min_ticks=4,
            tier1_max_ticks=9,
        )

        trade = manager.open_trades[12345]
        self.assertEqual(trade.exit_profile, "swing_fast")
        self.assertEqual(trade.tier1_min_ticks, 4)
        self.assertEqual(trade.tier1_max_ticks, 9)
        self.assertEqual(trade.be_trigger_r, 0.50)
        self.assertEqual(trade.breakeven_min_hold_seconds, 20)
        self.assertEqual(trade.timeout_min_progress_r, 0.25)

        args = manager.order_db.store_order.call_args[0]
        self.assertEqual(args[0], 12345)
        self.assertEqual(args[13]["profile_name"], "swing_fast")
        self.assertEqual(args[13]["trail_lock_r"], 0.60)
        self.assertFalse(args[13]["velocity_drop_enabled"])

    def test_register_trade_uses_profile_be_trigger_when_signal_does_not_override(self):
        manager = self._build_manager()

        manager.register_trade(
            22334, "SELL", 0.01, 4610.0, 4612.0, 4607.0, 2.0,
            strategy="M15_SCALP_DEEP",
            confidence=0.8,
            reason="profile default regression",
            scalp=True,
            features={"profile_name": "m15_scalp_deep"},
        )

        trade = manager.open_trades[22334]
        self.assertEqual(trade.exit_profile, "m15_scalp_deep")
        self.assertEqual(trade.be_trigger_r, cfg.get_exit_profile_config("m15_scalp_deep")["be_trigger_r"])

    def test_register_trade_keeps_m15_scalp_tp_when_fixed_profit_target_disabled(self):
        manager = self._build_manager()

        manager.register_trade(
            22335, "BUY", 0.02, 4579.0, 4576.98, 4582.4, 1.5,
            strategy="M15_ZONE_SCALP",
            confidence=0.85,
            reason="fixed target tp clamp",
            scalp=True,
            features=self._m15_zone_features(),
        )

        trade = manager.open_trades[22335]
        self.assertEqual(trade.tp, 4582.4)
        manager.bridge.modify_trade.assert_not_called()
        args = manager.order_db.store_order.call_args[0]
        self.assertEqual(args[5], 4582.4)

    def test_scalp_reversal_waits_for_min_hold(self):
        manager = self._build_manager()
        trade = TradeRecord(
            8398970520, "BUY", 0.02, 4579.0, 4576.98, 4582.4, 1.5,
            strategy="M15_SCALP_DEEP", scalp=True, be_trigger=0.30, timeout=60,
            early_fail=0.20, features=self._scalp_features(),
        )
        trade.fill_ts = time.time() - 11
        trade.peak_pnl = 2.78
        trade.live_pnl = 0.42

        manager._manage_scalp(trade)

        manager.bridge.close_trade.assert_not_called()
        self.assertEqual(manager._pending_close_reasons, {})

    def test_scalp_reversal_does_not_arm_below_one_r(self):
        manager = self._build_manager()
        trade = TradeRecord(
            1001, "BUY", 0.02, 4579.0, 4576.98, 4582.4, 1.5,
            strategy="M15_SCALP_DEEP", scalp=True, be_trigger=0.30, timeout=60,
            early_fail=0.20, features=self._scalp_features(),
        )
        trade.fill_ts = time.time() - 30
        trade.peak_pnl = 2.70
        trade.live_pnl = 0.40

        manager._manage_scalp(trade)

        manager.bridge.close_trade.assert_not_called()
        self.assertEqual(manager._pending_close_reasons, {})

    def test_scalp_reversal_closes_after_arm_and_large_giveback(self):
        manager = self._build_manager()
        features = self._scalp_features()
        features["profit_lock_1_arm_r"] = 0.0
        features["profit_lock_2_arm_r"] = 0.0
        trade = TradeRecord(
            1002, "BUY", 0.02, 4579.0, 4576.98, 4582.4, 1.5,
            strategy="M15_SCALP_DEEP", scalp=True, be_trigger=0.30, timeout=60,
            early_fail=0.20, features=features,
        )
        trade.fill_ts = time.time() - 30
        trade.peak_pnl = 4.50
        trade.live_pnl = 0.60

        manager._manage_scalp(trade)

        manager.bridge.close_trade.assert_called_once()
        self.assertEqual(manager._pending_close_reasons[1002]["category"], "reversal")
        self.assertEqual(manager._pending_close_reasons[1002]["close_signal_live_pnl"], 0.60)

    def test_scalp_profit_lock_stage_one_tightens_stop(self):
        manager = self._build_manager()
        trade = TradeRecord(
            1015, "BUY", 0.02, 4579.0, 4576.98, 4582.4, 1.5,
            strategy="M15_SCALP_DEEP", scalp=True, be_trigger=0.30, timeout=60,
            early_fail=0.20, features=self._scalp_features(),
        )
        trade.fill_ts = time.time() - 35
        trade.peak_pnl = 1.80
        trade.live_pnl = 0.84

        manager._manage_scalp(trade)

        manager.bridge.modify_trade.assert_called_once_with(1015, 4579.18, 4582.4)
        manager.bridge.close_trade.assert_not_called()
        self.assertEqual(trade.sl, 4579.18)
        self.assertTrue(trade.trail_active)
        manager.order_db.update_management_flags.assert_called_with(1015, trail_active=True)

    def test_scalp_profit_lock_stage_two_tightens_existing_lock(self):
        manager = self._build_manager()
        trade = TradeRecord(
            1016, "BUY", 0.02, 4579.0, 4579.18, 4582.4, 1.5,
            strategy="M15_SCALP_DEEP", scalp=True, be_trigger=0.30, timeout=60,
            early_fail=0.20, features=self._scalp_features(),
        )
        trade.fill_ts = time.time() - 45
        trade.peak_pnl = 2.70
        trade.live_pnl = 1.20
        trade.trail_active = True

        manager._manage_scalp(trade)

        manager.bridge.modify_trade.assert_called_once_with(1016, 4579.33, 4582.4)
        self.assertEqual(trade.sl, 4579.33)
        manager.order_db.update_management_flags.assert_called_with(1016, trail_active=True)

    def test_breakeven_uses_r_multiple(self):
        manager = self._build_manager()
        trade = TradeRecord(
            1003, "BUY", 0.02, 4579.0, 4576.98, 4582.4, 1.5,
            strategy="M15_SCALP_DEEP", scalp=True, be_trigger=0.30, timeout=60,
            early_fail=0.20, features=self._scalp_features(),
        )
        trade.fill_ts = time.time() - 20
        trade.live_pnl = 1.20

        handled = manager._apply_universal_management(trade, {"tick_count": 420, "velocity": 9.0})

        self.assertTrue(handled)
        manager.bridge.modify_trade.assert_called_once_with(1003, 4579.0, 4582.4)
        self.assertTrue(trade.sl_breakeven)
        manager.order_db.update_management_flags.assert_called()

    def test_profit_dollar_ratchet_locks_two_point_five_after_three_dollar_profit(self):
        manager = self._build_manager()
        previous_enabled = cfg.PROFIT_DOLLAR_RATCHET_ENABLED
        previous_levels = cfg.PROFIT_DOLLAR_RATCHET_LEVELS
        cfg.PROFIT_DOLLAR_RATCHET_ENABLED = True
        cfg.PROFIT_DOLLAR_RATCHET_LEVELS = [[3.0, 2.5], [4.0, 3.5]]
        try:
            trade = TradeRecord(
                10030, "SELL", 0.01, 4518.72, 4522.39, 4515.04, 3.67,
                strategy="M15_ZONE_SCALP", scalp=True, be_trigger=0.30, timeout=60,
                early_fail=0.12, features=self._m15_zone_features(),
            )
            trade.live_pnl = 3.12

            manager._manage_trade(trade, {"tick_count": 420, "velocity": 9.0}, {})

            manager.bridge.modify_trade.assert_called_once_with(10030, 4516.22, 4515.04)
            self.assertEqual(trade.sl, 4516.22)
            self.assertTrue(trade.sl_breakeven)
            self.assertTrue(trade.trail_active)
            manager.order_db.update_management_flags.assert_called_with(10030, sl_breakeven=True, trail_active=True)
        finally:
            cfg.PROFIT_DOLLAR_RATCHET_ENABLED = previous_enabled
            cfg.PROFIT_DOLLAR_RATCHET_LEVELS = previous_levels

    def test_profit_dollar_ratchet_applies_before_isolated_swing_manager(self):
        manager = self._build_manager()
        previous_enabled = cfg.PROFIT_DOLLAR_RATCHET_ENABLED
        previous_levels = cfg.PROFIT_DOLLAR_RATCHET_LEVELS
        cfg.PROFIT_DOLLAR_RATCHET_ENABLED = True
        cfg.PROFIT_DOLLAR_RATCHET_LEVELS = [[3.0, 2.5], [4.0, 3.5]]
        try:
            manager._manage_swing_engine = Mock()
            trade = TradeRecord(
                10040, "BUY", 0.01, 4500.0, 4496.0, 4508.0, 4.0,
                strategy="SWING_ENGINE", scalp=False, features={"profile_name": "swing_engine"},
            )
            trade.live_pnl = 4.18

            manager._manage_trade(trade, {"tick_count": 420, "velocity": 9.0}, {})

            manager.bridge.modify_trade.assert_called_once_with(10040, 4503.5, 4508.0)
            manager._manage_swing_engine.assert_not_called()
            self.assertEqual(trade.sl, 4503.5)
        finally:
            cfg.PROFIT_DOLLAR_RATCHET_ENABLED = previous_enabled
            cfg.PROFIT_DOLLAR_RATCHET_LEVELS = previous_levels

    def test_m15_zone_scalp_chokes_small_winner_early(self):
        manager = self._build_manager()
        trade = TradeRecord(
            10034, "BUY", 0.02, 4579.0, 4576.98, 4582.4, 1.5,
            strategy="M15_ZONE_SCALP", scalp=True, be_trigger=0.30, timeout=60,
            early_fail=0.12, features=self._m15_zone_features(),
        )
        trade.fill_ts = time.time() - 5
        trade.live_pnl = 1.62

        handled = manager._apply_universal_management(trade, {"tick_count": 420, "velocity": 9.0})

        self.assertTrue(handled)
        manager.bridge.close_trade.assert_called_once_with(10034)
        manager.bridge.modify_trade.assert_not_called()
        self.assertEqual(manager._pending_close_reasons[10034]["category"], "m15_zone_profit_choke")

    def test_m15_zone_scalp_chokes_profit_before_breakeven(self):
        manager = self._build_manager()
        trade = TradeRecord(
            10037, "BUY", 0.01, 4538.89, 4535.23, 4541.23, 3.67,
            strategy="M15_ZONE_SCALP", scalp=True, be_trigger=0.30, timeout=60,
            early_fail=0.12, features=self._m15_zone_features(),
        )
        trade.fill_ts = time.time() - 20
        trade.live_pnl = 1.46
        trade.peak_pnl = 1.52

        handled = manager._apply_universal_management(trade, {"tick_count": 420, "velocity": 9.0})

        self.assertTrue(handled)
        manager.bridge.close_trade.assert_called_once_with(10037)
        manager.bridge.modify_trade.assert_not_called()
        self.assertEqual(manager._pending_close_reasons[10037]["category"], "m15_zone_profit_choke")

    def test_m15_zone_scalp_chokes_profit_before_time_invested_lock(self):
        manager = self._build_manager()
        previous_enabled = cfg.TIME_INVESTED_PROFIT_LOCK_ENABLED
        previous_seconds = cfg.TIME_INVESTED_PROFIT_LOCK_SECONDS
        previous_usd = cfg.TIME_INVESTED_PROFIT_LOCK_USD
        previous_buffer = cfg.TIME_INVESTED_PROFIT_LOCK_MIN_BUFFER_R
        cfg.TIME_INVESTED_PROFIT_LOCK_ENABLED = True
        cfg.TIME_INVESTED_PROFIT_LOCK_SECONDS = 300
        cfg.TIME_INVESTED_PROFIT_LOCK_USD = 0.75
        cfg.TIME_INVESTED_PROFIT_LOCK_MIN_BUFFER_R = 0.02
        try:
            trade = TradeRecord(
                10038, "BUY", 0.01, 4538.89, 4535.23, 4541.23, 3.67,
                strategy="M15_ZONE_SCALP", scalp=True, be_trigger=0.30, timeout=60,
                early_fail=0.12, features=self._m15_zone_features(),
            )
            trade.fill_ts = time.time() - 320
            trade.live_pnl = 1.46
            trade.peak_pnl = 1.52

            handled = manager._apply_universal_management(trade, {"tick_count": 420, "velocity": 9.0})

            self.assertTrue(handled)
            manager.bridge.close_trade.assert_called_once_with(10038)
            manager.bridge.modify_trade.assert_not_called()
            self.assertEqual(manager._pending_close_reasons[10038]["category"], "m15_zone_profit_choke")
        finally:
            cfg.TIME_INVESTED_PROFIT_LOCK_ENABLED = previous_enabled
            cfg.TIME_INVESTED_PROFIT_LOCK_SECONDS = previous_seconds
            cfg.TIME_INVESTED_PROFIT_LOCK_USD = previous_usd
            cfg.TIME_INVESTED_PROFIT_LOCK_MIN_BUFFER_R = previous_buffer

    def test_m15_zone_scalp_chokes_profit_before_trailing_upgrade(self):
        manager = self._build_manager()
        previous_enabled = cfg.TIME_INVESTED_PROFIT_LOCK_ENABLED
        previous_seconds = cfg.TIME_INVESTED_PROFIT_LOCK_SECONDS
        previous_usd = cfg.TIME_INVESTED_PROFIT_LOCK_USD
        previous_buffer = cfg.TIME_INVESTED_PROFIT_LOCK_MIN_BUFFER_R
        cfg.TIME_INVESTED_PROFIT_LOCK_ENABLED = True
        cfg.TIME_INVESTED_PROFIT_LOCK_SECONDS = 300
        cfg.TIME_INVESTED_PROFIT_LOCK_USD = 0.75
        cfg.TIME_INVESTED_PROFIT_LOCK_MIN_BUFFER_R = 0.02
        try:
            trade = TradeRecord(
                10039, "SELL", 0.01, 4518.72, 4518.72, 4515.04, 3.67,
                strategy="M15_ZONE_SCALP", scalp=True, be_trigger=0.30, timeout=60,
                early_fail=0.12, features=self._m15_zone_features(),
            )
            trade.fill_ts = time.time() - 620
            trade.sl_breakeven = True
            trade.live_pnl = 2.06
            trade.peak_pnl = 3.17

            handled = manager._apply_universal_management(trade, {"tick_count": 420, "velocity": 9.0})

            self.assertTrue(handled)
            manager.bridge.close_trade.assert_called_once_with(10039)
            manager.bridge.modify_trade.assert_not_called()
            self.assertEqual(manager._pending_close_reasons[10039]["category"], "m15_zone_profit_choke")
        finally:
            cfg.TIME_INVESTED_PROFIT_LOCK_ENABLED = previous_enabled
            cfg.TIME_INVESTED_PROFIT_LOCK_SECONDS = previous_seconds
            cfg.TIME_INVESTED_PROFIT_LOCK_USD = previous_usd
            cfg.TIME_INVESTED_PROFIT_LOCK_MIN_BUFFER_R = previous_buffer

    def test_m15_zone_anti_style_disables_tier1_reentry_lockout(self):
        manager = self._build_manager()
        features = self._m15_zone_features()
        features["context_hash"] = "m15-zone|buy|same-candle"
        features["entry_tick_count"] = 10
        trade = TradeRecord(
            10036, "BUY", 0.02, 4579.0, 4576.98, 4582.4, 1.5,
            strategy="M15_ZONE_SCALP", scalp=True, be_trigger=0.30, timeout=60,
            early_fail=0.12, features=features,
        )
        trade.current_price = 4578.80

        handled = manager._apply_universal_management(trade, {"tick_count": 14, "velocity": 9.0})

        self.assertFalse(handled)
        manager.bridge.close_trade.assert_not_called()
        self.assertNotIn("M15_ZONE_SCALP", manager._strategy_lockouts)

    def test_non_m15_profiles_ignore_fixed_profit_target_rule(self):
        manager = self._build_manager()
        trade = TradeRecord(
            10035, "BUY", 0.02, 4579.0, 4576.98, 4582.4, 1.5,
            strategy="SMC_CONFLUENCE", scalp=False, be_trigger=0.30, timeout=60,
            early_fail=0.20, features=self._swing_features(),
        )
        trade.fill_ts = time.time() - 5
        trade.live_pnl = 1.62

        handled = manager._apply_universal_management(trade, {"tick_count": 420, "velocity": 9.0})

        self.assertFalse(handled)
        manager.bridge.close_trade.assert_not_called()

    def test_breakeven_waits_for_min_hold(self):
        manager = self._build_manager()
        trade = TradeRecord(
            10031, "BUY", 0.02, 4579.0, 4576.98, 4582.4, 1.5,
            strategy="M15_SCALP_DEEP", scalp=True, be_trigger=0.30, timeout=60,
            early_fail=0.20, features=self._scalp_features(),
        )
        trade.fill_ts = time.time() - 10
        trade.live_pnl = 1.20

        handled = manager._apply_universal_management(trade, {"tick_count": 420, "velocity": 9.0})

        self.assertFalse(handled)
        self.assertFalse(trade.sl_breakeven)
        manager.bridge.modify_trade.assert_not_called()

    def test_breakeven_waits_while_volume_is_still_strong(self):
        manager = self._build_manager()
        trade = TradeRecord(
            10032, "BUY", 0.02, 4579.0, 4576.98, 4582.4, 1.5,
            strategy="M15_SCALP_DEEP", scalp=True, be_trigger=0.30, timeout=60,
            early_fail=0.20, features=self._scalp_features(),
        )
        trade.fill_ts = time.time() - 20
        trade.live_pnl = 1.20
        market_context = {
            "m1_df": pd.DataFrame({"volume": [100, 100, 100, 100, 100, 100, 100, 100, 180]})
        }

        handled = manager._apply_universal_management(
            trade,
            {"tick_count": 420, "velocity": 9.0},
            market_context,
        )

        self.assertFalse(handled)
        self.assertFalse(trade.sl_breakeven)
        manager.bridge.modify_trade.assert_not_called()

    def test_breakeven_waits_while_tick_pressure_supports_trade(self):
        manager = self._build_manager()
        trade = TradeRecord(
            10033, "BUY", 0.02, 4579.0, 4576.98, 4582.4, 1.5,
            strategy="M15_SCALP_DEEP", scalp=True, be_trigger=0.30, timeout=60,
            early_fail=0.20, features=self._scalp_features(),
        )
        trade.fill_ts = time.time() - 20
        trade.live_pnl = 1.20

        handled = manager._apply_universal_management(
            trade,
            {"tick_count": 420, "velocity": 9.0},
            {"tick_pressure": {"ready": True, "pressure_score": 0.42, "favorable_long": True}},
        )

        self.assertFalse(handled)
        self.assertFalse(trade.sl_breakeven)
        manager.bridge.modify_trade.assert_not_called()

    def test_timeout_requires_both_age_and_weak_progress(self):
        manager = self._build_manager()
        trade = TradeRecord(
            1004, "BUY", 0.02, 4579.0, 4576.98, 4582.4, 1.5,
            strategy="M15_SCALP_DEEP", scalp=True, be_trigger=0.30, timeout=60,
            early_fail=0.20, features=self._scalp_features(),
        )
        trade.fill_ts = time.time() - 95
        trade.live_pnl = 0.15

        handled = manager._apply_universal_management(trade, {"tick_count": 430, "velocity": 8.0})

        self.assertTrue(handled)
        self.assertEqual(manager._pending_close_reasons[1004]["category"], "timeout")

    def test_manual_timeout_extension_persists_and_defers_timeout_exit(self):
        manager = self._build_manager()
        trade = TradeRecord(
            1014, "BUY", 0.02, 4579.0, 4576.98, 4582.4, 1.5,
            strategy="M15_SCALP_DEEP", scalp=True, be_trigger=0.30, timeout=60,
            early_fail=0.20, features=self._scalp_features(),
        )
        trade.fill_ts = time.time() - 95
        trade.live_pnl = 0.15
        manager.open_trades[1014] = trade

        updated = manager.extend_trade_timeout(1014, 300)
        handled = manager._apply_universal_management(trade, {"tick_count": 430, "velocity": 8.0})

        self.assertEqual(updated["timeout_seconds"], 390)
        self.assertEqual(updated["manual_timeout_extension_seconds"], 300)
        self.assertEqual(trade.features["timeout_seconds"], 390)
        self.assertFalse(handled)
        manager.bridge.close_trade.assert_not_called()
        manager._save_state.assert_called()

    def test_manual_timeout_extension_returns_none_for_missing_trade(self):
        manager = self._build_manager()
        self.assertIsNone(manager.extend_trade_timeout(999999, 120))

    def test_manual_exit_trade_records_pending_reason(self):
        manager = self._build_manager()
        trade = TradeRecord(
            1017, "BUY", 0.02, 4579.0, 4576.98, 4582.4, 1.5,
            strategy="M15_SCALP_DEEP", scalp=True, be_trigger=0.30, timeout=60,
            early_fail=0.20, features=self._scalp_features(),
        )
        trade.fill_ts = time.time() - 25
        trade.live_pnl = 0.84
        trade.peak_pnl = 1.20
        manager.open_trades[1017] = trade

        result = manager.manually_exit_trade(1017)

        self.assertTrue(result["success"])
        self.assertEqual(result["close_reason_category"], "manual_exit")
        self.assertEqual(result["reason"], "Manual dashboard exit")
        manager.bridge.close_trade.assert_called_once_with(1017)
        self.assertEqual(manager._pending_close_reasons[1017]["category"], "manual_exit")

    def test_manual_exit_trade_returns_none_for_missing_trade(self):
        manager = self._build_manager()
        self.assertIsNone(manager.manually_exit_trade(999999))

    def test_manual_exit_trade_surfaces_bridge_error(self):
        manager = self._build_manager()
        manager.bridge.close_trade.return_value = {"success": False, "error": "Code 10027: AutoTrading disabled by client"}
        trade = TradeRecord(
            1018, "BUY", 0.02, 4579.0, 4576.98, 4582.4, 1.5,
            strategy="M15_SCALP_DEEP", scalp=True, be_trigger=0.30, timeout=60,
            early_fail=0.20, features=self._scalp_features(),
        )
        manager.open_trades[1018] = trade

        result = manager.manually_exit_trade(1018)

        self.assertFalse(result["success"])
        self.assertIn("AutoTrading disabled", result["error"])
        self.assertNotIn(1018, manager._pending_close_reasons)

    def test_velocity_drop_is_disabled_for_swing_profiles(self):
        manager = self._build_manager()
        trade = TradeRecord(
            1005, "SELL", 0.02, 4576.38, 4577.96, 4573.20, 1.58,
            strategy="SMC_CONFLUENCE", scalp=False, be_trigger=0.30, timeout=60,
            early_fail=0.25, features=self._swing_features(),
        )
        trade.fill_ts = time.time() - 15
        trade.entry_tick_velocity = 10.0
        trade.live_pnl = 0.20
        trade.current_price = 4576.30

        handled = manager._apply_universal_management(trade, {"tick_count": 500, "velocity": 3.0})

        self.assertFalse(handled)
        manager.bridge.close_trade.assert_not_called()

    def test_anti_trade_closes_small_winner_early_to_choke_profit(self):
        manager = self._build_manager()
        features = self._m15_zone_features()
        features["anti_mode"] = True
        features["anti_profit_choke_r"] = 0.12
        trade = TradeRecord(
            10051, "SELL", 0.02, 4579.0, 4584.4, 4578.25, 5.4,
            strategy="M15_ZONE_SCALP", scalp=True, be_trigger=0.30, timeout=60,
            early_fail=0.12, features=features,
        )
        trade.fill_ts = time.time() - 5
        trade.live_pnl = 1.40

        handled = manager._apply_universal_management(trade, {"tick_count": 420, "velocity": 9.0})

        self.assertTrue(handled)
        manager.bridge.close_trade.assert_called_once()
        self.assertEqual(manager._pending_close_reasons[10051]["category"], "anti_profit_choke")

    def test_anti_trade_profile_disables_protective_management_snapshot(self):
        features = self._m15_zone_features()
        features["anti_mode"] = True
        trade = TradeRecord(
            10052, "SELL", 0.02, 4579.0, 4584.4, 4578.25, 5.4,
            strategy="M15_ZONE_SCALP", scalp=True, be_trigger=0.30, timeout=60,
            early_fail=0.12, features=features,
        )

        self.assertEqual(trade.be_trigger_r, 0.0)
        self.assertEqual(trade.profit_lock_1_arm_r, 0.0)
        self.assertEqual(trade.profit_lock_2_arm_r, 0.0)
        self.assertFalse(trade.velocity_drop_enabled)
        self.assertGreaterEqual(trade.timeout_seconds, 12 * 3600)

    def test_m15_zone_profile_uses_anti_style_exit_snapshot(self):
        trade = TradeRecord(
            10053, "BUY", 0.02, 4579.0, 4576.98, 4582.4, 1.5,
            strategy="M15_ZONE_SCALP", scalp=True, be_trigger=0.30, timeout=60,
            early_fail=0.12, features=self._m15_zone_features(),
        )

        self.assertTrue(trade.features.get("m15_zone_anti_exit"))
        self.assertEqual(trade.be_trigger_r, 0.0)
        self.assertEqual(trade.profit_lock_1_arm_r, 0.0)
        self.assertEqual(trade.profit_lock_2_arm_r, 0.0)
        self.assertFalse(trade.velocity_drop_enabled)
        self.assertGreaterEqual(trade.timeout_seconds, 12 * 3600)

    def test_manage_all_records_close_diagnostics(self):
        manager = self._build_manager()
        trade = TradeRecord(
            1006, "BUY", 0.02, 4579.0, 4576.98, 4582.4, 1.5,
            strategy="M15_SCALP_DEEP", scalp=True, be_trigger=0.30, timeout=60,
            early_fail=0.20, features=self._scalp_features(),
        )
        trade.fill_ts = time.time() - 11
        trade.peak_pnl = 2.78
        trade.live_pnl = 0.42
        manager.open_trades[1006] = trade
        manager._pending_close_reasons[1006] = {
            "reason": "Scalp reversal (0.93R -> 0.14R, 85% giveback)",
            "category": "reversal",
            "close_signal_live_pnl": 0.42,
            "close_signal_live_r": 0.14,
            "peak_r": 0.93,
            "held_seconds": 11.0,
            "profile_name": "scalp",
        }
        manager._fetch_closed_pnl = Mock(return_value={"pnl": 0.08, "exit_price": 4579.04, "swap": 0.0, "commission": 0.0})

        closed = manager.manage_all([], {})

        self.assertEqual(closed, [(1006, 0.08, True)])
        self.assertNotIn(1006, manager.open_trades)
        kwargs = manager.order_db.close_order.call_args.kwargs
        self.assertEqual(kwargs["close_reason_category"], "reversal")
        self.assertEqual(kwargs["close_signal_live_pnl"], 0.42)
        self.assertEqual(kwargs["profile_name"], "scalp")
        self.assertEqual(kwargs["mt5_close_reason"], "")
        self.assertAlmostEqual(kwargs["final_r"], 0.027, places=3)
        self.assertAlmostEqual(kwargs["drawdown_from_peak_r"], 0.79, places=3)

    def test_manage_all_uses_mt5_sl_reason_for_breakeven_stop(self):
        manager = self._build_manager()
        trade = TradeRecord(
            1008, "SELL", 0.02, 4572.42, 4572.42, 4568.17, 1.5,
            strategy="M15_SCALP_DEEP", scalp=True, be_trigger=0.30, timeout=60,
            early_fail=0.20, features=self._scalp_features(),
        )
        trade.sl_breakeven = True
        trade.fill_ts = time.time() - 18
        trade.peak_pnl = 1.28
        trade.live_pnl = 0.42
        manager.open_trades[1008] = trade
        manager._fetch_closed_pnl = Mock(return_value={
            "pnl": 0.0,
            "exit_price": 4572.42,
            "swap": 0.0,
            "commission": 0.0,
            "reason_code": 4,
            "reason_name": "sl",
            "deal_comment": "[sl 4572.42]",
            "deal_ticket": 7980325384,
            "deal_time": "2026-04-28T14:09:12+00:00",
        })

        closed = manager.manage_all([], {})

        self.assertEqual(closed, [(1008, 0.0, False)])
        close_args = manager.order_db.close_order.call_args.args
        self.assertEqual(close_args[2], "MT5_SL")
        kwargs = manager.order_db.close_order.call_args.kwargs
        self.assertEqual(kwargs["close_reason_category"], "breakeven_stop")
        self.assertEqual(kwargs["mt5_close_reason"], "sl")
        self.assertEqual(kwargs["mt5_close_comment"], "[sl 4572.42]")

    def test_trade_record_from_dict_restores_exit_profile(self):
        trade = TradeRecord(
            1007, "BUY", 0.02, 4579.0, 4576.98, 4582.4, 1.5,
            strategy="M15_SCALP_DEEP", scalp=True, be_trigger=0.30, timeout=60,
            early_fail=0.20, features=self._scalp_features(),
        )
        trade.live_pnl = 0.50
        restored = TradeRecord.from_dict(trade.to_dict())

        self.assertEqual(restored.exit_profile, "scalp")
        self.assertEqual(restored.min_hold_seconds, 20)
        self.assertEqual(restored.reversal_arm_r, 1.0)
        self.assertEqual(restored.profit_lock_1_r, 0.12)
        self.assertEqual(restored.profit_lock_2_r, 0.22)
        self.assertEqual(restored.trail_lock_r, 0.40)

    def test_swing_engine_locks_profit_without_partial_close(self):
        manager = self._build_manager()
        manager._partial_close = Mock()
        trade = TradeRecord(
            2002, "BUY", 0.04, 4500.0, 4488.0, 4536.0, 12.0,
            strategy="SWING_ENGINE", scalp=False, features={"profile_name": "swing_engine"},
        )
        trade.live_pnl = 72.0  # 1.5R with 12pt stop and 0.04 lot
        h1_df = pd.DataFrame(
            {
                "high": [4505, 4508, 4510, 4514, 4518, 4522],
                "low": [4492, 4496, 4501, 4506, 4509, 4512],
                "close": [4501, 4505, 4508, 4512, 4516, 4520],
            }
        )

        manager._manage_swing_engine(trade, {"h1_df": h1_df})
        manager._partial_close.assert_not_called()
        manager.bridge.modify_trade.assert_called_once_with(2002, 4512.6, 4536.0)
        self.assertEqual(trade.sl, 4512.6)
        self.assertTrue(trade.trail_active)

        manager.bridge.modify_trade.reset_mock()
        manager._manage_swing_engine(trade, {"h1_df": h1_df})
        manager.bridge.modify_trade.assert_not_called()

    def test_swing_engine_profile_does_not_inherit_shared_timeout_or_breakeven(self):
        trade = TradeRecord(
            20021, "BUY", 0.04, 4500.0, 4488.0, 4536.0, 12.0,
            strategy="SWING_ENGINE", scalp=False, be_trigger=0.30, timeout=60,
            early_fail=0.20, features={"profile_name": "swing_engine"},
        )

        self.assertEqual(trade.exit_profile, "swing_engine")
        self.assertEqual(trade.timeout_seconds, 0)
        self.assertEqual(trade.be_trigger_r, 0.0)
        self.assertEqual(trade.early_fail_points, 0.0)
        self.assertFalse(trade.velocity_drop_enabled)

    def test_swing_engine_hard_profit_floor_closes_retraced_winner(self):
        manager = self._build_manager()
        manager._close_early = Mock()
        trade = TradeRecord(
            20022, "BUY", 0.04, 4500.0, 4488.0, 4536.0, 12.0,
            strategy="SWING_ENGINE", scalp=False, features={"profile_name": "swing_engine"},
        )
        trade.live_pnl = 57.6   # 1.2R
        trade.peak_pnl = 96.0   # 2.0R

        manager._manage_swing_engine(trade, {"h1_df": pd.DataFrame({"open": [1]*6, "high": [1]*6, "low": [1]*6, "close": [1]*6})})

        manager._close_early.assert_not_called()
        manager.bridge.modify_trade.assert_called_once_with(20022, 4516.8, 4536.0)

    def test_swing_engine_giveback_closes_after_large_retracement(self):
        manager = self._build_manager()
        manager._close_early = Mock()
        trade = TradeRecord(
            20023, "BUY", 0.04, 4500.0, 4488.0, 4536.0, 12.0,
            strategy="SWING_ENGINE", scalp=False, features={"profile_name": "swing_engine"},
        )
        trade.live_pnl = 76.4   # 1.592R
        trade.peak_pnl = 115.2  # 2.4R

        manager._manage_swing_engine(trade, {"h1_df": pd.DataFrame({"open": [1]*6, "high": [1]*6, "low": [1]*6, "close": [1]*6})})

        manager._close_early.assert_called_once()
        self.assertIn("giveback", manager._close_early.call_args[0][1].lower())

    def test_swing_engine_runner_mode_trails_to_m15_structure(self):
        manager = self._build_manager()
        trade = TradeRecord(
            20024, "SELL", 0.04, 4500.0, 4502.0, 4490.0, 2.0,
            strategy="SWING_ENGINE", scalp=False, features={"profile_name": "swing_engine"},
        )
        trade.live_pnl = 20.8   # 2.6R
        trade.peak_pnl = 20.8
        h1_df = pd.DataFrame(
            {
                "open": [4501, 4500, 4499, 4498, 4497, 4496],
                "high": [4502, 4501, 4500, 4499, 4498, 4497],
                "low": [4499, 4498, 4497, 4496, 4495, 4494],
                "close": [4500, 4499, 4498, 4497, 4496, 4495],
            }
        )
        m15_df = pd.DataFrame(
            {
                "open": [4498, 4497, 4496, 4495, 4494, 4493],
                "high": [4499, 4498, 4497, 4496, 4495, 4494],
                "low": [4496, 4495, 4494, 4493, 4492, 4491],
                "close": [4497, 4496, 4495, 4494, 4493, 4492],
            }
        )

        manager._manage_swing_engine(trade, {"h1_df": h1_df, "m15_df": m15_df})

        manager.bridge.modify_trade.assert_called_once_with(20024, 4496.0, 4490.0)
        self.assertEqual(trade.sl, 4496.0)
        self.assertTrue(trade.trail_active)

    def test_manage_trade_bypasses_universal_layer_for_isolated_engine_profiles(self):
        manager = self._build_manager()
        manager._apply_universal_management = Mock(return_value=True)
        manager._manage_swing_engine = Mock()

        swing_trade = TradeRecord(
            3002, "BUY", 0.04, 4500.0, 4488.0, 4536.0, 12.0,
            strategy="SWING_ENGINE", scalp=False, features={"profile_name": "swing_engine"},
        )

        manager._manage_trade(swing_trade, {"tick_count": 1}, {"h4_df": pd.DataFrame({"high": [1]*6, "low": [1]*6, "close": [1]*6})})

        manager._apply_universal_management.assert_not_called()
        manager._manage_swing_engine.assert_called_once()


if __name__ == "__main__":
    unittest.main()
