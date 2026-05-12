import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd

import config as cfg
from engine.m15_sr_strategy import M15SupportResistanceStrategy


def _frame(rows, start_hour=0):
    start = datetime(2026, 4, 1, start_hour, 0, tzinfo=timezone.utc)
    payload = []
    for idx, (open_px, high_px, low_px, close_px, volume) in enumerate(rows):
        payload.append(
            {
                "datetime": start + timedelta(minutes=15 * idx),
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": close_px,
                "volume": volume,
            }
        )
    return pd.DataFrame(payload)


class M15SupportResistanceStrategyTests(unittest.TestCase):
    def setUp(self):
        self.strategy = M15SupportResistanceStrategy()

    def _context(
        self,
        *,
        h1="RANGE",
        m15="DOWN",
        bias="SHORT",
        session="ASIAN",
        score=0.0,
        tick_bias="NEUTRAL",
        macro_long=False,
        macro_short=True,
        regime="RANGING",
        ema20=4704.0,
        vwap=4702.0,
        range_mid=4705.0,
    ):
        return {
            "h1_direction": h1,
            "m15_direction": m15,
            "status_bias": bias,
            "session": session,
            "regime": regime,
            "tick_pressure": {"pressure_score": score, "directional_bias": tick_bias},
            "macro_long_supportive": macro_long,
            "macro_short_supportive": macro_short,
            "macro_support_status": "OPPOSING" if macro_long is False else "SUPPORTIVE",
            "ema20": ema20,
            "vwap": vwap,
            "range_mid": range_mid,
            "bias": {"m15_structure": {"pattern": "BEARISH", "bos_direction": "SHORT"}},
        }

    def test_zone_detection_clusters_multiple_support_touches(self):
        df = _frame(
            [
                (101.4, 101.8, 101.1, 101.6, 100),
                (101.2, 101.4, 100.1, 101.0, 110),
                (101.3, 101.7, 101.0, 101.5, 105),
                (101.1, 101.4, 100.2, 101.0, 108),
                (101.2, 101.8, 101.0, 101.6, 102),
                (101.0, 101.2, 100.1, 101.1, 115),
                (101.4, 101.9, 101.2, 101.8, 120),
                (101.7, 102.0, 101.4, 101.9, 118),
            ]
        )
        atr_m15 = self.strategy._atr_m15(df)
        zones = self.strategy._build_zones(df, atr_m15, "support", 2, current_price=100.25)
        self.assertTrue(zones)
        self.assertGreaterEqual(zones[0]["touches"], 2)
        self.assertEqual(zones[0]["type"], "support")

    def test_zone_invalidation_rejects_recent_close_break(self):
        df = _frame(
            [
                (101.4, 101.8, 101.1, 101.6, 100),
                (101.2, 101.4, 100.1, 101.0, 110),
                (101.3, 101.7, 101.0, 101.5, 105),
                (101.1, 101.4, 100.2, 101.0, 108),
                (101.2, 101.8, 101.0, 101.6, 102),
                (100.9, 101.1, 99.6, 99.7, 115),
                (100.0, 100.3, 99.4, 99.5, 120),
                (99.8, 100.2, 99.3, 99.4, 118),
            ]
        )
        atr_m15 = self.strategy._atr_m15(df)
        zones = self.strategy._build_zones(df, atr_m15, "support", 2, current_price=99.5)
        self.assertEqual(zones, [])

    def test_buy_rejection_setup_detected(self):
        candle = pd.Series(
            {
                "open": 100.30,
                "high": 100.90,
                "low": 99.80,
                "close": 100.75,
            }
        )
        zone = {"zone_low": 100.00, "zone_high": 100.40, "zone_mid": 100.20}
        self.assertTrue(self.strategy._is_buy_rejection(candle, zone))

    def test_sell_rejection_setup_detected(self):
        candle = pd.Series(
            {
                "open": 100.80,
                "high": 101.35,
                "low": 100.20,
                "close": 100.35,
            }
        )
        zone = {"zone_low": 100.70, "zone_high": 101.00, "zone_mid": 100.85}
        self.assertTrue(self.strategy._is_sell_rejection(candle, zone))

    def test_spike_manipulation_block_activates(self):
        df = _frame(
            [
                (100.2, 100.6, 99.9, 100.3, 100),
                (100.3, 100.7, 100.0, 100.4, 105),
                (100.4, 100.8, 100.1, 100.5, 98),
                (100.5, 100.9, 100.2, 100.6, 102),
                (100.6, 101.0, 100.3, 100.7, 101),
                (100.7, 101.1, 100.4, 100.8, 99),
                (100.8, 101.2, 100.5, 100.9, 104),
                (100.9, 103.6, 100.1, 101.0, 260),
            ]
        )
        atr_m15 = self.strategy._atr_m15(df)
        blocked, reason = self.strategy._check_spike_block(
            closed=df,
            atr_m15=atr_m15,
            now_utc=df.iloc[-1]["datetime"].to_pydatetime(),
        )
        self.assertFalse(blocked)
        self.assertIn("Spike/manipulation block active", reason)

    def test_rr_calculation_prefers_opposite_zone_when_valid(self):
        plan = self.strategy._build_trade_plan(
            direction="BUY",
            zone={"zone_low": 100.00, "zone_high": 100.20, "zone_mid": 100.10},
            opposite_zones=[{"zone_id": "RES_1", "zone_mid": 101.40}],
            tick={"bid": 100.35, "ask": 100.37},
            atr_m15=0.40,
        )
        self.assertIsNotNone(plan)
        self.assertGreaterEqual(plan["rr"], 1.3)
        self.assertEqual(plan["_target_source"], "opposite_zone")

    def test_buy_sweep_reclaim_uses_sweep_low_for_stop_anchor(self):
        closed = _frame(
            [
                (100.25, 100.40, 99.88, 100.16, 120),
                (100.14, 100.58, 100.06, 100.42, 125),
            ]
        )
        zone = {"zone_low": 100.00, "zone_high": 100.20, "zone_mid": 100.10}
        sweep = self.strategy._recent_sweep_context(closed, zone, atr_m15=0.40, direction="BUY")
        self.assertIsNotNone(sweep)
        self.assertTrue(sweep["reclaimed"])
        plan = self.strategy._build_trade_plan(
            direction="BUY",
            zone=zone,
            opposite_zones=[{"zone_id": "RES_1", "zone_mid": 101.60}],
            tick={"bid": 100.41, "ask": 100.43},
            atr_m15=0.40,
            sweep_context=sweep,
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan["_sl_anchor"], 99.88)
        self.assertEqual(plan["_sweep_extreme"], 99.88)

    def test_sell_sweep_reclaim_uses_sweep_high_for_stop_anchor(self):
        closed = _frame(
            [
                (100.80, 101.12, 100.55, 100.92, 120),
                (100.90, 100.95, 100.32, 100.48, 128),
            ]
        )
        zone = {"zone_low": 100.70, "zone_high": 101.00, "zone_mid": 100.85}
        sweep = self.strategy._recent_sweep_context(closed, zone, atr_m15=0.40, direction="SELL")
        self.assertIsNotNone(sweep)
        self.assertTrue(sweep["reclaimed"])
        plan = self.strategy._build_trade_plan(
            direction="SELL",
            zone=zone,
            opposite_zones=[{"zone_id": "SUP_1", "zone_mid": 99.30}],
            tick={"bid": 100.47, "ask": 100.49},
            atr_m15=0.40,
            sweep_context=sweep,
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan["_sl_anchor"], 101.12)
        self.assertEqual(plan["_sweep_extreme"], 101.12)

    def test_prior_support_sweep_requires_strong_bullish_reclaim_confirmation(self):
        closed = _frame(
            [
                (100.25, 100.42, 99.88, 100.14, 120),
                (100.12, 100.18, 100.00, 100.05, 118),
            ]
        )
        zone = {
            "zone_id": "SUP_6H_100.10",
            "type": "support",
            "zone_low": 100.00,
            "zone_high": 100.20,
            "zone_mid": 100.10,
            "touches": 3,
            "distance": 0.03,
            "max_distance_abs": 0.40,
            "lookback_hours": 6,
            "single_extreme": False,
            "spike_based": False,
            "random_cross": False,
        }
        outcome = self.strategy._evaluate_directional_setup(
            direction="BUY",
            active_zone=zone,
            opposite_zones=[{"zone_id": "RES_6H_101.20", "zone_mid": 101.20}],
            closed=closed,
            tick={"bid": 100.06, "ask": 100.08, "spread": 0.02},
            current_price=100.07,
            atr_m15=0.40,
            positions=[],
            now_utc=closed.iloc[-1]["datetime"].to_pydatetime(),
            filter_results={
                "spread": {"passed": True, "reason": "OK", "current": 0.02},
                "manipulation": {"passed": True, "reason": "OK"},
                "zone_quality": {"passed": True, "reason": "OK"},
            },
            nearest_support=zone,
            nearest_resistance={"zone_id": "RES_6H_101.20", "zone_low": 101.00, "zone_high": 101.30, "zone_mid": 101.20, "touches": 2},
        )
        self.assertEqual(outcome["signal"], "NO_TRADE")
        self.assertEqual(outcome["reason"], "Post-sweep candle did not confirm BUY")

    def test_directional_setup_uses_passed_tick_pressure_without_name_error(self):
        closed = _frame(
            [
                (100.30, 100.50, 100.10, 100.22, 110),
                (100.18, 100.60, 99.92, 100.48, 135),
            ]
        )
        zone = {
            "zone_id": "SUP_6H_100.10",
            "type": "support",
            "zone_low": 100.00,
            "zone_high": 100.20,
            "zone_mid": 100.10,
            "touches": 3,
            "distance": 0.02,
            "max_distance_abs": 0.40,
            "lookback_hours": 6,
            "single_extreme": False,
            "spike_based": False,
            "random_cross": False,
        }
        outcome = self.strategy._evaluate_directional_setup(
            direction="BUY",
            active_zone=zone,
            opposite_zones=[{"zone_id": "RES_6H_101.20", "zone_mid": 101.20}],
            closed=closed,
            tick={"bid": 100.47, "ask": 100.49, "spread": 0.02},
            current_price=100.48,
            atr_m15=0.40,
            positions=[],
            now_utc=closed.iloc[-1]["datetime"].to_pydatetime(),
            filter_results={
                "spread": {"passed": True, "reason": "OK", "current": 0.02},
                "manipulation": {"passed": True, "reason": "OK"},
                "tick_pressure": {"passed": True, "reason": "OK"},
                "zone_quality": {"passed": True, "reason": "OK"},
            },
            nearest_support=zone,
            nearest_resistance={"zone_id": "RES_6H_101.20", "zone_low": 101.00, "zone_high": 101.30, "zone_mid": 101.20, "touches": 2},
            tick_pressure={"pressure_score": 0.18, "directional_bias": "LONG"},
        )
        self.assertIn(outcome["signal"], ("BUY", "NO_TRADE"))

    def test_cooldown_logic_blocks_before_required_candles(self):
        last_trade = datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)
        self.strategy._last_trade_candle_by_direction["BUY"] = last_trade
        blocked = self.strategy._cooldown_reason("BUY", last_trade + timedelta(minutes=30))
        clear = self.strategy._cooldown_reason("BUY", last_trade + timedelta(minutes=45))
        self.assertIn("Cooldown active", blocked)
        self.assertIsNone(clear)

    def test_extended_lookback_finds_older_zone_when_6h_window_has_none(self):
        original_extended = cfg.M15_SR_EXTENDED_LOOKBACK_HOURS
        try:
            cfg.M15_SR_EXTENDED_LOOKBACK_HOURS = 7
            rows = []
            price = 100.0
            for idx in range(28):
                open_px = price
                close_px = price + (0.02 if idx % 2 == 0 else -0.01)
                high_px = max(open_px, close_px) + 0.25
                low_px = min(open_px, close_px) - 0.20
                volume = 100 + idx
                rows.append((open_px, high_px, low_px, close_px, volume))
                price = close_px
            rows[1] = (100.05, 100.95, 99.95, 100.10, 110)
            rows[3] = (100.08, 100.92, 99.98, 100.12, 112)
            df = _frame(rows)
            atr_m15 = self.strategy._atr_m15(df)
            support, resistance, lookback_used = self.strategy._detect_nearby_zones(
                closed=df,
                atr_m15=atr_m15,
                current_price=100.90,
            )
            self.assertEqual(lookback_used, 7)
            self.assertFalse(support)
            self.assertTrue(resistance)
        finally:
            cfg.M15_SR_EXTENDED_LOOKBACK_HOURS = original_extended


    def test_choppy_market_not_triggered_by_single_wide_candle(self):
        """A single candle spanning both zones should NOT be flagged as choppy."""
        support_zone = {"zone_low": 100.00, "zone_high": 100.20, "zone_mid": 100.10}
        resistance_zone = {"zone_low": 101.80, "zone_high": 102.00, "zone_mid": 101.90}
        # One wide candle that touches both zones simultaneously
        df = _frame([
            (100.50, 100.60, 100.40, 100.55, 100),
            (100.55, 100.65, 100.45, 100.60, 100),
            (100.10, 102.00, 99.95, 101.50, 200),  # wide candle touching both
        ])
        self.assertFalse(self.strategy._is_choppy_market(df, support_zone, resistance_zone))

    def test_choppy_market_triggered_by_different_candles_touching_each_zone(self):
        """Touches on different candles — one at support, one at resistance — is genuinely choppy."""
        support_zone = {"zone_low": 100.00, "zone_high": 100.20, "zone_mid": 100.10}
        resistance_zone = {"zone_low": 101.80, "zone_high": 102.00, "zone_mid": 101.90}
        df = _frame([
            (100.50, 100.60, 100.40, 100.55, 100),
            (100.55, 102.00, 101.70, 101.90, 110),  # touches resistance only
            (101.80, 101.95, 100.05, 100.15, 120),  # touches support only
        ])
        self.assertTrue(self.strategy._is_choppy_market(df, support_zone, resistance_zone))


    def test_random_cross_not_flagged_when_last_close_above_support_zone(self):
        """Support zone: last close above zone_high means price has bounced — not a random cross."""
        # Older closes dipped below, but last close is cleanly above — valid support bounce
        closes = [100.50, 99.80, 100.10, 100.55, 100.70]
        self.assertFalse(M15SupportResistanceStrategy._has_random_closes(closes, 100.00, 100.20, "support"))

    def test_random_cross_not_flagged_when_last_close_below_resistance_zone(self):
        """Resistance zone: last close below zone_low means price has rejected — not a random cross."""
        closes = [100.50, 101.30, 100.90, 100.60, 100.40]
        self.assertFalse(M15SupportResistanceStrategy._has_random_closes(closes, 101.00, 101.20, "resistance"))

    def test_random_cross_flagged_when_recent_closes_straddle_zone_indecisively(self):
        """Recent closes alternating above and below with last close inside zone — genuine random cross."""
        # Last close is inside the zone (no clear directional bias)
        closes = [100.50, 101.30, 100.90, 101.25, 101.05]
        self.assertTrue(M15SupportResistanceStrategy._has_random_closes(closes, 101.00, 101.10, "resistance"))

    def test_random_cross_only_uses_last_3_closes_not_older_history(self):
        """Old closes that straddle the zone should not flag it if the 3 most recent are clean."""
        # First 2 closes straddle the zone, but last 3 are all cleanly above
        closes = [99.80, 101.50, 100.60, 100.65, 100.70]
        self.assertFalse(M15SupportResistanceStrategy._has_random_closes(closes, 100.00, 100.20, "support"))

    def test_trade_8563247461_style_setup_is_blocked(self):
        candle = pd.Series(
            {
                "datetime": datetime(2026, 5, 11, 2, 0, tzinfo=timezone.utc),
                "open": 4685.69,
                "high": 4693.07,
                "low": 4678.40,
                "close": 4685.94,
                "volume": 5783,
            }
        )
        zone = {
            "zone_id": "SUPPORT_6H_4676.89",
            "type": "support",
            "zone_low": 4675.19,
            "zone_high": 4678.59,
            "zone_mid": 4676.89,
            "touches": 2,
        }
        outcome = self.strategy.evaluate_rejection_setup(
            direction="BUY",
            zone=zone,
            candle=candle,
            tick={"bid": 4686.00, "ask": 4686.21, "spread": 0.21},
            atr_m15=13.5678,
            market_context=self._context(score=0.037, tick_bias="LONG"),
            opposite_zones=[{"zone_id": "RES_1", "zone_mid": 4709.87}],
            now_utc=datetime(2026, 5, 11, 2, 15, tzinfo=timezone.utc),
        )
        self.assertEqual(outcome["signal"], "NO_TRADE")
        self.assertIn("COUNTER_BIAS_BUY_NEEDS_STRONG_LONG_TICK_PRESSURE", outcome["reason"])
        self.assertIn("COUNTER_BIAS_BUY_CANDLE_NOT_A_PLUS", outcome["reason"])

        stale_signal = {
            "signal": "BUY",
            "_strategy_name": self.strategy.name,
            "_signal_id": "sig-8563247461",
            "_zone_id": zone["zone_id"],
            "_decision_time": "2026-05-11T02:15:04+00:00",
            "_ttl_seconds": 10,
            "_atr14": 13.5678,
            "_min_rr_required": 0.8,
            "_max_spread": 0.45,
            "_max_entry_drift_atr": 0.10,
            "_context_hash": "x",
            "_context_classification": "MEAN_REVERSION_BOUNCE_ONLY",
            "_status_bias": "SHORT",
            "entry": 4686.21,
            "sl": 4670.44,
            "tp": 4700.00,
            "confidence": 0.62,
            "_signal_family": self.strategy.FAMILY_BOUNCE,
        }
        pre_send = self.strategy.pre_send_revalidate(
            stale_signal,
            {
                "now_utc": datetime(2026, 5, 11, 2, 17, 51, tzinfo=timezone.utc),
                "tick": {"bid": 4685.00, "ask": 4685.21, "spread": 0.21},
                "tick_pressure": {"pressure_score": 0.037, "directional_bias": "LONG"},
                "bias": {"direction": "SHORT"},
                "regime": {"state": "RANGING"},
                "market_state": {"trend": {"H1": "RANGE", "M15": "DOWN"}, "structure": {"H1": "RANGE", "H4": "DOWN", "D1": "DOWN"}},
            },
        )
        self.assertFalse(pre_send["allowed"])
        self.assertEqual(pre_send["reason"], "STALE_SIGNAL")

    def test_trade_8563943603_style_setup_is_bounce_only(self):
        candle = pd.Series(
            {
                "datetime": datetime(2026, 5, 11, 3, 0, tzinfo=timezone.utc),
                "open": 4681.69,
                "high": 4689.70,
                "low": 4675.09,
                "close": 4687.15,
                "volume": 4682,
            }
        )
        zone = {
            "zone_id": "SUPPORT_6H_4677.39",
            "type": "support",
            "zone_low": 4675.73,
            "zone_high": 4679.06,
            "zone_mid": 4677.39,
            "touches": 3,
        }
        outcome = self.strategy.evaluate_rejection_setup(
            direction="BUY",
            zone=zone,
            candle=candle,
            tick={"bid": 4687.21, "ask": 4687.42, "spread": 0.21},
            atr_m15=13.3321,
            market_context=self._context(score=0.375, tick_bias="LONG"),
            opposite_zones=[{"zone_id": "RES_1", "zone_mid": 4705.00}],
            now_utc=datetime(2026, 5, 11, 3, 15, tzinfo=timezone.utc),
        )
        self.assertEqual(outcome["signal"], "BUY")
        self.assertEqual(outcome["_signal_family"], self.strategy.FAMILY_BOUNCE)
        self.assertEqual(outcome["_exit_profile"], "m15_mean_reversion_fast")
        self.assertNotEqual(outcome["_exit_profile"], "swing_structured")

    def test_asian_counter_bias_can_use_normal_rejection_when_a_plus_disabled(self):
        candle = pd.Series(
            {
                "datetime": datetime(2026, 5, 11, 3, 0, tzinfo=timezone.utc),
                "open": 99.95,
                "high": 100.95,
                "low": 99.20,
                "close": 100.43,
                "volume": 4200,
            }
        )
        zone = {
            "zone_id": "SUPPORT_6H_100.00",
            "type": "support",
            "zone_low": 99.70,
            "zone_high": 100.00,
            "zone_mid": 99.85,
            "touches": 3,
        }
        with patch.object(cfg, "M15_SR_ASIAN_COUNTER_BIAS_REQUIRE_A_PLUS", False):
            outcome = self.strategy.evaluate_rejection_setup(
                direction="BUY",
                zone=zone,
                candle=candle,
                tick={"bid": 100.44, "ask": 100.65, "spread": 0.21},
                atr_m15=10.0,
                market_context=self._context(score=0.35, tick_bias="LONG"),
                opposite_zones=[{"zone_id": "RES_1", "zone_mid": 118.00}],
                now_utc=datetime(2026, 5, 11, 3, 15, tzinfo=timezone.utc),
            )

        self.assertEqual(outcome["signal"], "BUY")
        self.assertEqual(outcome["_signal_family"], self.strategy.FAMILY_BOUNCE)

    def test_trade_8564640195_style_setup_is_blocked(self):
        candle = pd.Series(
            {
                "datetime": datetime(2026, 5, 11, 4, 0, tzinfo=timezone.utc),
                "open": 4678.71,
                "high": 4679.91,
                "low": 4673.47,
                "close": 4677.55,
                "volume": 3555,
            }
        )
        zone = {
            "zone_id": "SUPPORT_6H_4675.42",
            "type": "support",
            "zone_low": 4674.04,
            "zone_high": 4676.80,
            "zone_mid": 4675.42,
            "touches": 2,
        }
        outcome = self.strategy.evaluate_rejection_setup(
            direction="BUY",
            zone=zone,
            candle=candle,
            tick={"bid": 4677.58, "ask": 4677.79, "spread": 0.21},
            atr_m15=11.0103,
            market_context=self._context(score=-0.167, tick_bias="SHORT"),
            opposite_zones=[{"zone_id": "RES_1", "zone_mid": 4690.05}],
            now_utc=datetime(2026, 5, 11, 4, 15, tzinfo=timezone.utc),
        )
        self.assertEqual(outcome["signal"], "NO_TRADE")
        self.assertIn("COUNTER_BIAS_BUY_TICK_NEGATIVE", outcome["reason"])
        self.assertIn("COUNTER_BIAS_BUY_TICK_BIAS_SHORT", outcome["reason"])
        self.assertIn("COUNTER_BIAS_BUY_CANDLE_NOT_A_PLUS", outcome["reason"])

    def test_counter_bias_sell_requires_strong_short_pressure(self):
        candle = pd.Series(
            {
                "datetime": datetime(2026, 5, 11, 5, 0, tzinfo=timezone.utc),
                "open": 4712.10,
                "high": 4716.40,
                "low": 4708.80,
                "close": 4710.30,
                "volume": 4100,
            }
        )
        zone = {
            "zone_id": "RESISTANCE_6H_4714.20",
            "type": "resistance",
            "zone_low": 4712.80,
            "zone_high": 4715.60,
            "zone_mid": 4714.20,
            "touches": 2,
        }
        outcome = self.strategy.evaluate_rejection_setup(
            direction="SELL",
            zone=zone,
            candle=candle,
            tick={"bid": 4710.20, "ask": 4710.41, "spread": 0.21},
            atr_m15=10.50,
            market_context=self._context(h1="RANGE", m15="UP", bias="LONG", score=0.12, tick_bias="LONG", macro_long=True, macro_short=False),
            opposite_zones=[{"zone_id": "SUP_1", "zone_mid": 4698.0}],
            now_utc=datetime(2026, 5, 11, 5, 15, tzinfo=timezone.utc),
        )
        self.assertEqual(outcome["signal"], "NO_TRADE")
        self.assertIn("COUNTER_BIAS_RESISTANCE_NOT_STRONG_ENOUGH", outcome["reason"])
        self.assertIn("COUNTER_BIAS_SELL_TICK_POSITIVE", outcome["reason"])
        self.assertIn("COUNTER_BIAS_SELL_TICK_BIAS_LONG", outcome["reason"])


if __name__ == "__main__":
    unittest.main()
