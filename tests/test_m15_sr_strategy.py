import unittest
from datetime import datetime, timedelta, timezone

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


if __name__ == "__main__":
    unittest.main()
