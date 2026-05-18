"""Unit tests for M15 zone scalp strategies."""
import unittest
from datetime import datetime, timezone

import pandas as pd
from unittest.mock import patch

from engine.m15_zone_scalp_strategy import (
    M15ZoneScalpStrategy,
    _bearish_reaction,
    _bullish_reaction,
    _bullish_htf_ok,
    _bearish_htf_ok,
)
from engine.m15_zone_scalp_inverse_strategy import M15ZoneScalpInverseStrategy
from engine.master_trade_gate import _family_bucket
from config import get_exit_profile_config


class M15ZoneScalpLogicTests(unittest.TestCase):
    def test_pick_demand_requires_bid_ge_zone_low_and_touch(self):
        strat = M15ZoneScalpStrategy()
        clusters = [{"zone_low": 100.0, "zone_high": 101.0, "zone_mid": 100.5, "strength": 0.5}]
        self.assertIsNone(strat._pick_demand_zone(99.5, clusters, proximity=2.0, min_str=0.1))
        self.assertIsNone(strat._pick_demand_zone(100.25, clusters, proximity=2.0, min_str=0.1))
        self.assertIsNotNone(strat._pick_demand_zone(100.75, clusters, proximity=2.0, min_str=0.1))

    def test_pick_supply_requires_bid_le_zone_high_and_touch(self):
        strat = M15ZoneScalpStrategy()
        clusters = [{"zone_low": 100.0, "zone_high": 101.0, "zone_mid": 100.5, "strength": 0.5}]
        self.assertIsNone(strat._pick_supply_zone(101.5, clusters, proximity=2.0, min_str=0.1))
        self.assertIsNone(strat._pick_supply_zone(100.75, clusters, proximity=2.0, min_str=0.1))
        self.assertIsNotNone(strat._pick_supply_zone(100.25, clusters, proximity=2.0, min_str=0.1))

    def test_bullish_reaction_rejection_or_displacement(self):
        rej = pd.Series({"open": 100.0, "high": 100.8, "low": 99.0, "close": 100.5})
        self.assertTrue(_bullish_reaction(rej, 0.30, 0.52, 0.62))
        disp = pd.Series({"open": 100.0, "high": 101.5, "low": 99.9, "close": 101.35})
        self.assertTrue(_bullish_reaction(disp, 0.32, 0.52, 0.62))

    def test_bearish_reaction(self):
        rej = pd.Series({"open": 101.0, "high": 102.0, "low": 100.2, "close": 100.4})
        self.assertTrue(_bearish_reaction(rej, 0.30, 0.52, 0.62))

    def test_htf_fallback_when_ms_ambiguous(self):
        ms = {"trend": {"D1": "RANGE", "H4": "RANGE"}, "structure": {"D1": "RANGE", "H4": "RANGE"}}
        data = {"market_state": ms, "bias": {"direction": "LONG", "confidence": 0.5}}
        s_cfg = {"require_htf_bias": True, "bias_fallback_min_confidence": 0.35}
        self.assertTrue(_bullish_htf_ok(data, s_cfg))
        data_short = {"market_state": ms, "bias": {"direction": "SHORT", "confidence": 0.5}}
        self.assertTrue(_bearish_htf_ok(data_short, s_cfg))

    def test_m15_zone_family_buckets_as_m15(self):
        self.assertEqual(_family_bucket("M15_ZONE"), "M15")
        self.assertEqual(_family_bucket("M15_ZONE_SCALP"), "M15")
        self.assertEqual(_family_bucket("M15"), "M15")

    def test_m15_zone_scalp_exit_profile_is_defined(self):
        profile = get_exit_profile_config("m15_zone_scalp")
        self.assertEqual(profile["profile_name"], "m15_zone_scalp")
        self.assertGreater(profile["be_trigger_r"], 0)

    def test_inverse_strategy_flips_valid_base_signal(self):
        strat = M15ZoneScalpInverseStrategy()
        data = {
            "tick": {"bid": 100.0, "ask": 100.2},
            "account": {"balance": 10000.0},
        }

        with patch(
            "engine.m15_zone_scalp_strategy.M15ZoneScalpStrategy.generate_signal",
            return_value={
                "signal": "BUY",
                "entry": 100.0,
                "sl": 98.0,
                "tp": 104.0,
                "confidence": 0.8,
                "reason": "BUY M15 zone scalp | demand touch",
                "_bias_direction": "LONG",
                "_body_ratio": 0.75,
                "_exit_profile": "m15_zone_scalp",
                "_scalp": True,
                "_m15_zone_confirmed": True,
                "_candle_confirmation": True,
                "_zone_mid": 99.5,
                "_pip_size": 0.1,
                "_session": "LONDON",
            },
        ):
            signal = strat.generate_signal(data)

        self.assertEqual(signal["signal"], "SELL")
        self.assertEqual(signal["sl"], 104.0)
        self.assertEqual(signal["tp"], 98.0)
        self.assertEqual(signal["entry"], 100.0)
        self.assertEqual(signal["_strategy_name"], "M15_ZONE_SCALP_INVERSE")
        self.assertEqual(signal["_source_strategy_name"], "M15_ZONE_SCALP")
        self.assertEqual(signal["_source_signal"], "BUY")
        self.assertEqual(signal["_bias_direction"], "LONG")
        self.assertGreater(signal["rr"], 0)


if __name__ == "__main__":
    unittest.main()
