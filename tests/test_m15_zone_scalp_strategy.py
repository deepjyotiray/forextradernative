"""Unit tests for M15ZoneScalpStrategy zone geometry and reaction helpers."""
import unittest
from datetime import datetime, timezone

import pandas as pd

from engine.m15_zone_scalp_strategy import (
    M15ZoneScalpStrategy,
    _bearish_reaction,
    _bullish_reaction,
    _bullish_htf_ok,
    _bearish_htf_ok,
)
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


if __name__ == "__main__":
    unittest.main()
