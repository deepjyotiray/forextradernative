"""Unit tests for M15 zone scalp strategies."""
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

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
from engine.m15_zone_micro_bias import resolve_m15_zone_micro_bias
from engine.master_trade_gate import _family_bucket
from config import get_exit_profile_config


def _micro_bias_context(
    *,
    d1="UP",
    h4="DOWN",
    h1="RANGE",
    tick_bias="LONG",
    pressure_score=0.41,
    favorable_long=True,
    favorable_short=False,
    body_ratio=0.61,
    bullish_reaction=False,
    bearish_reaction=True,
    zone_type="SUPPLY",
    zone_low=100.0,
    zone_mid=100.5,
    zone_high=101.0,
    bid=101.2,
    ask=101.4,
    m15_rows=None,
):
    if m15_rows is None:
        m15_rows = [
            {"open": 99.9, "high": 100.4, "low": 99.7, "close": 100.2},
            {"open": 100.2, "high": 100.6, "low": 100.0, "close": 100.45},
            {"open": 100.45, "high": 100.8, "low": 100.2, "close": 100.7},
            {"open": 100.7, "high": 101.0, "low": 100.5, "close": 100.9},
            {"open": 100.9, "high": 101.3, "low": 100.8, "close": 101.15},
        ]
    return {
        "data": {
            "tick": {"bid": bid, "ask": ask},
            "market_state": {
                "trend": {"D1": d1, "H4": h4, "H1": h1},
                "structure": {"D1": d1, "H4": h4, "H1": h1},
            },
            "tick_pressure": {
                "pressure_score": pressure_score,
                "directional_bias": tick_bias,
                "favorable_long": favorable_long,
                "favorable_short": favorable_short,
                "burst_rate": 5.2,
            },
            "indicators": {"atr_ratio": 1.05},
            "regime": {"atr_ratio": 1.05},
            "m15_df": pd.DataFrame(m15_rows),
        },
        "signal": {
            "_body_ratio": body_ratio,
            "_bullish_reaction": bullish_reaction,
            "_bearish_reaction": bearish_reaction,
            "_zone_low": zone_low,
            "_zone_mid": zone_mid,
            "_zone_high": zone_high,
            "_zone_type": zone_type,
        },
    }


def _candidate(signal="SELL"):
    direction = str(signal).upper()
    is_buy = direction == "BUY"
    return {
        "signal": direction,
        "entry": 100.0,
        "sl": 98.0 if is_buy else 102.0,
        "tp": 102.5 if is_buy else 97.5,
        "tp_levels": [102.5 if is_buy else 97.5],
        "confidence": 0.8,
        "rr": 1.5,
        "_zone_strength": 0.6,
        "_zone_type": "DEMAND" if is_buy else "SUPPLY",
        "_body_ratio": 0.61 if not is_buy else 0.74,
        "_bullish_reaction": is_buy,
        "_bearish_reaction": not is_buy,
        "_zone_low": 99.5 if is_buy else 100.0,
        "_zone_mid": 100.0 if is_buy else 100.5,
        "_zone_high": 100.5 if is_buy else 101.0,
        "_signal_family": "M15",
        "_setup_direction": "LONG" if is_buy else "SHORT",
        "_bias_direction": "LONG" if is_buy else "SHORT",
        "_exit_profile": "m15_zone_scalp",
        "_scalp": True,
        "_tp_levels": [102.5 if is_buy else 97.5],
        "_pip_size": 0.1,
        "_session": "LONDON",
    }


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

    def test_resolver_routes_conflicted_sell_to_inverse_buy_on_long_pressure(self):
        resolver = resolve_m15_zone_micro_bias(
            _micro_bias_context(),
            setup_direction="SELL",
            zone_type="SUPPLY",
        )

        self.assertEqual(resolver["micro_bias_direction"], "LONG")
        self.assertEqual(resolver["recommended_action"], "ROUTE_TO_INVERSE")
        self.assertEqual(resolver["final_direction"], "BUY")
        self.assertEqual(resolver["micro_bias_confidence"], "HIGH")

    def test_resolver_allows_sell_when_bearish_micro_confirmation_is_strong(self):
        resolver = resolve_m15_zone_micro_bias(
            _micro_bias_context(
                d1="DOWN",
                h4="DOWN",
                h1="DOWN",
                tick_bias="SHORT",
                pressure_score=-0.34,
                favorable_long=False,
                favorable_short=True,
                body_ratio=0.93,
                bullish_reaction=False,
                bearish_reaction=True,
                bid=99.7,
                ask=99.9,
                m15_rows=[
                    {"open": 101.4, "high": 101.5, "low": 101.0, "close": 101.1},
                    {"open": 101.1, "high": 101.2, "low": 100.7, "close": 100.8},
                    {"open": 100.8, "high": 100.9, "low": 100.3, "close": 100.4},
                    {"open": 100.4, "high": 100.5, "low": 99.9, "close": 100.0},
                    {"open": 100.0, "high": 100.1, "low": 99.5, "close": 99.7},
                ],
            ),
            setup_direction="SELL",
            zone_type="SUPPLY",
        )

        self.assertEqual(resolver["micro_bias_direction"], "SHORT")
        self.assertEqual(resolver["recommended_action"], "ALLOW_BASE_TRADE")
        self.assertEqual(resolver["final_direction"], "SELL")

    def test_resolver_allows_buy_in_d1_up_h4_down_when_micro_bias_is_long(self):
        resolver = resolve_m15_zone_micro_bias(
            _micro_bias_context(
                d1="UP",
                h4="DOWN",
                h1="RANGE",
                tick_bias="LONG",
                pressure_score=0.22,
                favorable_long=True,
                favorable_short=False,
                body_ratio=0.74,
                bullish_reaction=True,
                bearish_reaction=False,
                zone_type="DEMAND",
                zone_low=99.0,
                zone_mid=99.5,
                zone_high=100.0,
                bid=100.2,
                ask=100.4,
                m15_rows=[
                    {"open": 98.8, "high": 99.3, "low": 98.7, "close": 99.1},
                    {"open": 99.1, "high": 99.5, "low": 99.0, "close": 99.35},
                    {"open": 99.35, "high": 99.8, "low": 99.2, "close": 99.65},
                    {"open": 99.65, "high": 100.0, "low": 99.5, "close": 99.9},
                    {"open": 99.9, "high": 100.3, "low": 99.8, "close": 100.15},
                ],
            ),
            setup_direction="BUY",
            zone_type="DEMAND",
        )

        self.assertEqual(resolver["micro_bias_direction"], "LONG")
        self.assertEqual(resolver["recommended_action"], "ALLOW_BASE_TRADE")
        self.assertEqual(resolver["final_direction"], "BUY")

    def test_resolver_blocks_medium_bias_when_score_floor_is_higher(self):
        cfg_module = SimpleNamespace(
            M15_ZONE_MICRO_BIAS_MIN_CONFIDENCE="MEDIUM",
            M15_ZONE_MICRO_BIAS_MIN_SCORE=60.0,
            M15_ZONE_ROUTE_AGAINST_BIAS_TO_INVERSE=True,
            M15_ZONE_BLOCK_NEUTRAL_BIAS=True,
            M15_ZONE_SELL_REQUIRE_SHORT_PRESSURE=True,
            M15_ZONE_SELL_MIN_NEGATIVE_PRESSURE_SCORE=-0.15,
            M15_ZONE_SELL_MIN_BODY_RATIO_IF_D1_UP=0.85,
            M15_ZONE_SELL_BLOCK_IF_PRESSURE_LONG=True,
            M15_ZONE_BUY_REQUIRE_LONG_PRESSURE=False,
            M15_ZONE_BUY_MIN_BODY_RATIO_IF_D1_DOWN=0.85,
            M15_ZONE_BUY_BLOCK_IF_PRESSURE_SHORT=True,
        )
        resolver = resolve_m15_zone_micro_bias(
            _micro_bias_context(
                d1="UP",
                h4="DOWN",
                h1="RANGE",
                tick_bias="LONG",
                pressure_score=0.05,
                favorable_long=False,
                favorable_short=False,
                body_ratio=0.45,
                bullish_reaction=False,
                bearish_reaction=False,
                zone_type="DEMAND",
                zone_low=99.0,
                zone_mid=99.5,
                zone_high=100.0,
                bid=100.2,
                ask=100.4,
                m15_rows=[
                    {"open": 99.8, "high": 100.1, "low": 99.7, "close": 100.0},
                    {"open": 100.0, "high": 100.2, "low": 99.9, "close": 100.05},
                    {"open": 100.05, "high": 100.15, "low": 99.95, "close": 100.00},
                    {"open": 100.00, "high": 100.10, "low": 99.90, "close": 100.02},
                    {"open": 100.02, "high": 100.12, "low": 99.92, "close": 100.01},
                ],
            ),
            setup_direction="BUY",
            zone_type="DEMAND",
            cfg_module=cfg_module,
        )

        self.assertEqual(resolver["micro_bias_direction"], "LONG")
        self.assertEqual(resolver["micro_bias_confidence"], "MEDIUM")
        self.assertEqual(resolver["recommended_action"], "BLOCK")
        self.assertEqual(resolver["final_direction"], "NONE")
        self.assertIn("blocked: score below configured minimum", resolver["reasons"])

    def test_base_strategy_blocks_conflicted_sell_and_inverse_routes_it(self):
        data = _micro_bias_context()["data"]
        candidate = _candidate("SELL")

        with patch.object(M15ZoneScalpStrategy, "_family_candidates", return_value=([candidate], None)):
            base_signal = M15ZoneScalpStrategy().generate_signal(data)
        with patch.object(M15ZoneScalpInverseStrategy, "_family_candidates", return_value=([candidate], None)):
            inverse_signal = M15ZoneScalpInverseStrategy().generate_signal(data)

        self.assertEqual(base_signal["signal"], "NO_TRADE")
        self.assertEqual(base_signal["_micro_bias_action"], "ROUTE_TO_INVERSE")
        self.assertEqual(base_signal["_route_type"], "BLOCKED_CONFLICTING_BIAS")
        self.assertIn("action=ROUTE_TO_INVERSE", base_signal["reason"])

        self.assertEqual(inverse_signal["signal"], "BUY")
        self.assertEqual(inverse_signal["_route_type"], "ROUTED_TO_INVERSE")
        self.assertEqual(inverse_signal["_source_signal"], "SELL")
        self.assertEqual(inverse_signal["_micro_bias_action"], "ROUTE_TO_INVERSE")

    def test_inverse_strategy_stays_blocked_when_base_trade_is_supported(self):
        data = _micro_bias_context(
            d1="DOWN",
            h4="DOWN",
            h1="DOWN",
            tick_bias="SHORT",
            pressure_score=-0.34,
            favorable_long=False,
            favorable_short=True,
            body_ratio=0.93,
            bullish_reaction=False,
            bearish_reaction=True,
            bid=99.7,
            ask=99.9,
            m15_rows=[
                {"open": 101.4, "high": 101.5, "low": 101.0, "close": 101.1},
                {"open": 101.1, "high": 101.2, "low": 100.7, "close": 100.8},
                {"open": 100.8, "high": 100.9, "low": 100.3, "close": 100.4},
                {"open": 100.4, "high": 100.5, "low": 99.9, "close": 100.0},
                {"open": 100.0, "high": 100.1, "low": 99.5, "close": 99.7},
            ],
        )["data"]
        candidate = _candidate("SELL")
        candidate["_body_ratio"] = 0.93

        with patch.object(M15ZoneScalpInverseStrategy, "_family_candidates", return_value=([candidate], None)):
            signal = M15ZoneScalpInverseStrategy().generate_signal(data)

        self.assertEqual(signal["signal"], "NO_TRADE")
        self.assertEqual(signal["_micro_bias_action"], "ALLOW_BASE_TRADE")
        self.assertEqual(signal["_route_type"], "BLOCKED_BASE_WON")


if __name__ == "__main__":
    unittest.main()
