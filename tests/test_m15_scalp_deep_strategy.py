import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from config import get_exit_profile_config
from engine.m15_scalp_deep_strategy import M15ScalpDeepStrategy


def _cfg(**overrides):
    base = {
        "spread_max": 0.45,
        "pip_size": 0.1,
        "min_h1_bars": 18,
        "min_m5_bars": 10,
        "target_rr": 1.8,
        "min_rr": 1.3,
        "sl_buffer_atr_mult": 0.20,
        "sl_buffer_pips": 3,
        "sl_floor_pips": 12,
        "sl_ceiling_pips": 40,
        "htf_zone_touch_pts": 0.35,
        "htf_zone_touch_atr_mult": 0.0,
        "h1_swing_window": 2,
        "min_consecutive_bos": 2,
        "h1_impulse_atr_mult": 0.8,
        "m5_lookback_bars": 16,
        "m5_structure_reference_bars": 4,
        "m5_stop_lookback": 6,
        "breaker_overlap_pts": 0.10,
        "breaker_overlap_atr_mult": 0.0,
        "breaker_entry_tolerance_pts": 0.20,
        "breaker_entry_tolerance_atr_mult": 0.0,
        "min_entry_velocity": 5.0,
        "min_entry_burst_rate": 3.0,
        "min_signed_pressure": -0.02,
        "opposing_bias_pressure_threshold": 0.05,
        "allow_ranging": False,
        "sessions": ["LONDON", "NEW_YORK"],
        "news_block": False,
        "max_active_trades": 1,
        "risk_pct": 0.35,
        "fixed_lot": None,
    }
    base.update(overrides)
    return base


def _build_h1_bullish_frame(mitigated: bool = False) -> pd.DataFrame:
    rows = [
        {"open": 100.0, "high": 100.6, "low": 99.8, "close": 100.4},
        {"open": 100.4, "high": 101.1, "low": 100.2, "close": 100.9},
        {"open": 100.9, "high": 101.0, "low": 100.7, "close": 100.8},
        {"open": 100.8, "high": 101.9, "low": 100.7, "close": 101.6},
        {"open": 101.6, "high": 101.7, "low": 101.0, "close": 101.2},
        {"open": 101.2, "high": 101.3, "low": 100.6, "close": 100.8},
        {"open": 100.8, "high": 102.1, "low": 100.7, "close": 101.9},
        {"open": 101.9, "high": 102.3, "low": 101.8, "close": 102.1},
        {"open": 102.1, "high": 102.2, "low": 101.7, "close": 101.9},
        {"open": 101.9, "high": 102.0, "low": 101.6, "close": 101.7},
        {"open": 101.7, "high": 102.4, "low": 101.95 if not mitigated else 101.85, "close": 102.2},
        {"open": 102.2, "high": 102.9, "low": 102.1, "close": 102.8},
        {"open": 102.8, "high": 103.0, "low": 102.6, "close": 102.9},
        {"open": 102.9, "high": 102.95, "low": 102.2, "close": 102.4},
        {"open": 102.4, "high": 102.5, "low": 102.0, "close": 102.1},
        {"open": 102.1, "high": 102.35, "low": 102.15, "close": 102.25},
        {"open": 102.25, "high": 102.4, "low": 102.1, "close": 102.3},
        {"open": 102.3, "high": 102.45, "low": 102.2, "close": 102.35},
    ]
    start = datetime(2026, 5, 15, 0, 0, tzinfo=timezone.utc)
    for i, row in enumerate(rows):
        row["datetime"] = start + pd.Timedelta(hours=i)
    return pd.DataFrame(rows)


def _build_m5_breaker_frame() -> pd.DataFrame:
    rows = [
        {"open": 102.04, "high": 102.05, "low": 101.99, "close": 102.02},
        {"open": 102.02, "high": 102.03, "low": 101.96, "close": 101.99},
        {"open": 101.99, "high": 102.00, "low": 101.93, "close": 101.96},
        {"open": 101.96, "high": 101.99, "low": 101.88, "close": 101.92},
        {"open": 101.92, "high": 101.95, "low": 101.72, "close": 101.80},
        {"open": 101.80, "high": 101.96, "low": 101.78, "close": 101.94},
        {"open": 101.94, "high": 102.12, "low": 101.92, "close": 102.10},
        {"open": 102.10, "high": 102.14, "low": 102.00, "close": 102.02},
        {"open": 102.02, "high": 102.04, "low": 101.82, "close": 101.88},
        {"open": 101.88, "high": 101.93, "low": 101.84, "close": 101.90},
    ]
    start = datetime(2026, 5, 15, 8, 0, tzinfo=timezone.utc)
    for i, row in enumerate(rows):
        row["datetime"] = start + pd.Timedelta(minutes=5 * i)
    return pd.DataFrame(rows)


class M15ScalpDeepStrategyTests(unittest.TestCase):
    def test_find_h1_unmitigated_bullish_zone(self):
        strat = M15ScalpDeepStrategy()
        setup = strat._find_htf_setup(_build_h1_bullish_frame(), _cfg())

        self.assertIsNotNone(setup)
        self.assertEqual(setup["direction"], "BUY")
        self.assertEqual(setup["zone"]["idx"], 9)
        self.assertAlmostEqual(setup["zone"]["zone_low"], 101.6, places=2)
        self.assertAlmostEqual(setup["zone"]["zone_high"], 101.9, places=2)

    def test_blocks_mitigated_h1_zone(self):
        strat = M15ScalpDeepStrategy()
        setup = strat._find_htf_setup(_build_h1_bullish_frame(mitigated=True), _cfg())

        self.assertIsNone(setup)

    def test_finds_m5_breaker_confirmation_inside_h1_zone(self):
        strat = M15ScalpDeepStrategy()
        zone = {"zone_low": 101.6, "zone_high": 101.9}
        breaker = strat._find_m5_breaker_confirmation(_build_m5_breaker_frame(), "BUY", zone, _cfg())

        self.assertIsNotNone(breaker)
        self.assertEqual(breaker["idx"], 4)
        self.assertEqual(breaker["shift_idx"], 6)
        self.assertAlmostEqual(breaker["zone_low"], 101.72, places=2)
        self.assertAlmostEqual(breaker["zone_high"], 101.92, places=2)

    def test_generate_signal_returns_buy_for_strategy2_setup(self):
        strat = M15ScalpDeepStrategy()
        data = {
            "symbol": "XAUUSD",
            "tick": {"bid": 101.88, "ask": 101.96, "spread": 0.08},
            "h1_df": _build_h1_bullish_frame(),
            "m5_df": _build_m5_breaker_frame(),
            "account": {"balance": 10000.0},
            "calendar": {},
            "regime": {"state": "TRENDING"},
            "liquidity": {"key_levels": {"session_high": 103.2}},
            "tick_snapshot": {"velocity": 6.2},
            "tick_pressure": {"pressure_score": 0.18, "directional_bias": "LONG", "burst_rate": 5.4},
            "strategy_trade_counts": {},
            "now_utc": datetime(2026, 5, 15, 8, 45, tzinfo=timezone.utc),
        }

        with patch("engine.m15_scalp_deep_strategy._scfg.get", return_value=_cfg()):
            signal = strat.generate_signal(data)

        self.assertEqual(signal["signal"], "BUY")
        self.assertEqual(signal["_exit_profile"], "m15_scalp_deep")
        self.assertTrue(signal["_breaker_block_confirmed"])
        self.assertGreaterEqual(signal["rr"], 1.3)
        self.assertGreater(signal["tp"], signal["entry"])

    def test_exit_profile_is_defined(self):
        profile = get_exit_profile_config("m15_scalp_deep")

        self.assertEqual(profile["profile_name"], "m15_scalp_deep")
        self.assertGreater(profile["be_trigger_r"], 0)


if __name__ == "__main__":
    unittest.main()
