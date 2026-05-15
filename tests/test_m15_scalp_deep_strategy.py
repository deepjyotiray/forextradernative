import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from config import get_exit_profile_config
from engine.m15_scalp_deep_strategy import M15ScalpDeepStrategy


def _build_m15_frame() -> pd.DataFrame:
    base = []
    start = datetime(2026, 5, 15, 0, 0, tzinfo=timezone.utc)
    price = 4600.0
    for i in range(48):
        if i < 42:
            o = price
            c = price - 0.30
            h = max(o, c) + 0.20
            l = min(o, c) - 0.20
            price = c
        elif i == 42:
            o, h, l, c = 4591.0, 4591.6, 4589.7, 4590.2
            price = c
        elif i == 43:
            o, h, l, c = 4590.2, 4590.5, 4588.8, 4589.0
            price = c
        elif i == 44:
            o, h, l, c = 4589.0, 4589.2, 4587.7, 4588.1
            price = c
        elif i == 45:
            o, h, l, c = 4588.1, 4588.8, 4587.8, 4588.6
            price = c
        elif i == 46:
            o, h, l, c = 4589.6, 4591.6, 4587.4, 4587.8
            price = c
        else:
            o, h, l, c = 4587.8, 4588.0, 4586.7, 4587.2
            price = c
        base.append(
            {
                "datetime": start + pd.Timedelta(minutes=15 * i),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
            }
        )
    return pd.DataFrame(base)


def _build_m15_frame_long() -> pd.DataFrame:
    base = []
    start = datetime(2026, 5, 15, 0, 0, tzinfo=timezone.utc)
    price = 4600.0
    for i in range(48):
        if i < 42:
            o = price
            c = price + 0.30
            h = max(o, c) + 0.20
            l = min(o, c) - 0.20
            price = c
        elif i == 42:
            o, h, l, c = 4609.0, 4610.1, 4608.7, 4609.6
            price = c
        elif i == 43:
            o, h, l, c = 4609.6, 4610.6, 4609.4, 4610.3
            price = c
        elif i == 44:
            o, h, l, c = 4610.3, 4611.1, 4610.0, 4610.9
            price = c
        elif i == 45:
            o, h, l, c = 4610.9, 4611.2, 4610.4, 4610.8
            price = c
        elif i == 46:
            o, h, l, c = 4609.8, 4612.0, 4609.0, 4611.8
            price = c
        else:
            o, h, l, c = 4611.8, 4612.1, 4611.3, 4612.0
            price = c
        base.append(
            {
                "datetime": start + pd.Timedelta(minutes=15 * i),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
            }
        )
    return pd.DataFrame(base)


class M15ScalpDeepStrategyTests(unittest.TestCase):
    def _data(self):
        return {
            "symbol": "XAUUSD",
            "tick": {"bid": 4590.25, "ask": 4590.43, "spread": 0.18},
            "m15_df": _build_m15_frame(),
            "account": {"balance": 10000.0},
            "calendar": {},
            "regime": {"state": "TRENDING"},
            "bias": {"direction": "SHORT", "confidence": 0.85},
            "market_state": {
                "trend": {"D1": "DOWN", "H4": "DOWN"},
                "structure": {"D1": "DOWN", "H4": "DOWN"},
            },
            "tick_snapshot": {"velocity": 6.2},
            "tick_pressure": {"pressure_score": -0.18, "directional_bias": "SHORT", "burst_rate": 5.4},
            "strategy_trade_counts": {},
            "now_utc": datetime(2026, 5, 15, 3, 30, tzinfo=timezone.utc),
        }

    def _zone_patch(self):
        return patch(
            "engine.m15_scalp_deep_strategy.ZoneDetector.scored_clusters",
            return_value=[
                {
                    "zone_low": 4590.0,
                    "zone_high": 4591.2,
                    "zone_mid": 4590.6,
                    "strength": 0.82,
                }
            ],
        )

    def _long_data(self):
        return {
            "symbol": "XAUUSD",
            "tick": {"bid": 4610.45, "ask": 4610.63, "spread": 0.18},
            "m15_df": _build_m15_frame_long(),
            "account": {"balance": 10000.0},
            "calendar": {},
            "regime": {"state": "TRENDING"},
            "bias": {"direction": "LONG", "confidence": 0.85},
            "market_state": {
                "trend": {"D1": "UP", "H4": "UP"},
                "structure": {"D1": "UP", "H4": "UP"},
            },
            "tick_snapshot": {"velocity": 6.2},
            "tick_pressure": {"pressure_score": 0.18, "directional_bias": "LONG", "burst_rate": 5.4},
            "strategy_trade_counts": {},
            "now_utc": datetime(2026, 5, 15, 3, 30, tzinfo=timezone.utc),
        }

    def _long_zone_patch(self):
        return patch(
            "engine.m15_scalp_deep_strategy.ZoneDetector.scored_clusters",
            return_value=[
                {
                    "zone_low": 4609.1,
                    "zone_high": 4609.9,
                    "zone_mid": 4609.5,
                    "strength": 0.82,
                }
            ],
        )

    def test_generate_signal_returns_rr_above_minimum(self):
        strat = M15ScalpDeepStrategy()

        with self._zone_patch():
            signal = strat.generate_signal(self._data())

        self.assertEqual(signal["signal"], "SELL")
        self.assertEqual(signal["_exit_profile"], "m15_scalp_deep")
        self.assertGreaterEqual(signal["rr"], 1.15)
        self.assertGreater(signal["tp"], 0)

    def test_generate_buy_signal_returns_rr_above_minimum(self):
        strat = M15ScalpDeepStrategy()

        with self._long_zone_patch():
            signal = strat.generate_signal(self._long_data())

        self.assertEqual(signal["signal"], "BUY")
        self.assertEqual(signal["_exit_profile"], "m15_scalp_deep")
        self.assertGreaterEqual(signal["rr"], 1.15)
        self.assertGreater(signal["tp"], signal["entry"])

    def test_blocks_mixed_higher_timeframes_even_with_bias_fallback(self):
        strat = M15ScalpDeepStrategy()
        data = self._data()
        data["market_state"] = {
            "trend": {"D1": "UP", "H4": "DOWN"},
            "structure": {"D1": "UP", "H4": "DOWN"},
        }

        with self._zone_patch():
            signal = strat.generate_signal(data)

        self.assertEqual(signal["signal"], "NO_TRADE")
        self.assertIn("No deep-confluence", signal["reason"])

    def test_blocks_weak_micro_velocity(self):
        strat = M15ScalpDeepStrategy()
        data = self._data()
        data["tick_snapshot"] = {"velocity": 3.8}

        with self._zone_patch():
            signal = strat.generate_signal(data)

        self.assertEqual(signal["signal"], "NO_TRADE")
        self.assertIn("No deep-confluence", signal["reason"])

    def test_exit_profile_is_defined(self):
        profile = get_exit_profile_config("m15_scalp_deep")

        self.assertEqual(profile["profile_name"], "m15_scalp_deep")
        self.assertGreater(profile["be_trigger_r"], 0)


if __name__ == "__main__":
    unittest.main()
