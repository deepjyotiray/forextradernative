import unittest

from engine.tick_processor import analyze_tick_pressure, entry_pressure_block_reason


class TickPressureTests(unittest.TestCase):
    def test_analyze_tick_pressure_detects_long_bias(self):
        raw_ticks = [
            {"time_msc": 1000, "bid": 100.00, "ask": 100.02},
            {"time_msc": 1200, "bid": 100.01, "ask": 100.03},
            {"time_msc": 1400, "bid": 100.03, "ask": 100.05},
            {"time_msc": 1600, "bid": 100.04, "ask": 100.06},
            {"time_msc": 1800, "bid": 100.06, "ask": 100.08},
            {"time_msc": 2000, "bid": 100.07, "ask": 100.09},
        ]

        pressure = analyze_tick_pressure(raw_ticks)

        self.assertTrue(pressure["ready"])
        self.assertEqual(pressure["directional_bias"], "LONG")
        self.assertGreater(pressure["pressure_score"], 0)
        self.assertTrue(pressure["favorable_long"])

    def test_analyze_tick_pressure_detects_short_bias(self):
        raw_ticks = [
            {"time_msc": 1000, "bid": 100.09, "ask": 100.11},
            {"time_msc": 1200, "bid": 100.08, "ask": 100.10},
            {"time_msc": 1400, "bid": 100.06, "ask": 100.08},
            {"time_msc": 1600, "bid": 100.05, "ask": 100.07},
            {"time_msc": 1800, "bid": 100.03, "ask": 100.05},
            {"time_msc": 2000, "bid": 100.02, "ask": 100.04},
        ]

        pressure = analyze_tick_pressure(raw_ticks)

        self.assertTrue(pressure["ready"])
        self.assertEqual(pressure["directional_bias"], "SHORT")
        self.assertLess(pressure["pressure_score"], 0)
        self.assertTrue(pressure["favorable_short"])

    def test_entry_pressure_blocks_positive_pressure_shorts(self):
        reason = entry_pressure_block_reason(
            "SHORT",
            {"ready": True, "pressure_score": 0.12, "directional_bias": "NEUTRAL"},
            short_positive_veto_threshold=0.05,
        )

        self.assertEqual(reason, "Tick pressure positive (+0.12) against SHORT setup")

    def test_entry_pressure_allows_negative_pressure_shorts(self):
        reason = entry_pressure_block_reason(
            "SELL",
            {"ready": True, "pressure_score": -0.18, "directional_bias": "SHORT"},
            short_positive_veto_threshold=0.05,
        )

        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
