import unittest

import pandas as pd

from engine.smc_strategy import SMCStrategy


def _trend_df(direction: str) -> pd.DataFrame:
    rows = []
    price = 100.0
    step = 0.2 if direction == "LONG" else -0.2
    for _ in range(30):
        open_px = price
        close_px = price + step
        high = max(open_px, close_px) + 0.05
        low = min(open_px, close_px) - 0.05
        rows.append({"open": open_px, "high": high, "low": low, "close": close_px})
        price = close_px
    return pd.DataFrame(rows)


class SMCStrategyDirectionalGuardTests(unittest.TestCase):
    def setUp(self):
        self.strategy = SMCStrategy()

    def test_short_requires_sweep_or_rejection(self):
        reason = self.strategy._check_directional_entry_guard(
            direction="SHORT",
            setup={"has_sweep": False, "has_rejection": False},
            m5=_trend_df("SHORT"),
            bias={"direction": "SHORT"},
        )
        self.assertEqual(reason, "SHORT setup needs sweep or rejection")

    def test_counter_bias_short_requires_sweep(self):
        reason = self.strategy._check_directional_entry_guard(
            direction="SHORT",
            setup={"has_sweep": False, "has_rejection": True},
            m5=_trend_df("SHORT"),
            bias={"direction": "LONG"},
        )
        self.assertEqual(reason, "Counter-bias SHORT needs sweep confirmation")

    def test_well_aligned_short_passes_guard(self):
        reason = self.strategy._check_directional_entry_guard(
            direction="SHORT",
            setup={"has_sweep": True, "has_rejection": True},
            m5=_trend_df("SHORT"),
            bias={"direction": "SHORT"},
        )
        self.assertIsNone(reason)

    def test_strong_rejection_setup_softens_counter_trend_m15_structure_guard(self):
        self.assertTrue(
            self.strategy._should_soften_counter_trend_structure_guard(
                {"zone_score": 0.25, "has_rejection": True}
            )
        )

    def test_weak_zone_does_not_soften_counter_trend_m15_structure_guard(self):
        self.assertFalse(
            self.strategy._should_soften_counter_trend_structure_guard(
                {"zone_score": 0.2, "has_rejection": True}
            )
        )

    def test_quality_score_applies_softened_m15_structure_penalty(self):
        setup = {
            "zone_type": "Bullish OB [100.00-101.00]",
            "has_sweep": False,
            "has_rejection": True,
            "rejection_detail": "M5 bullish rejection",
            "_m15_structure_turned": False,
            "_m15_structure_softened": True,
            "_m15_structure_penalty": 0.10,
        }
        score_without_penalty, _ = self.strategy._quality_score(
            {**setup, "_m15_structure_turned": True, "_m15_structure_softened": False},
            direction="LONG",
            m1=_trend_df("LONG"),
            m5=_trend_df("LONG"),
            ind={"body_ratio": 0.6, "range_10": 1.0, "atr14": 2.0, "atr_slope": 0.1},
            tick_snap={"dir_pct": 0.7, "vel_increasing": True},
            counter_trend=True,
            threshold=0.4,
        )
        score_with_penalty, reasons = self.strategy._quality_score(
            setup,
            direction="LONG",
            m1=_trend_df("LONG"),
            m5=_trend_df("LONG"),
            ind={"body_ratio": 0.6, "range_10": 1.0, "atr14": 2.0, "atr_slope": 0.1},
            tick_snap={"dir_pct": 0.7, "vel_increasing": True},
            counter_trend=True,
            threshold=0.4,
        )
        self.assertAlmostEqual(score_without_penalty - score_with_penalty, 0.10, places=3)
        self.assertIn("M15 structure turn pending (-10%)", reasons)


if __name__ == "__main__":
    unittest.main()
