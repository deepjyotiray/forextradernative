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


if __name__ == "__main__":
    unittest.main()
