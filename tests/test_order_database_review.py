import os
import tempfile
import unittest
import gc

import engine.order_database as order_database_module
from engine.order_database import OrderDatabase


class OrderDatabaseReviewTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_db_path = order_database_module._DB_PATH
        order_database_module._DB_PATH = os.path.join(self._tmpdir.name, "orders.db")
        self.db = OrderDatabase()

    def tearDown(self):
        self.db = None
        gc.collect()
        order_database_module._DB_PATH = self._orig_db_path
        self._tmpdir.cleanup()

    def _store_closed_order(
        self,
        ticket: int,
        strategy: str,
        profile_name: str,
        final_pnl: float,
        close_reason_category: str,
        mt5_close_reason: str,
        entry_volume_ratio: float,
        entry_tick_pressure_score: float,
    ):
        features = {
            "profile_name": profile_name,
            "entry_volume_ratio": entry_volume_ratio,
            "entry_tick_pressure_score": entry_tick_pressure_score,
        }
        self.db.store_order(
            ticket=ticket,
            direction="BUY",
            volume=0.02,
            entry_price=100.0,
            sl=99.0,
            tp=102.0,
            sl_distance=1.0,
            strategy=strategy,
            confidence=0.8,
            reason="test",
            scalp=(profile_name == "scalp"),
            be_trigger=0.3,
            timeout=60,
            features=features,
            session_type="LONDON",
            market_phase="TRENDING",
        )
        self.db.close_order(
            ticket,
            final_pnl,
            "TEST_CLOSE",
            close_reason_category=close_reason_category,
            peak_r=1.0 if final_pnl >= 0 else 0.4,
            final_r=0.4 if final_pnl > 0 else (-0.3 if final_pnl < 0 else 0.0),
            held_seconds=18.0,
            profile_name=profile_name,
            mt5_close_reason=mt5_close_reason,
        )

    def test_trade_outcome_review_segments_recent_trades(self):
        self._store_closed_order(
            1,
            "SWEEP_SCALPER",
            "scalp",
            0.0,
            "breakeven_stop",
            "sl",
            1.40,
            0.42,
        )
        self._store_closed_order(
            2,
            "SMC_CONFLUENCE",
            "swing_fast",
            2.50,
            "tp",
            "tp",
            1.10,
            0.28,
        )
        self._store_closed_order(
            3,
            "SWEEP_SCALPER",
            "scalp",
            -1.20,
            "sl",
            "sl",
            0.90,
            -0.40,
        )

        report = self.db.get_trade_outcome_review(30)

        self.assertEqual(report["summary"]["trades"], 3)
        self.assertEqual(report["summary"]["wins"], 1)
        self.assertEqual(report["summary"]["losses"], 1)
        self.assertEqual(report["summary"]["breakeven"], 1)
        self.assertEqual(report["breakeven_review"]["count"], 1)

        close_categories = {row["close_reason_category"]: row for row in report["close_reason_categories"]}
        self.assertEqual(close_categories["breakeven_stop"]["trades"], 1)
        self.assertEqual(close_categories["tp"]["wins"], 1)

        mt5_reasons = {row["mt5_close_reason"]: row for row in report["mt5_close_reasons"]}
        self.assertEqual(mt5_reasons["sl"]["trades"], 2)
        self.assertEqual(mt5_reasons["tp"]["wins"], 1)

        volume_buckets = {row["volume_ratio_bucket"]: row for row in report["volume_ratio_buckets"]}
        self.assertEqual(volume_buckets["strong(1.2-1.5x)"]["trades"], 1)
        self.assertEqual(volume_buckets["low(<1.0x)"]["losses"], 1)

        pressure_buckets = {row["pressure_score_bucket"]: row for row in report["pressure_score_buckets"]}
        self.assertEqual(pressure_buckets["strong_long(>=0.35)"]["trades"], 1)
        self.assertEqual(pressure_buckets["strong_short(<=-0.35)"]["losses"], 1)


if __name__ == "__main__":
    unittest.main()
