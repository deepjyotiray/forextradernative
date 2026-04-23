import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from engine.visual_analytics import VisualAnalyticsEngine


def build_sample_decision_data() -> pd.DataFrame:
    rows = []
    base_time = pd.Timestamp("2026-04-20T07:00:00Z")

    completed_rows = [
        ("TRADE_TAKEN", "WIN", 8.4, "LONG", "LONDON", "WITH_TREND", 0.81, 0.18, 0.72, 18.0, 0.22, 55),
        ("TRADE_TAKEN", "LOSS", -6.1, "SHORT", "LONDON", "COUNTER_TREND", 0.63, 0.29, 0.42, 11.0, 0.64, 120),
        ("TRADE_TAKEN", "WIN", 10.2, "LONG", "OVERLAP", "WITH_TREND", 0.77, 0.21, 0.69, 22.0, 0.31, 88),
        ("TRADE_TAKEN", "WIN", 6.7, "SHORT", "NY", "WITH_TREND", 0.74, 0.24, 0.67, 16.0, 0.35, 76),
        ("TRADE_TAKEN", "LOSS", -4.8, "LONG", "NY", "COUNTER_TREND", 0.61, 0.33, 0.54, 9.0, 0.71, 142),
        ("TRADE_TAKEN", "BE", 0.0, "SHORT", "OVERLAP", "NEUTRAL", 0.68, 0.26, 0.57, 13.5, 0.49, 64),
        ("TRADE_TAKEN", "WIN", 7.9, "LONG", "LONDON", "WITH_TREND", 0.79, 0.19, 0.75, 19.5, 0.18, 51),
        ("TRADE_TAKEN", "LOSS", -5.3, "SHORT", "NY", "COUNTER_TREND", 0.58, 0.31, 0.46, 10.5, 0.68, 133),
    ]

    for idx, row in enumerate(completed_rows, start=1):
        decision, outcome, pnl, direction, session, trade_type, quality, spread, tick_ratio, tick_velocity, spread_pct, duration = row
        rows.append(
            {
                "timestamp": base_time + pd.Timedelta(minutes=idx * 18),
                "completion_time": base_time + pd.Timedelta(minutes=idx * 18 + 4),
                "unix_time": float((base_time + pd.Timedelta(minutes=idx * 18)).timestamp()),
                "strategy": "SCALPER" if idx % 2 else "SMC",
                "setup_direction": direction,
                "bias_direction": direction if trade_type == "WITH_TREND" else ("LONG" if direction == "SHORT" else "SHORT"),
                "trade_type": trade_type,
                "quality_score": quality,
                "threshold_used": 0.65,
                "spread_mean": spread,
                "spread_std": 0.03 + idx * 0.002,
                "spread_percentile": spread_pct,
                "atr_value": 1.1 + idx * 0.08,
                "compression_flag": idx % 3 == 0,
                "ltf_conflict_flag": idx % 4 == 0,
                "tick_ratio": tick_ratio,
                "tick_velocity": tick_velocity,
                "session": session,
                "decision": decision,
                "reason": "signal_confirmed",
                "price": 3320.0 + idx,
                "trade_id": f"trade_{idx}",
                "entry_price": 3320.0 + idx,
                "exit_price": 3320.0 + idx + (pnl / 10.0),
                "sl": 3319.0 + idx,
                "tp": 3321.0 + idx,
                "outcome": outcome,
                "pnl": pnl,
                "trade_duration": duration,
                "exit_reason": "TP" if outcome == "WIN" else "SL" if outcome == "LOSS" else "TIMEOUT",
            }
        )

    skipped_rows = [
        ("LONDON", "Spread too wide | mean=0.34"),
        ("LONDON", "Quality below threshold | 0.58"),
        ("OVERLAP", "Low tick ratio | 0.44"),
        ("NY", "Spread too wide | mean=0.36"),
    ]

    for offset, (session, reason) in enumerate(skipped_rows, start=len(completed_rows) + 1):
        rows.append(
            {
                "timestamp": base_time + pd.Timedelta(minutes=offset * 18),
                "completion_time": pd.NaT,
                "unix_time": float((base_time + pd.Timedelta(minutes=offset * 18)).timestamp()),
                "strategy": "SCALPER",
                "setup_direction": "LONG" if offset % 2 else "SHORT",
                "bias_direction": "LONG",
                "trade_type": "WITH_TREND",
                "quality_score": 0.55,
                "threshold_used": 0.65,
                "spread_mean": 0.34,
                "spread_std": 0.04,
                "spread_percentile": 0.72,
                "atr_value": 1.5,
                "compression_flag": False,
                "ltf_conflict_flag": False,
                "tick_ratio": 0.48,
                "tick_velocity": 8.5,
                "session": session,
                "decision": "TRADE_SKIPPED",
                "reason": reason,
                "price": 3335.0 + offset,
                "trade_id": None,
                "entry_price": None,
                "exit_price": None,
                "sl": None,
                "tp": None,
                "outcome": None,
                "pnl": None,
                "trade_duration": None,
                "exit_reason": None,
            }
        )

    return pd.DataFrame(rows)


class VisualAnalyticsEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = VisualAnalyticsEngine()
        self.decision_df = build_sample_decision_data()
        self.completed_df = self.decision_df[self.decision_df["decision"] == "TRADE_TAKEN"].dropna(subset=["outcome"])
        self.temp_dir = Path(tempfile.mkdtemp(prefix="phase3_visuals_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_visual_suite_exports_all_phase3_charts(self):
        result = self.engine.generate_visual_suite_from_dataframes(
            decision_df=self.decision_df,
            completed_df=self.completed_df,
            output_dir=str(self.temp_dir),
            metadata={"period_days": 7},
        )

        self.assertEqual(result["chart_count"], 6)
        self.assertEqual(result["completed_trade_count"], len(self.completed_df))
        self.assertEqual(result["decision_count"], len(self.decision_df))
        self.assertEqual(result["period_days"], 7)

        for chart_key, chart in result["charts"].items():
            with self.subTest(chart=chart_key):
                chart_path = Path(chart["path"])
                self.assertTrue(chart_path.exists(), f"{chart_key} was not exported")
                self.assertGreater(chart_path.stat().st_size, 0, f"{chart_key} export is empty")

    def test_empty_data_produces_placeholder_exports(self):
        empty_df = pd.DataFrame(columns=self.decision_df.columns)
        result = self.engine.generate_visual_suite_from_dataframes(
            decision_df=empty_df,
            completed_df=empty_df,
            output_dir=str(self.temp_dir / "empty"),
        )

        self.assertEqual(result["chart_count"], 6)
        self.assertEqual(result["completed_trade_count"], 0)
        self.assertEqual(result["decision_count"], 0)

        for chart in result["charts"].values():
            chart_path = Path(chart["path"])
            self.assertTrue(chart_path.exists())
            self.assertGreater(chart_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
