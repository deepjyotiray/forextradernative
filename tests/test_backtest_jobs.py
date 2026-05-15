import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd

from engine.backtest_jobs import BacktestJob, BacktestJobManager
from engine.backtest_sim import BacktestRequest


class StubProvider:
    def __init__(self):
        self.coverage_checks = []

    def disconnect(self):
        return None

    def load_tick_density_profile(self, symbol, start_utc, end_utc):
        minute_index = pd.date_range(start=start_utc, end=end_utc - timedelta(minutes=1), freq="min", tz="UTC")
        frame = pd.DataFrame(
            {
                "bucket_start": minute_index,
                "estimated_ticks": [10] * len(minute_index),
            }
        )
        frame.attrs["density_source"] = "stub_density"
        return frame

    def describe_cache_coverage(self, symbol, start_utc, end_utc, warmup_hours=72):
        self.coverage_checks.append((start_utc, end_utc))
        if end_utc <= datetime(2026, 1, 2, 1, 0, tzinfo=timezone.utc):
            return {
                "cache_entry_found": True,
                "has_tick_overlap": False,
                "effective_start_utc": "2026-01-02T01:00:01+00:00",
                "effective_end_utc": "2026-04-24T22:59:59+00:00",
            }
        return {
            "cache_entry_found": True,
            "has_tick_overlap": True,
            "effective_start_utc": "2026-01-02T01:00:01+00:00",
            "effective_end_utc": "2026-04-24T22:59:59+00:00",
        }


class BacktestJobsTests(unittest.TestCase):
    def _build_job(self, split_mode="DAILY"):
        req = BacktestRequest.from_payload(
            {
                "symbol": "XAUUSD",
                "strategy": "AUTO",
                "start_utc": "2026-01-01T06:09:00Z",
                "end_utc": "2026-01-04T06:09:00Z",
                "initial_balance": 10000,
                "fast_mode": True,
                "split_mode": split_mode,
                "parallel_workers": 20,
                "cache_only": True,
                "max_ticks_per_batch": 50000,
                "replay_max_points": 1500,
            }
        )
        return BacktestJob(job_id="job123", request=req)

    def test_parallel_preflight_fails_before_submitting_uncovered_cache_chunk(self):
        mgr = BacktestJobManager()
        job = self._build_job(split_mode="DAILY")
        provider = StubProvider()

        with patch("engine.backtest_jobs.BacktestDataProvider", return_value=provider):
            with self.assertRaises(RuntimeError) as ctx:
                mgr._run_job_parallel_chunks(job)

        self.assertIn("no tick overlap", str(ctx.exception))
        self.assertEqual(len(job.chunk_states), 4)
        self.assertEqual(job.chunk_states[0]["status"], "queued")
        self.assertGreaterEqual(len(provider.coverage_checks), 1)

    def test_time_split_populates_density_and_estimated_ticks(self):
        mgr = BacktestJobManager()
        job = self._build_job(split_mode="DAILY")
        provider = StubProvider()

        with patch("engine.backtest_jobs.BacktestDataProvider", return_value=provider):
            try:
                mgr._run_job_parallel_chunks(job)
            except RuntimeError:
                pass

        self.assertEqual(job.planning_meta.get("planning_mode"), "time_split")
        self.assertEqual(job.planning_meta.get("density_source"), "stub_density")
        self.assertTrue(job.planning_meta.get("timeline_density"))

        ranges = [
            (
                datetime(2026, 1, 1, 6, 9, tzinfo=timezone.utc),
                datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc),
            ),
            (
                datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc),
                datetime(2026, 1, 3, 0, 0, tzinfo=timezone.utc),
            ),
        ]
        density = provider.load_tick_density_profile("XAUUSD", job.request.start_utc, job.request.end_utc)
        estimates = mgr._estimate_chunk_ticks_from_density(density, ranges)
        self.assertGreater(estimates[0], 0)
        self.assertGreater(estimates[1], estimates[0])

    def test_create_job_accepts_m15_zone_scalp(self):
        mgr = BacktestJobManager()
        req = BacktestRequest.from_payload(
            {
                "symbol": "XAUUSD",
                "strategy": "M15_ZONE_SCALP",
                "start_utc": "2026-01-01T00:00:00Z",
                "end_utc": "2026-01-02T00:00:00Z",
                "initial_balance": 10000,
            }
        )

        with patch.object(mgr._executor, "submit", return_value=None):
            job = mgr.create_job(req)

        self.assertEqual(job.request.strategy, "M15_ZONE_SCALP")

    def test_create_job_accepts_m15_scalp_deep(self):
        mgr = BacktestJobManager()
        req = BacktestRequest.from_payload(
            {
                "symbol": "XAUUSD",
                "strategy": "M15_SCALP_DEEP",
                "start_utc": "2026-01-01T00:00:00Z",
                "end_utc": "2026-01-02T00:00:00Z",
                "initial_balance": 10000,
            }
        )

        with patch.object(mgr._executor, "submit", return_value=None):
            job = mgr.create_job(req)

        self.assertEqual(job.request.strategy, "M15_SCALP_DEEP")

    def test_real_cache_chunk_load_survives_empty_m1_cache_file(self):
        from engine.backtest_data import BacktestDataProvider

        provider = BacktestDataProvider()
        start = datetime.fromisoformat("2026-03-25T20:33:00+00:00")
        end = datetime.fromisoformat("2026-04-02T10:57:00+00:00")

        ds = provider.load_dataset("XAUUSD", start, end, cache_only=True)
        self.assertGreater(len(ds.ticks), 0)
        self.assertEqual(ds.data_source, "cache")


if __name__ == "__main__":
    unittest.main()
