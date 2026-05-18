import unittest
from datetime import datetime, timedelta, timezone

from engine.backtest_sim import BacktestRequest, BacktestRunner, SimExecutionEngine
from engine.backtest_context import set_backtest_now
from engine.session_filter import get_session_at, is_market_open_at
from engine.session_filter import get_session
from engine.tick_processor import TickProcessor


class BacktestCoreTests(unittest.TestCase):
    def test_backtest_request_parse(self):
        req = BacktestRequest.from_payload(
            {
                "symbol": "xauusd",
                "strategy": "auto",
                "start_utc": "2026-04-01T00:00:00Z",
                "end_utc": "2026-04-02T00:00:00Z",
                "initial_balance": 10000,
                "max_ticks_per_batch": 100,
            }
        )
        self.assertEqual(req.symbol, "XAUUSD")
        self.assertEqual(req.strategy, "AUTO")
        self.assertEqual(req.start_utc.tzinfo, timezone.utc)
        self.assertGreater(req.end_utc, req.start_utc)
        self.assertEqual(req.max_ticks_per_batch, 200)  # lower bound guard

    def test_tick_processor_uses_tick_timestamp(self):
        tp = TickProcessor(buffer_size=50)
        base = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc).timestamp()
        for i in range(12):
            tp.feed({"bid": 100 + i * 0.01, "ask": 100.02 + i * 0.01, "spread": 0.02, "ts": base + i})
        snap = tp.snapshot()
        self.assertTrue(snap["ready"])
        self.assertGreater(snap["velocity"], 0)

    def test_sim_engine_opens_and_closes_trade(self):
        sim = SimExecutionEngine(initial_balance=10000.0)
        now = datetime.now(timezone.utc)
        ticket = sim.open_from_signal(
            symbol="XAUUSD",
            strategy="SMC_CONFLUENCE",
            signal={
                "signal": "BUY",
                "sl": 99.0,
                "tp": 101.0,
                "confidence": 0.8,
                "_timeout": 120,
            },
            now_utc=now,
            tick={"bid": 100.0, "ask": 100.02},
            tick_count=10,
        )
        self.assertIsNotNone(ticket)
        sim.on_tick(now + timedelta(seconds=1), {"bid": 101.1, "ask": 101.12}, 11, {"velocity": 5})
        self.assertEqual(len(sim.open_trades), 0)
        self.assertEqual(len(sim.closed_trades), 1)
        self.assertGreater(sim.closed_trades[0]["pnl"], 0)

    def test_session_helpers_at_time(self):
        self.assertEqual(get_session_at(datetime(2026, 4, 1, 8, 0, tzinfo=timezone.utc)), "LONDON")
        # Saturday always closed
        self.assertFalse(is_market_open_at(datetime(2026, 4, 4, 8, 0, tzinfo=timezone.utc)))

    def test_session_uses_backtest_override_time(self):
        try:
            set_backtest_now(datetime(2026, 4, 1, 14, 0, tzinfo=timezone.utc))
            self.assertEqual(get_session(), "NEW_YORK")
        finally:
            set_backtest_now(None)

    def test_backtest_runner_registers_all_live_strategies(self):
        runner = BacktestRunner()
        self.assertEqual(
            runner.strategy_manager.available,
            [
                "AUTO",
                "SMC_CONFLUENCE",
                "M15_SCALP_DEEP",
                "M15_ZONE_SCALP",
                "M15_ZONE_SCALP_INVERSE",
                "TREND_CHANNEL",
                "SWING_ENGINE",
                "HTF_LONG",
                "HTF_SHORT",
            ],
        )


if __name__ == "__main__":
    unittest.main()
