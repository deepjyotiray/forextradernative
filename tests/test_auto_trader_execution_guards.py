from types import SimpleNamespace

from auto_trader import AutoTrader


class _StubTrades:
    def __init__(self, open_trades=None, lock_reason=""):
        self.open_trades = open_trades or {}
        self._lock_reason = lock_reason

    def get_strategy_entry_lockout_reason(self, strategy_name: str, *, setup_signature: str = "") -> str:
        return self._lock_reason


class _StubRisk:
    def calculate_lot(self, account, sl_distance, high_conf=False, strategy=""):
        return 0.01


class _StubStrategyManager:
    def get(self, name: str):
        return None


def _build_trader(open_trades=None, lock_reason=""):
    trader = AutoTrader.__new__(AutoTrader)
    trader.trades = _StubTrades(open_trades=open_trades, lock_reason=lock_reason)
    trader.risk = _StubRisk()
    trader.strat_mgr = _StubStrategyManager()
    return trader


def test_prepare_execution_blocks_when_paired_m15_strategy_is_open():
    trader = _build_trader(
        open_trades={
            1: SimpleNamespace(strategy="M15_ZONE_SCALP_INVERSE"),
        }
    )
    signal = {"signal": "BUY", "sl": 99.0, "tp": 101.0, "lot": 0.01}

    prepared, reason = trader._prepare_execution_order(
        "M15_ZONE_SCALP",
        signal,
        {"ask": 100.0, "bid": 99.9},
        {},
        {"m15_df": None},
    )

    assert prepared is None
    assert reason == "Paired strategy M15_ZONE_SCALP_INVERSE already has an open trade"


def test_prepare_execution_blocks_when_strategy_is_in_post_tier1_lockout():
    trader = _build_trader(lock_reason="post-TIER1 cooldown active (180s remaining, source #123)")
    signal = {"signal": "BUY", "sl": 99.0, "tp": 101.0, "lot": 0.01}

    prepared, reason = trader._prepare_execution_order(
        "M15_ZONE_SCALP",
        signal,
        {"ask": 100.0, "bid": 99.9},
        {},
        {"m15_df": None},
    )

    assert prepared is None
    assert reason == "Execution blocked: post-TIER1 cooldown active (180s remaining, source #123)"
