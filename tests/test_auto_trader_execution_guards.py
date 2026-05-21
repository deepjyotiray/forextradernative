from types import SimpleNamespace

import auto_trader as auto_trader_module
import config as cfg

from auto_trader import AutoTrader


class _StubTrades:
    class _StubOrderDb:
        def __init__(self, closed_orders=None):
            self._closed_orders = list(closed_orders or [])

        def get_closed_orders_opened_between(self, start_time: str, end_time: str):
            return list(self._closed_orders)

    def __init__(self, open_trades=None, lock_reason="", closed_orders=None):
        self.open_trades = open_trades or {}
        self._lock_reason = lock_reason
        self.order_db = self._StubOrderDb(closed_orders=closed_orders)

    def get_strategy_entry_lockout_reason(self, strategy_name: str, *, setup_signature: str = "") -> str:
        return self._lock_reason


class _StubRisk:
    def calculate_lot(self, account, sl_distance, high_conf=False, strategy=""):
        return 0.01


class _StubStrategyManager:
    def get(self, name: str):
        return None


class _SelectableStrategyManager:
    def __init__(self):
        self._auto = True
        self._selected = ["AUTO"]

    @property
    def is_auto(self):
        return self._auto

    @property
    def selected(self):
        return [] if self._auto else list(self._selected)

    def set_selection(self, names):
        cleaned = [str(name or "").upper() for name in names or [] if str(name or "").strip()]
        if cleaned == ["AUTO"] or not cleaned:
            self._auto = True
            self._selected = ["AUTO"]
            return True
        self._auto = False
        self._selected = cleaned
        return True


def _build_trader(open_trades=None, lock_reason="", closed_orders=None):
    trader = AutoTrader.__new__(AutoTrader)
    trader.trades = _StubTrades(open_trades=open_trades, lock_reason=lock_reason, closed_orders=closed_orders)
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


def test_prepare_execution_blocks_after_good_profit_burst_in_current_m15_candle():
    trader = _build_trader(
        closed_orders=[
            {"final_pnl": 0.80},
            {"final_pnl": 0.75},
            {"final_pnl": 0.70},
        ]
    )
    previous_enabled = cfg.M15_CANDLE_PROFIT_THROTTLE_ENABLED
    previous_wins = cfg.M15_CANDLE_PROFIT_THROTTLE_MIN_WINS
    previous_pnl = cfg.M15_CANDLE_PROFIT_THROTTLE_MIN_PNL
    cfg.M15_CANDLE_PROFIT_THROTTLE_ENABLED = True
    cfg.M15_CANDLE_PROFIT_THROTTLE_MIN_WINS = 3
    cfg.M15_CANDLE_PROFIT_THROTTLE_MIN_PNL = 2.0
    try:
        signal = {"signal": "BUY", "sl": 99.0, "tp": 101.0, "lot": 0.01}

        prepared, reason = trader._prepare_execution_order(
            "M15_ZONE_SCALP",
            signal,
            {"ask": 100.0, "bid": 99.9},
            {},
            {"m15_df": None},
        )

        assert prepared is None
        assert reason == "Execution blocked: M15 candle profit throttle active (3 wins, $2.25 booked in current candle)"
    finally:
        cfg.M15_CANDLE_PROFIT_THROTTLE_ENABLED = previous_enabled
        cfg.M15_CANDLE_PROFIT_THROTTLE_MIN_WINS = previous_wins
        cfg.M15_CANDLE_PROFIT_THROTTLE_MIN_PNL = previous_pnl


def test_prepare_execution_reverses_targeted_m15_strategy_when_anti_mode_enabled():
    trader = _build_trader()
    previous = cfg.ANTI_MODE_ENABLED
    previous_mult = cfg.ANTI_MODE_SL_MULTIPLIER
    previous_cap = cfg.ANTI_MODE_MAX_SL_POINTS
    cfg.ANTI_MODE_ENABLED = True
    cfg.ANTI_MODE_SL_MULTIPLIER = 0.8
    cfg.ANTI_MODE_MAX_SL_POINTS = 3.8
    try:
        signal = {"signal": "BUY", "sl": 99.0, "tp": 101.0, "lot": 0.01, "reason": "base long"}

        prepared, reason = trader._prepare_execution_order(
            "M15_ZONE_SCALP",
            signal,
            {"ask": 100.1, "bid": 100.0},
            {},
            {"m15_df": None},
        )

        assert reason == ""
        assert prepared["action"] == "SELL"
        assert prepared["signal"]["_anti_mode"] is True
        assert prepared["signal"]["_anti_original_signal"] == "BUY"
        assert prepared["signal"]["_anti_execution_signal"] == "SELL"
        assert prepared["signal"]["_anti_sl_multiplier"] == cfg.ANTI_MODE_SL_MULTIPLIER
        assert prepared["signal"]["_anti_sl_cap_points"] == cfg.ANTI_MODE_MAX_SL_POINTS
        assert prepared["signal"]["_anti_tp_multiplier"] == cfg.ANTI_MODE_TP_MULTIPLIER
        assert prepared["comment"].endswith("_ANTI")
        assert prepared["sl"] == 100.8
        assert prepared["tp"] == 99.5
    finally:
        cfg.ANTI_MODE_ENABLED = previous
        cfg.ANTI_MODE_SL_MULTIPLIER = previous_mult
        cfg.ANTI_MODE_MAX_SL_POINTS = previous_cap


def test_prepare_execution_does_not_reverse_inverse_strategy_in_anti_mode():
    trader = _build_trader()
    previous = cfg.ANTI_MODE_ENABLED
    cfg.ANTI_MODE_ENABLED = True
    try:
        signal = {"signal": "SELL", "sl": 101.0, "tp": 99.0, "lot": 0.01, "reason": "inverse short"}

        prepared, reason = trader._prepare_execution_order(
            "M15_ZONE_SCALP_INVERSE",
            signal,
            {"ask": 100.1, "bid": 100.0},
            {},
            {"m15_df": None},
        )

        assert reason == ""
        assert prepared["action"] == "SELL"
        assert prepared["signal"].get("_anti_mode") is None
        assert prepared["sl"] == 101.0
        assert prepared["tp"] == 99.0
        assert not prepared["comment"].endswith("_ANTI")
    finally:
        cfg.ANTI_MODE_ENABLED = previous


def test_prepare_execution_blocks_known_bad_anti_context_expansion():
    trader = _build_trader()
    previous_enabled = cfg.ANTI_MODE_ENABLED
    previous_block = cfg.ANTI_MODE_CONTEXT_BLOCK_ENABLED
    previous_session = auto_trader_module.get_session
    cfg.ANTI_MODE_ENABLED = True
    cfg.ANTI_MODE_CONTEXT_BLOCK_ENABLED = True
    auto_trader_module.get_session = lambda: "ASIAN"
    try:
        signal = {
            "signal": "BUY",
            "sl": 99.0,
            "tp": 101.0,
            "lot": 0.01,
            "_htf_state": {"D1": "UP", "H4": "DOWN", "H1": "UP", "M15": "RANGE"},
            "_micro_bias_direction": "SHORT",
            "_pressure_bias": "SHORT",
        }
        prepared, reason = trader._prepare_execution_order(
            "M15_ZONE_SCALP",
            signal,
            {"ask": 100.1, "bid": 100.0},
            {},
            {
                "m15_df": None,
                "regime": {"state": "RANGING"},
                "market_state": {"market_memory": {"direction": "SHORT", "phase": "EXPANSION"}},
            },
        )

        assert prepared is None
        assert reason == "Execution blocked: anti-context blocked: Asian ranging short-memory expansion under mixed HTF range"
    finally:
        cfg.ANTI_MODE_ENABLED = previous_enabled
        cfg.ANTI_MODE_CONTEXT_BLOCK_ENABLED = previous_block
        auto_trader_module.get_session = previous_session


def test_prepare_execution_blocks_known_bad_anti_context_balanced_sell_conflict():
    trader = _build_trader()
    previous_enabled = cfg.ANTI_MODE_ENABLED
    previous_block = cfg.ANTI_MODE_CONTEXT_BLOCK_ENABLED
    previous_session = auto_trader_module.get_session
    cfg.ANTI_MODE_ENABLED = True
    cfg.ANTI_MODE_CONTEXT_BLOCK_ENABLED = True
    auto_trader_module.get_session = lambda: "ASIAN"
    try:
        signal = {
            "signal": "BUY",
            "sl": 99.0,
            "tp": 101.0,
            "lot": 0.01,
            "_htf_state": {"D1": "UP", "H4": "DOWN", "H1": "UP", "M15": "RANGE"},
            "_micro_bias_direction": "LONG",
            "_pressure_bias": "SHORT",
        }
        prepared, reason = trader._prepare_execution_order(
            "M15_ZONE_SCALP",
            signal,
            {"ask": 100.1, "bid": 100.0},
            {},
            {
                "m15_df": None,
                "regime": {"state": "RANGING"},
                "market_state": {"market_memory": {"direction": "SHORT", "phase": "BALANCED"}},
            },
        )

        assert prepared is None
        assert reason == "Execution blocked: anti-context blocked: Asian balanced short-memory sell with long micro bias and short pressure"
    finally:
        cfg.ANTI_MODE_ENABLED = previous_enabled
        cfg.ANTI_MODE_CONTEXT_BLOCK_ENABLED = previous_block
        auto_trader_module.get_session = previous_session


def test_prepare_execution_caps_anti_stop_distance_when_configured():
    trader = _build_trader()
    previous_enabled = cfg.ANTI_MODE_ENABLED
    previous_mult = cfg.ANTI_MODE_SL_MULTIPLIER
    previous_cap = cfg.ANTI_MODE_MAX_SL_POINTS
    cfg.ANTI_MODE_ENABLED = True
    cfg.ANTI_MODE_SL_MULTIPLIER = 1.15
    cfg.ANTI_MODE_MAX_SL_POINTS = 4.5
    try:
        signal = {"signal": "BUY", "sl": 95.0, "tp": 104.0, "lot": 0.01, "reason": "wide base long"}

        prepared, reason = trader._prepare_execution_order(
            "M15_ZONE_SCALP",
            signal,
            {"ask": 100.1, "bid": 100.0},
            {},
            {"m15_df": None},
        )

        assert reason == ""
        assert prepared["action"] == "SELL"
        assert prepared["signal"]["_anti_mirrored_sl_distance"] == 4.0
        assert prepared["signal"]["_anti_effective_sl_distance"] == 4.5
        assert prepared["sl"] == 104.5
    finally:
        cfg.ANTI_MODE_ENABLED = previous_enabled
        cfg.ANTI_MODE_SL_MULTIPLIER = previous_mult
        cfg.ANTI_MODE_MAX_SL_POINTS = previous_cap


def test_set_anti_mode_forces_m15_pair_and_restores_previous_selection():
    trader = AutoTrader.__new__(AutoTrader)
    trader.strat_mgr = _SelectableStrategyManager()
    trader._anti_mode_previous_selection = []
    trader.log = lambda *args, **kwargs: None
    trader._apply_startup_strategy = lambda: trader.strat_mgr.set_selection(["AUTO"])

    previous = cfg.ANTI_MODE_ENABLED
    cfg.ANTI_MODE_ENABLED = False
    try:
        trader.strat_mgr.set_selection(["SMC_CONFLUENCE"])

        enabled_state = trader.set_anti_mode(True)
        assert enabled_state["enabled"] is True
        assert trader.strat_mgr.selected == ["M15_ZONE_SCALP"]

        disabled_state = trader.set_anti_mode(False)
        assert disabled_state["enabled"] is False
        assert trader.strat_mgr.selected == ["SMC_CONFLUENCE"]
    finally:
        cfg.ANTI_MODE_ENABLED = previous
