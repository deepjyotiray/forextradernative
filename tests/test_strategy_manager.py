from unittest.mock import patch

from engine.strategies.base_strategy import BaseStrategy
from engine.strategy_manager import StrategyManager, _normalize_signal


class _StubStrategy(BaseStrategy):
    def __init__(self, name, payload):
        self.name = name
        self._payload = dict(payload)

    def generate_signal(self, data):
        return dict(self._payload)


def test_normalize_signal_adds_sweep_gate_metadata():
    signal = _normalize_signal({"signal": "BUY", "entry": 100.0}, "M15_SCALP_DEEP")

    assert signal["_strategy_name"] == "M15_SCALP_DEEP"
    assert signal["_signal_family"] == "M15"
    assert signal["_sweep_confirmed"] is False
    assert signal["_candle_confirmation"] is True


def test_normalize_signal_adds_m15_gate_metadata_from_reclaim_flag():
    signal = _normalize_signal(
        {"signal": "SELL", "entry": 100.0, "_sweep_reclaim": True},
        "M15_ZONE_SCALP",
    )

    assert signal["_strategy_name"] == "M15_ZONE_SCALP"
    assert signal["_signal_family"] == "M15"
    assert signal["_sweep_confirmed"] is True
    assert signal["_candle_confirmation"] is True


def test_normalize_signal_marks_inverse_zone_scalp_as_m15_family():
    signal = _normalize_signal({"signal": "BUY", "entry": 100.0}, "M15_ZONE_SCALP_INVERSE")

    assert signal["_strategy_name"] == "M15_ZONE_SCALP_INVERSE"
    assert signal["_signal_family"] == "M15"
    assert signal["_candle_confirmation"] is True


def test_auto_mode_evaluates_inverse_zone_scalp_independently():
    manager = StrategyManager()
    manager.register(_StubStrategy("M15_ZONE_SCALP", {"signal": "BUY", "entry": 100.0, "confidence": 0.70, "rr": 1.8}))
    manager.register(_StubStrategy("M15_ZONE_SCALP_INVERSE", {"signal": "SELL", "entry": 100.0, "confidence": 0.95, "rr": 2.4}))

    def fake_gate(signal, market_state, risk_manager=None, m15_context_provider=None):
        return {
            "allowed": True,
            "reason": "MASTER_GATE_PASS",
            "score": 80,
            "confirmation_count": 3,
            "counter_trend": False,
        }

    with patch("engine.strategy_manager.master_trade_gate", side_effect=fake_gate):
        result = manager.evaluate_all({"bias": {"direction": "LONG"}})

    assert result["strategy"] == "M15_ZONE_SCALP"
    assert result["signal"]["signal"] == "BUY"
    assert result["all_results"]["M15_ZONE_SCALP"]["signal"] == "BUY"
    assert result["all_results"]["M15_ZONE_SCALP_INVERSE"]["signal"] == "SELL"
    assert result["all_results"]["M15_ZONE_SCALP_INVERSE"]["arb_score"] > 0


def test_normalize_signal_marks_m15_scalp_deep_as_m15_family():
    signal = _normalize_signal({"signal": "BUY", "entry": 100.0}, "M15_SCALP_DEEP")

    assert signal["_strategy_name"] == "M15_SCALP_DEEP"
    assert signal["_signal_family"] == "M15"
    assert signal["_candle_confirmation"] is True


def test_normalize_signal_preserves_existing_gate_metadata():
    signal = _normalize_signal(
        {
            "signal": "BUY",
            "entry": 100.0,
            "_signal_family": "SMC",
            "_sweep_confirmed": True,
            "_candle_confirmation": False,
        },
        "SMC_CONFLUENCE",
    )

    assert signal["_strategy_name"] == "SMC_CONFLUENCE"
    assert signal["_signal_family"] == "SMC"
    assert signal["_sweep_confirmed"] is True
    assert signal["_candle_confirmation"] is False


def test_auto_mode_prefers_gate_allowed_signal_over_higher_confidence_blocked_signal():
    manager = StrategyManager()
    manager.register(_StubStrategy("M15_SCALP_DEEP", {"signal": "BUY", "entry": 100.0, "confidence": 0.92, "rr": 1.7}))
    manager.register(_StubStrategy("SMC_CONFLUENCE", {"signal": "BUY", "entry": 100.0, "confidence": 0.74, "rr": 1.8}))

    def fake_gate(signal, market_state, risk_manager=None, m15_context_provider=None):
        if signal["_strategy_name"] == "M15_SCALP_DEEP":
            return {
                "allowed": False,
                "reason": "SPREAD_BLOCK: 0.600 > 0.450",
                "score": 95,
                "confirmation_count": 4,
                "counter_trend": False,
            }
        return {
            "allowed": True,
            "reason": "MASTER_GATE_PASS",
            "score": 82,
            "confirmation_count": 4,
            "counter_trend": False,
        }

    with patch("engine.strategy_manager.master_trade_gate", side_effect=fake_gate):
        result = manager.evaluate_all({"bias": {"direction": "LONG"}, "tick_pressure": {"pressure_score": 0.2}})

    assert result["strategy"] == "SMC_CONFLUENCE"
    assert result["signal"]["signal"] == "BUY"
    assert result["signal"]["_arb_log"]["strategy"] == "SMC_CONFLUENCE"
    assert result["all_results"]["M15_SCALP_DEEP"]["gate_allowed"] is False
    assert "SPREAD_BLOCK" in result["all_results"]["M15_SCALP_DEEP"]["reason"]


def test_auto_mode_prefers_bias_aligned_signal_when_gate_quality_is_similar():
    manager = StrategyManager()
    manager.register(
        _StubStrategy(
            "M15_SCALP_DEEP",
            {"signal": "SELL", "entry": 100.0, "confidence": 0.88, "rr": 1.6, "_entry_tick_pressure_score": -0.05},
        )
    )
    manager.register(
        _StubStrategy(
            "SMC_CONFLUENCE",
            {"signal": "BUY", "entry": 100.0, "confidence": 0.75, "rr": 1.6, "_entry_tick_pressure_score": 0.22},
        )
    )

    def fake_gate(signal, market_state, risk_manager=None, m15_context_provider=None):
        is_counter = signal["_strategy_name"] == "M15_SCALP_DEEP"
        return {
            "allowed": True,
            "reason": "MASTER_GATE_PASS",
            "score": 86,
            "confirmation_count": 4,
            "counter_trend": is_counter,
        }

    with patch("engine.strategy_manager.master_trade_gate", side_effect=fake_gate):
        result = manager.evaluate_all({"bias": {"direction": "LONG"}, "tick_pressure": {"pressure_score": 0.22}})

    assert result["strategy"] == "SMC_CONFLUENCE"
    assert result["signal"]["signal"] == "BUY"
    assert result["all_results"]["SMC_CONFLUENCE"]["arb_score"] > result["all_results"]["M15_SCALP_DEEP"]["arb_score"]


def test_auto_mode_returns_no_trade_when_all_candidates_fail_gate():
    manager = StrategyManager()
    manager.register(_StubStrategy("M15_SCALP_DEEP", {"signal": "BUY", "entry": 100.0, "confidence": 0.81, "rr": 1.4}))
    manager.register(_StubStrategy("SMC_CONFLUENCE", {"signal": "SELL", "entry": 100.0, "confidence": 0.77, "rr": 1.5}))

    def fake_gate(signal, market_state, risk_manager=None, m15_context_provider=None):
        return {
            "allowed": False,
            "reason": f"BLOCKED_{signal['_strategy_name']}",
            "score": 78,
            "confirmation_count": 3,
            "counter_trend": False,
        }

    with patch("engine.strategy_manager.master_trade_gate", side_effect=fake_gate):
        result = manager.evaluate_all({"bias": {"direction": "LONG"}})

    assert result["signal"]["signal"] == "NO_TRADE"
    assert "BLOCKED_" in result["signal"]["reason"]
    assert result["strategy"] in ("M15_SCALP_DEEP", "SMC_CONFLUENCE")


def test_multi_select_status_tracks_explicit_selection():
    manager = StrategyManager()
    manager.register(_StubStrategy("SMC_CONFLUENCE", {"signal": "NO_TRADE", "reason": "smc"}))
    manager.register(_StubStrategy("M15_SCALP_DEEP", {"signal": "NO_TRADE", "reason": "scalper"}))

    assert manager.set_selection(["SMC_CONFLUENCE", "M15_SCALP_DEEP"]) is True

    status = manager.status()
    assert status["is_auto"] is False
    assert status["selected"] == ["SMC_CONFLUENCE", "M15_SCALP_DEEP"]
    assert status["enabled"] == ["SMC_CONFLUENCE", "M15_SCALP_DEEP"]
    assert manager.active_name == "SMC_CONFLUENCE,M15_SCALP_DEEP"


def test_auto_selection_enables_all_registered_strategies():
    manager = StrategyManager()
    manager.register(_StubStrategy("SMC_CONFLUENCE", {"signal": "NO_TRADE", "reason": "smc"}))
    manager.register(_StubStrategy("M15_SCALP_DEEP", {"signal": "NO_TRADE", "reason": "scalper"}))

    assert manager.set_selection(["AUTO"]) is True

    status = manager.status()
    assert status["is_auto"] is True
    assert status["enabled"] == ["SMC_CONFLUENCE", "M15_SCALP_DEEP"]


def test_group_selection_accepts_explicit_strategy_names():
    manager = StrategyManager()
    manager.register(_StubStrategy("SMC_CONFLUENCE", {"signal": "NO_TRADE", "reason": "smc"}))
    manager.register(_StubStrategy("M15_ZONE_SCALP", {"signal": "NO_TRADE", "reason": "m15"}))
    manager.register(_StubStrategy("M15_SCALP_DEEP", {"signal": "NO_TRADE", "reason": "scalper"}))
    manager.register(_StubStrategy("TREND_CHANNEL", {"signal": "NO_TRADE", "reason": "trend"}))
    manager.register(_StubStrategy("SWING_ENGINE", {"signal": "NO_TRADE", "reason": "swing"}))

    assert manager.set_selection(["SWING_ENGINE", "SMC_CONFLUENCE", "M15_ZONE_SCALP", "M15_SCALP_DEEP", "TREND_CHANNEL"]) is True

    status = manager.status()
    assert status["selected"] == ["SWING_ENGINE", "SMC_CONFLUENCE", "M15_ZONE_SCALP", "M15_SCALP_DEEP", "TREND_CHANNEL"]
    assert status["enabled"] == [
        "SWING_ENGINE",
        "SMC_CONFLUENCE",
        "M15_ZONE_SCALP",
        "M15_SCALP_DEEP",
        "TREND_CHANNEL",
    ]
    assert manager.active_name == "SWING_ENGINE,SMC_CONFLUENCE,M15_ZONE_SCALP,M15_SCALP_DEEP,TREND_CHANNEL"


def test_set_active_accepts_direct_strategy_name():
    manager = StrategyManager()
    manager.register(_StubStrategy("SMC_CONFLUENCE", {"signal": "NO_TRADE", "reason": "smc"}))
    manager.register(_StubStrategy("M15_ZONE_SCALP", {"signal": "NO_TRADE", "reason": "m15"}))
    manager.register(_StubStrategy("TREND_CHANNEL", {"signal": "NO_TRADE", "reason": "trend"}))
    manager.register(_StubStrategy("SWING_ENGINE", {"signal": "NO_TRADE", "reason": "swing"}))

    assert manager.set_active("SWING_ENGINE") is True

    status = manager.status()
    assert status["selected"] == ["SWING_ENGINE"]
    assert status["enabled"] == [
        "SWING_ENGINE",
    ]


def test_generate_signal_keeps_skip_global_signals_untouched():
    manager = StrategyManager()
    manager.register(
        _StubStrategy(
            "SMC_CONFLUENCE",
            {"signal": "BUY", "entry": 100.0, "sl": 90.0, "tp": 130.0, "_skip_global_filters": True},
        )
    )

    result = manager.generate_signal({})

    assert result["strategy"] == "SMC_CONFLUENCE"
    assert result["signal"]["signal"] == "BUY"
    assert result["trade_id"] is None
