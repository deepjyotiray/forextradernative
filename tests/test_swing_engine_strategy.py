from datetime import datetime, timezone

import pandas as pd

from engine.swing_engine_strategy import SwingEngineStrategy


def _frame(rows):
    return pd.DataFrame(rows)


def _swing_data():
    h4 = _frame(
        [
            {"open": 100.0, "high": 104.0, "low": 99.0, "close": 103.0},
            {"open": 103.0, "high": 105.0, "low": 101.0, "close": 104.0},
            {"open": 104.0, "high": 106.0, "low": 102.0, "close": 105.0},
            {"open": 105.0, "high": 108.0, "low": 103.0, "close": 107.8},
        ]
    )
    d1 = _frame(
        [
            {"open": 95.0, "high": 100.0, "low": 94.0, "close": 99.0},
            {"open": 99.0, "high": 103.0, "low": 98.0, "close": 102.0},
            {"open": 102.0, "high": 106.0, "low": 101.0, "close": 105.0},
            {"open": 105.0, "high": 109.0, "low": 104.0, "close": 108.0},
        ]
    )
    return {
        "tick": {"bid": 108.0, "ask": 108.2, "spread": 0.35},
        "h4_df": h4,
        "d1_df": d1,
        "account": {"balance": 1000.0},
        "strategy_trade_counts": {"SWING_ENGINE": 0},
        "market_state": {
            "trend": {"H1": "UP", "H4": "UP", "D1": "UP"},
            "structure": {"H1": "UP", "H4": "UP", "D1": "UP"},
            "prev_day": {"pdh": 110.0, "pdl": 96.0},
            "weekly_range": {"high_7d": 115.0, "low_7d": 95.0, "mid_7d": 105.0},
            "atr": {"H4": 10.0},
            "pullback_depth": 0.40,
        },
        "zones": {},
        "calendar": {"blocked": False},
        "correlation": {},
    }


def test_swing_engine_blocks_when_confidence_below_configured_threshold(monkeypatch):
    strategy = SwingEngineStrategy()
    data = _swing_data()

    monkeypatch.setattr("engine.swing_engine_strategy.sl_streak_guard.cooldown_remaining", lambda _name: 0.0)
    monkeypatch.setattr("engine.swing_engine_strategy.sl_streak_guard.is_strict_active", lambda _name: False)
    monkeypatch.setattr(
        "engine.swing_engine_strategy._scfg.get",
        lambda _: {
            "spread_max": 0.7,
            "max_active_trades": 1,
            "min_confidence_pct": 85,
            "sl_min": 14.0,
            "sl_max": 32.0,
            "risk_pct": 1.0,
            "tp1_r": 0.25,
            "tp2_r": 0.35,
            "tp3_r": 0.5,
        },
    )

    result = strategy.generate_signal(data)

    assert result["signal"] == "NO_TRADE"
    assert result["reason"] == "Confidence 80 < 85"


def test_swing_engine_passes_when_macro_boost_reaches_threshold(monkeypatch):
    strategy = SwingEngineStrategy()
    data = _swing_data()
    data["correlation"] = {"dxy": {"trend": "DOWN"}, "us10y": {"trend": "DOWN"}}

    monkeypatch.setattr("engine.swing_engine_strategy.sl_streak_guard.cooldown_remaining", lambda _name: 0.0)
    monkeypatch.setattr("engine.swing_engine_strategy.sl_streak_guard.is_strict_active", lambda _name: False)
    monkeypatch.setattr(
        "engine.swing_engine_strategy._scfg.get",
        lambda _: {
            "spread_max": 0.7,
            "max_active_trades": 1,
            "min_confidence_pct": 85,
            "sl_min": 14.0,
            "sl_max": 32.0,
            "risk_pct": 1.0,
            "tp1_r": 0.25,
            "tp2_r": 0.35,
            "tp3_r": 0.5,
        },
    )

    result = strategy.generate_signal(data)

    assert result["signal"] == "BUY"
    assert result["confidence_pct"] == 90
    assert result["sl_distance"] >= 14.0
