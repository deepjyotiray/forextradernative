from datetime import datetime, timezone

import pandas as pd

from engine.intraday_engine_strategy import IntradayEngineStrategy


def _make_frame(rows):
    return pd.DataFrame(rows)


def _base_data(spread: float = 0.30, last_volume: float = 45.0):
    m5 = _make_frame(
        [
            {"open": 99.40, "high": 99.80, "low": 99.20, "close": 99.60, "volume": 100.0},
            {"open": 99.50, "high": 99.90, "low": 99.30, "close": 99.70, "volume": 100.0},
            {"open": 99.55, "high": 99.95, "low": 99.35, "close": 99.75, "volume": 100.0},
            {"open": 99.60, "high": 100.00, "low": 99.40, "close": 99.80, "volume": 100.0},
            {"open": 99.70, "high": 100.10, "low": 99.00, "close": 99.85, "volume": 100.0},
            {"open": 99.55, "high": 100.20, "low": 99.50, "close": 100.15, "volume": last_volume},
        ]
    )
    higher_tf = _make_frame(
        [
            {"open": 99.0, "high": 100.0, "low": 98.8, "close": 99.8, "volume": 100.0},
            {"open": 99.2, "high": 100.2, "low": 99.0, "close": 100.0, "volume": 110.0},
        ]
    )
    return {
        "tick": {"bid": 100.00, "ask": round(100.00 + spread, 2), "spread": spread},
        "m5_df": m5,
        "m15_df": higher_tf,
        "h1_df": higher_tf,
        "now_utc": datetime(2026, 5, 7, 6, 30, tzinfo=timezone.utc),
        "calendar": {"blocked": False},
        "account": {"balance": 1000.0},
        "strategy_trade_counts": {"INTRADAY_ENGINE": 0},
        "market_state": {
            "trend": {"H1": "UP", "M15": "UP"},
            "asia_levels": {"asia_high": 101.0, "asia_low": 99.0},
            "prev_day": {"pdh": 101.5, "pdl": 99.0},
            "equal_lows": [99.0],
            "equal_highs": [],
        },
        "liquidity": {"sweeps": [{"type": "BUY_SWEEP", "level": 99.0}]},
        "correlation": {"dxy": {"trend": "DOWN"}, "us10y": {"trend": "DOWN"}},
    }


def test_intraday_engine_uses_configurable_volume_ratio_threshold(monkeypatch):
    strategy = IntradayEngineStrategy()
    data = _base_data(last_volume=45.0)

    monkeypatch.setattr(
        "engine.intraday_engine_strategy._scfg.get",
        lambda _: {
            "sessions": ["ASIAN", "LONDON", "NEW_YORK"],
            "news_block": True,
            "spread_max": 0.35,
            "m5_volume_ratio_min": 0.45,
            "min_confidence_pct": 85,
            "max_active_trades": 5,
            "risk_pct": 0.5,
            "sl_min": 2.5,
            "sl_max": 6.0,
            "tp1_r": 1.5,
            "tp2_r": 2.0,
        },
    )

    result = strategy.generate_signal(data)

    assert result["signal"] == "BUY"


def test_intraday_engine_blocks_when_volume_ratio_below_configured_threshold(monkeypatch):
    strategy = IntradayEngineStrategy()
    data = _base_data(last_volume=45.0)

    monkeypatch.setattr(
        "engine.intraday_engine_strategy._scfg.get",
        lambda _: {
            "sessions": ["ASIAN", "LONDON", "NEW_YORK"],
            "news_block": True,
            "spread_max": 0.35,
            "m5_volume_ratio_min": 0.60,
            "min_confidence_pct": 85,
            "max_active_trades": 5,
            "risk_pct": 0.5,
            "sl_min": 2.5,
            "sl_max": 6.0,
            "tp1_r": 1.5,
            "tp2_r": 2.0,
        },
    )

    result = strategy.generate_signal(data)

    assert result["signal"] == "NO_TRADE"
    assert "M5 volume ratio 0.45 < 0.6" in result["reason"]


def test_intraday_engine_blocks_when_confidence_below_configured_threshold(monkeypatch):
    strategy = IntradayEngineStrategy()
    data = _base_data(last_volume=45.0)
    data["correlation"] = {}

    monkeypatch.setattr(
        "engine.intraday_engine_strategy._scfg.get",
        lambda _: {
            "sessions": ["ASIAN", "LONDON", "NEW_YORK"],
            "news_block": True,
            "spread_max": 0.35,
            "m5_volume_ratio_min": 0.45,
            "min_confidence_pct": 85,
            "max_active_trades": 5,
            "risk_pct": 0.5,
            "sl_min": 2.5,
            "sl_max": 6.0,
            "tp1_r": 1.5,
            "tp2_r": 2.0,
        },
    )

    result = strategy.generate_signal(data)

    assert result["signal"] == "NO_TRADE"
    assert result["reason"] == "Confidence 80 < 85"
