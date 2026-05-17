from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from engine.smc_strategy import SMCStrategy


def _cfg(**overrides):
    base = {
        "spread_max": 0.6,
        "sessions": ["LONDON", "NEW_YORK"],
        "news_block": False,
        "max_active_trades": 1,
        "htf_timeframes": ["M30", "M15"],
        "ltf_timeframes": ["M5", "M1"],
        "min_htf_bars": 12,
        "min_ltf_bars": 8,
        "sweep_lookback": 12,
        "structure_lookback": 6,
        "max_bars_after_sweep": 6,
        "fvg_lookback": 12,
        "ifvg_recent_bars": 4,
        "ifvg_confirm_body_ratio": 0.45,
        "ifvg_overlap_tolerance": 0.2,
        "ifvg_overlap_atr_mult": 0.0,
        "ob_retest_tolerance": 0.25,
        "ob_retest_atr_mult": 0.0,
        "swing_lookback": 5,
        "sl_buffer_points": 0.2,
        "sl_min_points": 0.6,
        "sl_max_points": 6.0,
        "target_rr": 2.0,
        "min_rr": 1.5,
    }
    base.update(overrides)
    return base


def _htf_long_setup_df() -> pd.DataFrame:
    rows = [
        {"open": 100.3, "high": 100.8, "low": 100.0, "close": 100.5},
        {"open": 100.5, "high": 101.0, "low": 100.2, "close": 100.7},
        {"open": 100.7, "high": 101.2, "low": 100.4, "close": 100.9},
        {"open": 100.9, "high": 101.1, "low": 100.1, "close": 100.6},
        {"open": 100.6, "high": 101.3, "low": 100.4, "close": 101.0},
        {"open": 101.0, "high": 101.4, "low": 100.6, "close": 100.8},
        {"open": 100.8, "high": 101.0, "low": 100.2, "close": 100.4},
        {"open": 100.4, "high": 100.8, "low": 100.1, "close": 100.3},
        {"open": 99.8, "high": 101.0, "low": 99.2, "close": 100.8},
        {"open": 101.0, "high": 101.2, "low": 100.5, "close": 100.7},
        {"open": 100.7, "high": 102.0, "low": 100.6, "close": 101.8},
        {"open": 101.8, "high": 102.1, "low": 101.1, "close": 101.5},
        {"open": 101.5, "high": 101.8, "low": 100.7, "close": 101.0},
        {"open": 101.0, "high": 101.1, "low": 100.4, "close": 100.9},
    ]
    return pd.DataFrame(rows)


def _trend_ltf_df() -> pd.DataFrame:
    rows = []
    price = 100.6
    for _ in range(10):
        open_px = price
        close_px = price + 0.12
        rows.append(
            {
                "open": round(open_px, 2),
                "high": round(close_px + 0.18, 2),
                "low": round(open_px - 0.18, 2),
                "close": round(close_px, 2),
            }
        )
        price = close_px
    return pd.DataFrame(rows)


def test_smc_strategy_uses_canonical_strategy_name():
    strategy = SMCStrategy()
    assert strategy.name == "SMC_CONFLUENCE"


def test_smc_strategy_generates_buy_from_strategy1_model():
    strategy = SMCStrategy()
    ltf_df = _trend_ltf_df()
    setup = {
        "direction": "LONG",
        "sweep_idx": 8,
        "bos_idx": 10,
        "bos_level": 101.4,
        "order_block": {"idx": 9, "zone_low": 100.5, "zone_high": 101.0},
    }
    ifvg = {
        "label": "bearish FVG reclaimed -> bullish IFVG",
        "zone_low": 100.6,
        "zone_high": 100.95,
        "confirm_idx": len(ltf_df) - 1,
    }
    data = {
        "tick": {"bid": 100.92, "ask": 101.02, "spread": 0.2},
        "m30_df": _htf_long_setup_df(),
        "m15_df": _htf_long_setup_df(),
        "m5_df": ltf_df,
        "m1_df": ltf_df,
        "liquidity": {"key_levels": {"session_high": 102.6}},
        "bias": {"direction": "LONG"},
        "calendar": {},
        "now_utc": datetime(2026, 5, 18, 8, 0, tzinfo=timezone.utc),
        "strategy_trade_counts": {"SMC_CONFLUENCE": 0},
    }

    with patch("engine.liquidity_sweep_ob_strategy._scfg.get", return_value=_cfg()):
        with patch.object(strategy, "_find_recent_htf_setup", return_value=setup):
            with patch.object(strategy, "_find_ifvg_confirmation", return_value=ifvg):
                signal = strategy.generate_signal(data)

    assert signal["signal"] == "BUY"
    assert signal["_strategy_name"] == "SMC_CONFLUENCE"
    assert signal["_signal_family"] == "SMC"
    assert signal["rr"] >= 1.5
