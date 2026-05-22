from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from .config_adapter import safe_float


def _resolve_last_price(market_data: Dict[str, Any]) -> float:
    tick = market_data.get("tick") or {}
    return safe_float(tick.get("bid") or tick.get("ask") or 0.0, 0.0)


def _volume_weighted_average_price(df: Optional[pd.DataFrame]) -> float | None:
    if df is None or len(df) < 3:
        return None
    volume_col = None
    for candidate in ("real_volume", "tick_volume", "volume"):
        if candidate in df.columns:
            volume_col = candidate
            break
    if volume_col is None:
        return None
    window = df.tail(30).copy()
    vol = window[volume_col].astype(float)
    if float(vol.sum()) <= 0:
        return None
    typical = (window["high"].astype(float) + window["low"].astype(float) + window["close"].astype(float)) / 3.0
    return round(float((typical * vol).sum() / vol.sum()), 4)


def _price_relation(price: float, level: float | None, tolerance: float = 0.05) -> str:
    if level is None or level <= 0:
        return "UNKNOWN"
    if price > level + tolerance:
        return "ABOVE"
    if price < level - tolerance:
        return "BELOW"
    return "AT_LEVEL"


def _nearest_zone(zones: Dict[str, Any], side: str) -> Dict[str, Any]:
    rows = list((zones or {}).get(side, []) or [])
    return dict(rows[0]) if rows else {}


def build_market_context(signal: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
    indicators = market_data.get("indicators") or {}
    market_state = market_data.get("market_state") or {}
    ema_state = market_state.get("ema") or {}
    trend = market_state.get("trend") or {}
    regime = market_data.get("regime") or {}
    liquidity = market_data.get("liquidity") or {}
    zones = market_data.get("zones") or {}
    tick_pressure = market_data.get("tick_pressure") or {}
    m1_df = market_data.get("m1_df")
    m5_df = market_data.get("m5_df")
    m15_df = market_data.get("m15_df")
    price = _resolve_last_price(market_data)
    vwap = _volume_weighted_average_price(m1_df) or _volume_weighted_average_price(m5_df)

    ema9 = safe_float(indicators.get("ema9"), 0.0)
    ema20 = safe_float(indicators.get("ema20"), 0.0)
    ema20_slope = safe_float(indicators.get("ema20_slope"), safe_float((ema_state.get("M1_20") or {}).get("slope"), 0.0))

    recent_high = safe_float((m15_df["high"].tail(10).max() if m15_df is not None and len(m15_df) >= 3 else 0.0), 0.0)
    recent_low = safe_float((m15_df["low"].tail(10).min() if m15_df is not None and len(m15_df) >= 3 else 0.0), 0.0)
    support = _nearest_zone(zones, "support")
    resistance = _nearest_zone(zones, "resistance")

    body_ratio = safe_float(indicators.get("body_ratio"), 0.0)
    candle_confirmation = bool(signal.get("_candle_confirmation"))
    rejection_present = bool(signal.get("_bullish_reaction")) or bool(signal.get("_bearish_reaction")) or body_ratio >= 0.55

    current_spread = safe_float((market_data.get("tick") or {}).get("spread"), 0.0)
    spread_before_entry = safe_float((market_data.get("tick_snapshot") or {}).get("spread_mean"), current_spread)
    compression_range = safe_float(indicators.get("range_10"), 0.0)
    atr_value = safe_float(indicators.get("atr14", indicators.get("atr")), 0.0)
    compression_state = "COMPRESSED" if atr_value > 0 and compression_range <= atr_value * 2.6 else "NORMAL"

    abnormal_candle = bool(safe_float(indicators.get("candle_range"), 0.0) > atr_value * 2.0) if atr_value > 0 else False

    return {
        "symbol": market_data.get("symbol") or "",
        "session": signal.get("_session") or market_data.get("session") or "",
        "market_state": regime.get("state") or "",
        "trend": {
            "D1": trend.get("D1"),
            "H4": trend.get("H4"),
            "H1": trend.get("H1"),
            "M15": trend.get("M15"),
            "M5": trend.get("M5"),
            "M1": trend.get("M1"),
        },
        "atr": atr_value,
        "spread_at_signal": current_spread,
        "spread_before_entry": spread_before_entry,
        "price_vs_vwap": _price_relation(price, vwap),
        "vwap": vwap,
        "price_vs_ema9": _price_relation(price, ema9 if ema9 > 0 else None),
        "price_vs_ema20": _price_relation(price, ema20 if ema20 > 0 else None),
        "ema20_slope": ema20_slope,
        "recent_high": round(recent_high, 2) if recent_high else 0.0,
        "recent_low": round(recent_low, 2) if recent_low else 0.0,
        "nearest_support_zone": support,
        "nearest_resistance_zone": resistance,
        "liquidity_sweep": bool(liquidity.get("recent_buy_sweep") or liquidity.get("recent_sell_sweep")),
        "rejection_candle": rejection_present,
        "candle_confirmation": candle_confirmation,
        "tick_pressure": {
            "ready": bool(tick_pressure.get("ready")),
            "directional_bias": tick_pressure.get("directional_bias"),
            "pressure_score": tick_pressure.get("pressure_score"),
            "burst_rate": tick_pressure.get("burst_rate"),
        },
        "volume_state": "HIGH" if safe_float(signal.get("_entry_volume_ratio"), 0.0) >= 1.2 else "NORMAL",
        "compression_state": compression_state,
        "abnormal_candle_state": "SPIKE" if abnormal_candle else "NORMAL",
        "news_state": "BLOCKED" if bool((market_data.get("calendar") or {}).get("blocked")) else "CLEAR",
    }
