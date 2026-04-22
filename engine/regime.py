"""
Market Regime Engine — classifies market state.
States: TRENDING, RANGING, EXPANSION, NEWS_VOLATILITY
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
from .indicators import ema, atr


def classify_regime(h4_df: pd.DataFrame, h1_df: pd.DataFrame,
                    m15_df: pd.DataFrame, tick_snap: Dict) -> Dict:
    """
    Classify market regime from multi-TF data.
    Returns: {state, direction, trade_allowed, allowed_strategies, reasons}
    """
    reasons = []

    # --- ATR expansion/contraction (H1) ---
    atr_ratio = 1.0
    if h1_df is not None and len(h1_df) >= 30:
        c = h1_df["close"].values.astype(float)
        h = h1_df["high"].values.astype(float)
        l = h1_df["low"].values.astype(float)
        atr7 = atr(h, l, c, 7)
        atr20 = atr(h, l, c, 20)
        atr_ratio = atr7[-1] / atr20[-1] if atr20[-1] > 0 else 1.0

    # --- EMA slope (H4 macro direction) ---
    h4_slope = 0.0
    h4_direction = "NEUTRAL"
    if h4_df is not None and len(h4_df) >= 55:
        c4 = h4_df["close"].values.astype(float)
        ema50_h4 = ema(c4, 50)
        h4_slope = ema50_h4[-1] - ema50_h4[-3] if len(ema50_h4) >= 3 else 0
        if h4_slope > 0.1:
            h4_direction = "LONG"
        elif h4_slope < -0.1:
            h4_direction = "SHORT"

    # --- Price compression (M15 range vs ATR) ---
    compressed = False
    if m15_df is not None and len(m15_df) >= 20:
        recent = m15_df.iloc[-10:]
        range_width = recent["high"].max() - recent["low"].min()
        c15 = m15_df["close"].values.astype(float)
        h15 = m15_df["high"].values.astype(float)
        l15 = m15_df["low"].values.astype(float)
        atr14_m15 = atr(h15, l15, c15, 14)
        if atr14_m15[-1] > 0 and range_width < atr14_m15[-1] * 3:
            compressed = True

    # --- Tick chaos (news detection) ---
    tick_chaotic = False
    if tick_snap and tick_snap.get("ready"):
        if tick_snap["velocity"] > 50 and not tick_snap["spread_stable"]:
            tick_chaotic = True
        if tick_snap.get("max_spread", 0) > 1.0:
            tick_chaotic = True

    # --- Classification ---
    if tick_chaotic or atr_ratio > 3.0:
        return {
            "state": "NEWS_VOLATILITY",
            "direction": None,
            "trade_allowed": False,
            "allowed_strategies": [],
            "reasons": [f"ATR ratio {atr_ratio:.1f}x" if atr_ratio > 2.5 else "Tick chaos detected"],
            "atr_ratio": round(atr_ratio, 2),
            "h4_slope": round(h4_slope, 3),
        }

    if compressed and atr_ratio > 1.3:
        return {
            "state": "EXPANSION",
            "direction": h4_direction,
            "trade_allowed": True,
            "allowed_strategies": ["breakout"],
            "reasons": [f"Compression + ATR expanding ({atr_ratio:.1f}x)", f"H4 bias: {h4_direction}"],
            "atr_ratio": round(atr_ratio, 2),
            "h4_slope": round(h4_slope, 3),
        }

    if atr_ratio < 0.7 or (compressed and atr_ratio <= 1.0):
        return {
            "state": "RANGING",
            "direction": None,
            "trade_allowed": True,
            "allowed_strategies": ["mean_reversion"],
            "reasons": [f"Low ATR ratio {atr_ratio:.1f}x", "Price compressed" if compressed else "Low volatility"],
            "atr_ratio": round(atr_ratio, 2),
            "h4_slope": round(h4_slope, 3),
        }

    return {
        "state": "TRENDING",
        "direction": h4_direction,
        "trade_allowed": True,
        "allowed_strategies": ["trend"],
        "reasons": [f"ATR ratio {atr_ratio:.1f}x", f"H4 slope: {h4_slope:+.2f} ({h4_direction})"],
        "atr_ratio": round(atr_ratio, 2),
        "h4_slope": round(h4_slope, 3),
    }
