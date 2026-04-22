"""
Liquidity & Smart Money Engine.
A. Liquidity sweep detection (equal highs/lows + stop hunts)
B. Order block zones (last candle before impulse)
C. Fair value gaps (FVG)
D. Session highs/lows, previous day H/L
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime, timezone


def detect_liquidity_sweeps(df: pd.DataFrame, lookback: int = 50) -> List[Dict]:
    """Detect stop hunts: wick beyond equal highs/lows then reversal."""
    if len(df) < lookback:
        return []
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    o = df["open"].values.astype(float)
    sweeps = []
    start = max(0, len(df) - lookback)

    # Find equal highs/lows (within 0.5 tolerance for XAUUSD)
    tol = 0.5
    for i in range(start + 5, len(df) - 1):
        # Check for equal highs in prior candles
        for j in range(max(start, i - 20), i - 2):
            if abs(h[j] - h[i-1]) < tol:  # equal highs
                # Current candle wicks above then closes below
                if h[i] > h[j] + tol and c[i] < h[j] and c[i] < o[i]:
                    sweeps.append({
                        "type": "SELL_SWEEP", "level": round(float(h[j]), 2),
                        "wick_high": round(float(h[i]), 2), "idx": i,
                    })
                    break
            if abs(l[j] - l[i-1]) < tol:  # equal lows
                if l[i] < l[j] - tol and c[i] > l[j] and c[i] > o[i]:
                    sweeps.append({
                        "type": "BUY_SWEEP", "level": round(float(l[j]), 2),
                        "wick_low": round(float(l[i]), 2), "idx": i,
                    })
                    break
    return sweeps[-5:]  # last 5


def detect_order_blocks(df: pd.DataFrame, lookback: int = 80) -> List[Dict]:
    """Last bullish/bearish candle before a strong impulse move."""
    if len(df) < 20:
        return []
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    o = df["open"].values.astype(float)
    blocks = []
    start = max(0, len(df) - lookback)
    avg_range = np.mean(h[start:] - l[start:])
    if avg_range <= 0:
        return []

    for i in range(start + 1, len(df) - 1):
        move = c[i+1] - c[i]
        # Bullish OB: bearish candle followed by strong bullish impulse
        if c[i] < o[i] and move > avg_range * 1.5:
            blocks.append({
                "type": "BULLISH_OB",
                "zone_low": round(float(l[i]), 2),
                "zone_high": round(float(max(o[i], c[i])), 2),
                "idx": i, "strength": round(move / avg_range, 2),
            })
        # Bearish OB: bullish candle followed by strong bearish impulse
        if c[i] > o[i] and move < -avg_range * 1.5:
            blocks.append({
                "type": "BEARISH_OB",
                "zone_low": round(float(min(o[i], c[i])), 2),
                "zone_high": round(float(h[i]), 2),
                "idx": i, "strength": round(abs(move) / avg_range, 2),
            })
    return blocks[-10:]


def detect_fvg(df: pd.DataFrame, lookback: int = 50) -> List[Dict]:
    """Fair Value Gaps — 3-candle imbalance zones."""
    if len(df) < 5:
        return []
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    gaps = []
    start = max(0, len(df) - lookback)

    for i in range(start + 2, len(df)):
        # Bullish FVG: candle[i-2] high < candle[i] low (gap up)
        if l[i] > h[i-2]:
            gaps.append({
                "type": "BULLISH_FVG",
                "zone_low": round(float(h[i-2]), 2),
                "zone_high": round(float(l[i]), 2),
                "idx": i,
            })
        # Bearish FVG: candle[i-2] low > candle[i] high (gap down)
        if h[i] < l[i-2]:
            gaps.append({
                "type": "BEARISH_FVG",
                "zone_low": round(float(h[i]), 2),
                "zone_high": round(float(l[i-2]), 2),
                "idx": i,
            })
    return gaps[-10:]


def detect_key_levels(h1_df: pd.DataFrame, d1_df: pd.DataFrame = None) -> Dict:
    """Session highs/lows + previous day H/L."""
    levels = {}
    if h1_df is not None and len(h1_df) >= 24:
        # Last 24 H1 candles ~ 1 day
        recent = h1_df.iloc[-24:]
        levels["session_high"] = round(float(recent["high"].max()), 2)
        levels["session_low"] = round(float(recent["low"].min()), 2)

    if d1_df is not None and len(d1_df) >= 2:
        prev = d1_df.iloc[-2]
        levels["prev_day_high"] = round(float(prev["high"]), 2)
        levels["prev_day_low"] = round(float(prev["low"]), 2)

    return levels


def compute_liquidity(m5_df: pd.DataFrame, m15_df: pd.DataFrame,
                      h1_df: pd.DataFrame, d1_df: pd.DataFrame = None) -> Dict:
    """Full liquidity analysis."""
    sweeps_m5 = detect_liquidity_sweeps(m5_df) if m5_df is not None else []
    sweeps_m15 = detect_liquidity_sweeps(m15_df) if m15_df is not None else []
    obs = detect_order_blocks(m15_df) if m15_df is not None else []
    fvgs = detect_fvg(m15_df) if m15_df is not None else []
    levels = detect_key_levels(h1_df, d1_df)

    return {
        "sweeps": sweeps_m5 + sweeps_m15,
        "order_blocks": obs,
        "fvg": fvgs,
        "key_levels": levels,
        "recent_buy_sweep": any(s["type"] == "BUY_SWEEP" and s["idx"] >= (len(m5_df) - 5) for s in sweeps_m5) if sweeps_m5 and m5_df is not None else False,
        "recent_sell_sweep": any(s["type"] == "SELL_SWEEP" and s["idx"] >= (len(m5_df) - 5) for s in sweeps_m5) if sweeps_m5 and m5_df is not None else False,
    }
