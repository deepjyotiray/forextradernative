"""
Multi-Timeframe Bias Engine.
H4: EMA50/200 macro bias + slope
H1: Structure (HH, HL, LH, LL) + BOS detection
M15: Pullback zone identification
Output: direction (LONG/SHORT/NEUTRAL), confidence 0-1
"""
import numpy as np
import pandas as pd
from typing import Dict
from .indicators import ema, atr


def _swing_points(h: np.ndarray, l: np.ndarray, window: int = 3):
    highs, lows = [], []
    for i in range(window, len(h) - window):
        if all(h[i] >= h[i-j] for j in range(1, window+1)) and \
           all(h[i] >= h[i+j] for j in range(1, window+1)):
            highs.append((i, float(h[i])))
        if all(l[i] <= l[i-j] for j in range(1, window+1)) and \
           all(l[i] <= l[i+j] for j in range(1, window+1)):
            lows.append((i, float(l[i])))
    return highs, lows


def _detect_structure(highs: list, lows: list) -> Dict:
    if len(highs) < 2 or len(lows) < 2:
        return {"pattern": "UNKNOWN", "bos": False, "bos_direction": None}
    h1, h2 = highs[-2][1], highs[-1][1]
    l1, l2 = lows[-2][1], lows[-1][1]
    hh, hl = h2 > h1, l2 > l1
    lh, ll = h2 < h1, l2 < l1
    if hh and hl:
        return {"pattern": "BULLISH", "bos": True, "bos_direction": "LONG", "detail": "HH+HL"}
    if lh and ll:
        return {"pattern": "BEARISH", "bos": True, "bos_direction": "SHORT", "detail": "LH+LL"}
    if hh and ll:
        return {"pattern": "EXPANSION", "bos": False, "bos_direction": None, "detail": "HH+LL"}
    if lh and hl:
        return {"pattern": "COMPRESSION", "bos": False, "bos_direction": None, "detail": "LH+HL"}
    return {"pattern": "MIXED", "bos": False, "bos_direction": None}


def compute_bias(h4_df: pd.DataFrame, h1_df: pd.DataFrame,
                 m15_df: pd.DataFrame) -> Dict:
    reasons = []
    scores = {"LONG": 0.0, "SHORT": 0.0}

    # === H4 Macro Bias ===
    h4_bias = "NEUTRAL"
    if h4_df is not None and len(h4_df) >= 55:
        c = h4_df["close"].values.astype(float)
        ema50 = ema(c, 50)
        slope50 = ema50[-1] - ema50[-5] if len(ema50) >= 5 else 0

        # EMA50/200 alignment if enough data
        if len(c) >= 200:
            ema200 = ema(c, 200)
            if ema50[-1] > ema200[-1]:
                h4_bias = "LONG"
                scores["LONG"] += 0.20
                reasons.append(f"H4: EMA50 > EMA200")
            elif ema50[-1] < ema200[-1]:
                h4_bias = "SHORT"
                scores["SHORT"] += 0.20
                reasons.append(f"H4: EMA50 < EMA200")

        # Slope adds to bias (lower threshold: 0.1 instead of 0.5)
        if slope50 > 0.1:
            if h4_bias != "SHORT":
                h4_bias = "LONG"
            scores["LONG"] += min(0.20, abs(slope50) * 0.1)
            reasons.append(f"H4 slope +{slope50:.2f}")
        elif slope50 < -0.1:
            if h4_bias != "LONG":
                h4_bias = "SHORT"
            scores["SHORT"] += min(0.20, abs(slope50) * 0.1)
            reasons.append(f"H4 slope {slope50:.2f}")

        # Price vs EMA50 — simple but effective
        if c[-1] > ema50[-1]:
            scores["LONG"] += 0.10
        elif c[-1] < ema50[-1]:
            scores["SHORT"] += 0.10

    # === H1 Structure ===
    h1_struct = {"pattern": "UNKNOWN", "bos": False, "bos_direction": None}
    if h1_df is not None and len(h1_df) >= 30:
        h = h1_df["high"].values.astype(float)
        l = h1_df["low"].values.astype(float)
        c1 = h1_df["close"].values.astype(float)
        highs, lows = _swing_points(h, l, window=3)
        h1_struct = _detect_structure(highs, lows)

        if h1_struct["bos"]:
            d = h1_struct["bos_direction"]
            scores[d] += 0.30
            reasons.append(f"H1: BOS {d} ({h1_struct.get('detail','')})")
        elif h1_struct["pattern"] == "BULLISH":
            scores["LONG"] += 0.15
            reasons.append("H1: Bullish structure")
        elif h1_struct["pattern"] == "BEARISH":
            scores["SHORT"] += 0.15
            reasons.append("H1: Bearish structure")
        elif h1_struct["pattern"] == "EXPANSION":
            # Expansion — use recent price direction
            if len(c1) >= 5:
                recent_move = c1[-1] - c1[-5]
                if recent_move > 0:
                    scores["LONG"] += 0.10
                    reasons.append("H1: Expansion (recent bullish)")
                elif recent_move < 0:
                    scores["SHORT"] += 0.10
                    reasons.append("H1: Expansion (recent bearish)")

        # Additional: EMA20 direction on H1
        if len(c1) >= 25:
            ema20_h1 = ema(c1, 20)
            h1_slope = ema20_h1[-1] - ema20_h1[-3]
            if h1_slope > 0.3:
                scores["LONG"] += 0.10
                reasons.append(f"H1 EMA20 slope +{h1_slope:.2f}")
            elif h1_slope < -0.3:
                scores["SHORT"] += 0.10
                reasons.append(f"H1 EMA20 slope {h1_slope:.2f}")

    # === M15 Pullback Detection ===
    m15_pullback = {"active": False, "direction": None}
    if m15_df is not None and len(m15_df) >= 30:
        c = m15_df["close"].values.astype(float)
        h = m15_df["high"].values.astype(float)
        l = m15_df["low"].values.astype(float)
        ema20 = ema(c, 20)
        atr14 = atr(h, l, c, 14)
        price = c[-1]
        ema_val = ema20[-1]
        atr_val = atr14[-1]

        # Pullback to EMA in any direction with bias
        near_ema = abs(price - ema_val) < atr_val * 0.8
        if near_ema:
            if scores["LONG"] > scores["SHORT"] and price >= ema_val * 0.998:
                m15_pullback = {"active": True, "direction": "LONG"}
                scores["LONG"] += 0.15
                reasons.append("M15: Pullback to EMA20 (bullish)")
            elif scores["SHORT"] > scores["LONG"] and price <= ema_val * 1.002:
                m15_pullback = {"active": True, "direction": "SHORT"}
                scores["SHORT"] += 0.15
                reasons.append("M15: Pullback to EMA20 (bearish)")

    # === Final ===
    long_s, short_s = scores["LONG"], scores["SHORT"]
    total = long_s + short_s
    if total < 0.10:
        direction = "NEUTRAL"
        confidence = 0.0
    elif long_s > short_s:
        direction = "LONG"
        confidence = round(min(1.0, long_s), 3)
    else:
        direction = "SHORT"
        confidence = round(min(1.0, short_s), 3)

    return {
        "direction": direction,
        "confidence": confidence,
        "h4_bias": h4_bias,
        "h1_structure": h1_struct,
        "m15_pullback": m15_pullback,
        "scores": scores,
        "reasons": reasons,
    }
