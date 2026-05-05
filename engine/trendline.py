"""
Trendline Engine — swing detection, market structure, trendline construction.
Reuses atr() from indicators.py. Accepts DataFrames from mt5_bridge.fetch_candles.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from .indicators import atr as _atr


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_trendlines(df: pd.DataFrame, symbol: str = "XAUUSD", timeframe: str = "M5") -> Dict:
    """
    Run full trendline pipeline on a candle DataFrame.
    Columns required: open, high, low, close (volume optional).
    Returns machine-consumable dict matching the prompt schema.
    """
    if df is None or len(df) < 10:
        return {"error": "insufficient_data"}

    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)

    atr14 = _atr(h, l, c, 14)
    noise_floor = atr14[-1] * 0.15  # ATR-based noise threshold

    swing_highs = _detect_swing_highs(h, c, noise_floor)
    swing_lows  = _detect_swing_lows(l, c, noise_floor)

    structure   = _classify_structure(swing_highs, swing_lows)
    trend       = _classify_trend(structure)

    trendlines  = _build_trendlines(swing_highs, swing_lows, trend, h, l, c)
    channels    = _detect_channels(trendlines)

    return {
        "symbol":      symbol,
        "timeframe":   timeframe,
        "trend":       trend,
        "swing_highs": swing_highs,
        "swing_lows":  swing_lows,
        "structure":   structure,
        "trendlines":  trendlines,
        "channels":    channels,
    }


# ---------------------------------------------------------------------------
# Swing detection
# ---------------------------------------------------------------------------

def _detect_swing_highs(h: np.ndarray, c: np.ndarray, noise_floor: float) -> List[Dict]:
    out = []
    for i in range(1, len(h) - 1):
        if h[i] > h[i - 1] and h[i] > h[i + 1]:
            if abs(h[i] - c[i]) >= noise_floor or (h[i] - h[i - 1]) >= noise_floor:
                out.append({"index": i, "price": round(float(h[i]), 5)})
    return out


def _detect_swing_lows(l: np.ndarray, c: np.ndarray, noise_floor: float) -> List[Dict]:
    out = []
    for i in range(1, len(l) - 1):
        if l[i] < l[i - 1] and l[i] < l[i + 1]:
            if abs(c[i] - l[i]) >= noise_floor or (l[i - 1] - l[i]) >= noise_floor:
                out.append({"index": i, "price": round(float(l[i]), 5)})
    return out


# ---------------------------------------------------------------------------
# Market structure
# ---------------------------------------------------------------------------

def _classify_structure(swing_highs: List[Dict], swing_lows: List[Dict]) -> List[Dict]:
    labels = []
    for i in range(1, len(swing_highs)):
        cur, prev = swing_highs[i], swing_highs[i - 1]
        tag = "HH" if cur["price"] > prev["price"] else "LH"
        labels.append({"index": cur["index"], "type": tag, "price": cur["price"]})
    for i in range(1, len(swing_lows)):
        cur, prev = swing_lows[i], swing_lows[i - 1]
        tag = "HL" if cur["price"] > prev["price"] else "LL"
        labels.append({"index": cur["index"], "type": tag, "price": cur["price"]})
    labels.sort(key=lambda x: x["index"])
    return labels


def _classify_trend(structure: List[Dict]) -> str:
    if len(structure) < 2:
        return "range"
    types = [s["type"] for s in structure[-6:]]  # last 6 labels
    up_count   = sum(1 for t in types if t in ("HH", "HL"))
    down_count = sum(1 for t in types if t in ("LH", "LL"))
    if up_count >= 3 and up_count > down_count:
        return "uptrend"
    if down_count >= 3 and down_count > up_count:
        return "downtrend"
    return "range"


# ---------------------------------------------------------------------------
# Trendline construction
# ---------------------------------------------------------------------------

def _build_trendlines(
    swing_highs: List[Dict],
    swing_lows: List[Dict],
    trend: str,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
) -> List[Dict]:
    lines = []
    n = len(c)
    # Only use swing points from the recent half of the data
    # so trendlines are anchored to current market structure
    recent_cutoff = max(0, n - n // 2)
    recent_highs = [s for s in swing_highs if s["index"] >= recent_cutoff]
    recent_lows  = [s for s in swing_lows  if s["index"] >= recent_cutoff]

    # Uptrend support — connect Higher Lows
    hl_points = _filter_higher_lows(recent_lows)
    if len(hl_points) >= 2:
        line = _fit_line(hl_points, "support", h, l, c)
        if line:
            lines.append(line)

    # Downtrend resistance — connect Lower Highs
    lh_points = _filter_lower_highs(recent_highs)
    if len(lh_points) >= 2:
        line = _fit_line(lh_points, "resistance", h, l, c)
        if line:
            lines.append(line)

    # Fallback: if no structural lines, try raw recent swing sequences
    if not lines:
        if len(recent_lows) >= 2:
            line = _fit_line(recent_lows[-4:], "support", h, l, c)
            if line:
                lines.append(line)
        if len(recent_highs) >= 2:
            line = _fit_line(recent_highs[-4:], "resistance", h, l, c)
            if line:
                lines.append(line)

    # If a line is invalid due to steep slope, replace with a near-horizontal
    # line through the most recent anchor (acts as dynamic S/R level)
    final = []
    for line in lines:
        if line["valid"]:
            final.append(line)
        else:
            # Try re-fitting with only the 2 most recent anchors
            pts = line.get("points", [])
            if len(pts) >= 2:
                # Use last 2 points only for a shallower fit
                recent_pts = pts[-2:]
                refit = _fit_line(recent_pts, line["type"], h, l, c)
                if refit and refit["valid"]:
                    final.append(refit)
                else:
                    final.append(line)  # keep original even if invalid
            else:
                final.append(line)
    return final if final else lines


def _filter_higher_lows(swing_lows: List[Dict], min_bar_gap: int = 5) -> List[Dict]:
    result = []
    for pt in swing_lows:
        if not result or (pt["price"] > result[-1]["price"] and pt["index"] - result[-1]["index"] >= min_bar_gap):
            result.append(pt)
    return result


def _filter_lower_highs(swing_highs: List[Dict], min_bar_gap: int = 5) -> List[Dict]:
    result = []
    for pt in swing_highs:
        if not result or (pt["price"] < result[-1]["price"] and pt["index"] - result[-1]["index"] >= min_bar_gap):
            result.append(pt)
    return result


def _fit_line(
    points: List[Dict],
    line_type: str,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
) -> Optional[Dict]:
    if len(points) < 2:
        return None

    pts = points[-6:]  # cap at 6 anchors
    xs = np.array([p["index"] for p in pts], dtype=float)
    ys = np.array([p["price"] for p in pts], dtype=float)

    if len(pts) == 2:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0]) if xs[1] != xs[0] else 0.0
        intercept = ys[0] - slope * xs[0]
    else:
        coeffs = np.polyfit(xs, ys, 1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])

    slope     = round(slope, 6)
    intercept = round(intercept, 4)

    touches, valid = _validate_line(slope, intercept, pts, h, l, c, line_type)
    strength_score = round(min(1.0, touches * 0.4 + (0.6 if valid else 0.0)), 2)

    breakout = _detect_breakout(slope, intercept, line_type, h, l, c)

    return {
        "type":          line_type,
        "points":        [{"index": int(p["index"]), "price": p["price"]} for p in pts[:2]],
        "slope":         slope,
        "intercept":     intercept,
        "touches":       touches,
        "valid":         valid,
        "strength_score": strength_score,
        "breakout":      breakout,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_line(
    slope: float,
    intercept: float,
    anchor_points: List[Dict],
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    line_type: str,
) -> Tuple[int, bool]:
    n = len(c)
    touches = 0
    violations = 0
    tolerance_pct = 0.003  # 0.3% body-close deviation

    # Only validate bars within the anchor span + recent bars
    # Avoids false violations from bars far before the line was established
    first_anchor = min(p["index"] for p in anchor_points) if anchor_points else 0
    check_start  = max(0, first_anchor - 5)

    for i in range(check_start, n):
        line_price = slope * i + intercept
        tolerance  = line_price * tolerance_pct

        if line_type == "support":
            if abs(l[i] - line_price) <= tolerance:
                touches += 1
            if c[i] < line_price - tolerance:
                violations += 1
        else:
            if abs(h[i] - line_price) <= tolerance:
                touches += 1
            if c[i] > line_price + tolerance:
                violations += 1

    check_bars = n - check_start
    max_violations = max(2, int(check_bars * 0.20))
    violated = violations > max_violations

    # Only reject extreme slope directions
    if line_type == "support" and slope < -0.5:
        violated = True
    if line_type == "resistance" and slope > 0.5:
        violated = True

    return max(touches, len(anchor_points)), not violated


# ---------------------------------------------------------------------------
# Breakout detection
# ---------------------------------------------------------------------------

def _detect_breakout(
    slope: float,
    intercept: float,
    line_type: str,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
) -> Optional[Dict]:
    n = len(c)
    if n < 3:
        return None

    for i in range(n - 2, 0, -1):
        line_price = slope * i + intercept
        confirm    = slope * (i + 1) + intercept

        if line_type == "support":
            if c[i] < line_price and c[i + 1] < confirm:
                return {"type": "bearish", "index": i, "price": round(float(c[i]), 5)}
        else:
            if c[i] > line_price and c[i + 1] > confirm:
                return {"type": "bullish", "index": i, "price": round(float(c[i]), 5)}

    return None


# ---------------------------------------------------------------------------
# Channel detection
# ---------------------------------------------------------------------------

def _detect_channels(trendlines: List[Dict]) -> List[Dict]:
    channels = []
    slope_tol = 0.05  # relaxed: parallel lines within 0.05 slope units
    for i in range(len(trendlines)):
        for j in range(i + 1, len(trendlines)):
            a, b = trendlines[i], trendlines[j]
            if abs(a["slope"] - b["slope"]) < slope_tol and a["type"] != b["type"]:
                upper_idx = i if a["type"] == "resistance" else j
                lower_idx = j if a["type"] == "resistance" else i
                channels.append({
                    "upper_line_index": upper_idx,
                    "lower_line_index": lower_idx,
                    "slope": round((a["slope"] + b["slope"]) / 2, 6),
                })
    return channels
