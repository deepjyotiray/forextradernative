"""
MarketState — single source of truth for all derived market data.

The engine calls compute_market_state() once per cycle and passes the result
in strat_data["market_state"]. Strategies read from it; they never compute
indicators, structure, trend labels, or levels themselves.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .indicators import ema, atr, rsi, supertrend, wilder_atr
from .market_memory import compute_market_memory
from .trendline import detect_trendlines


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe(df) -> bool:
    return df is not None and len(df) >= 3


def _trend_label(df: Optional[pd.DataFrame], fast: int = 20, slow: int = 50) -> str:
    if df is None or len(df) < max(fast, slow) + 3:
        return "RANGE"
    c = df["close"].astype(float).values
    ef = ema(c, fast)
    es = ema(c, slow)
    slope = float(ef[-1] - ef[-3]) if len(ef) >= 3 else 0.0
    if ef[-1] > es[-1] and slope > 0:
        return "UP"
    if ef[-1] < es[-1] and slope < 0:
        return "DOWN"
    return "RANGE"


def _ema_snapshot(df: Optional[pd.DataFrame], period: int) -> Dict[str, float]:
    if df is None or len(df) < period + 3:
        return {"value": 0.0, "slope": 0.0}
    c = df["close"].astype(float).values
    e = ema(c, period)
    return {
        "value": round(float(e[-1]), 4),
        "slope": round(float(e[-1] - e[-3]), 4) if len(e) >= 3 else 0.0,
    }


def _atr_snapshot(df: Optional[pd.DataFrame], period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return 0.0
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    c = df["close"].astype(float).values
    return round(float(atr(h, l, c, period)[-1]), 4)


def _body_ratio(df: Optional[pd.DataFrame], n: int = 3) -> float:
    if df is None or len(df) < n:
        return 0.0
    sub = df.tail(n)
    bodies = (sub["close"] - sub["open"]).abs()
    ranges = (sub["high"] - sub["low"]).replace(0, np.nan)
    ratio = (bodies / ranges).dropna()
    return round(float(ratio.mean()), 4) if len(ratio) else 0.0


def _weekly_range(d1_df: Optional[pd.DataFrame], w1_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    src = w1_df if (w1_df is not None and len(w1_df) >= 1) else None
    if src is None and d1_df is not None and len(d1_df) >= 5:
        src = d1_df.tail(5)
    if src is None:
        return {"high_7d": 0.0, "low_7d": 0.0, "mid_7d": 0.0}
    high_7d = float(src["high"].max())
    low_7d = float(src["low"].min())
    return {"high_7d": high_7d, "low_7d": low_7d, "mid_7d": round((high_7d + low_7d) / 2, 2)}


def _pullback_depth(price: float, weekly: Dict[str, float]) -> float:
    high = weekly.get("high_7d", 0.0)
    low = weekly.get("low_7d", 0.0)
    rng = high - low
    if rng <= 0 or high <= 0:
        return 0.0
    return round(max(0.0, (high - price) / rng), 4)


def _asia_levels(m15_df: Optional[pd.DataFrame], now_utc) -> Dict[str, float]:
    if m15_df is None or m15_df.empty:
        return {"asia_high": 0.0, "asia_low": 0.0}
    try:
        from datetime import timedelta, timezone
        _UTC = timezone.utc
        session_start = now_utc.astimezone(_UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        london_start = session_start + timedelta(hours=7)
        times = m15_df["datetime"] if "datetime" in m15_df.columns else m15_df.index.to_series()
        window = m15_df[(times >= session_start) & (times < london_start)]
        if window.empty:
            return {"asia_high": 0.0, "asia_low": 0.0}
        return {
            "asia_high": round(float(window["high"].max()), 2),
            "asia_low": round(float(window["low"].min()), 2),
        }
    except Exception:
        return {"asia_high": 0.0, "asia_low": 0.0}


def _prev_day_levels(d1_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    if d1_df is None or len(d1_df) < 2:
        return {"pdh": 0.0, "pdl": 0.0}
    prev = d1_df.iloc[-2]
    return {"pdh": round(float(prev["high"]), 2), "pdl": round(float(prev["low"]), 2)}


def _equal_levels(df: Optional[pd.DataFrame], mode: str, tolerance: float = 0.5):
    if df is None or len(df) < 20:
        return []
    col = "high" if mode == "HIGH" else "low"
    values = df[col].tail(40).astype(float).tolist()
    matches = []
    for i, v in enumerate(values):
        for other in values[i + 1:]:
            if abs(v - other) <= tolerance:
                matches.append(round((v + other) / 2.0, 2))
                break
    return sorted(set(matches))


def _supertrend_snapshot(df: Optional[pd.DataFrame], period: int = 10, mult: float = 3.0) -> Dict:
    if df is None or len(df) < period + 5:
        return {"direction": 0, "value": 0.0, "stable": False}
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    c = df["close"].astype(float).values
    st_dir, st_line = supertrend(h, l, c, period, mult)
    direction = int(st_dir[-1])
    stable = len(st_dir) >= 3 and all(int(d) == direction for d in st_dir[-3:])
    return {
        "direction": direction,
        "value": round(float(st_line[-1]), 2),
        "stable": stable,
    }


def _htf_structure(df: Optional[pd.DataFrame]) -> str:
    """Classify UP/DOWN/RANGE using swing structure (HH/HL or LH/LL).
    Uses 2-bar pivot detection so it works on H4 where trends have wide pivots."""
    if df is None or len(df) < 10:
        return "RANGE"

    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    c = df["close"].astype(float).values
    n = len(h)
    strength = 2  # bars each side required to confirm a pivot

    highs, lows = [], []
    for i in range(strength, n - strength):
        if h[i] == max(h[i - strength: i + strength + 1]):
            highs.append(h[i])
        if l[i] == min(l[i - strength: i + strength + 1]):
            lows.append(l[i])

    hh = len(highs) >= 2 and highs[-1] > highs[-2]
    hl = len(lows)  >= 2 and lows[-1]  > lows[-2]
    lh = len(highs) >= 2 and highs[-1] < highs[-2]
    ll = len(lows)  >= 2 and lows[-1]  < lows[-2]

    if hh and hl:
        return "UP"
    if lh and ll:
        return "DOWN"
    # One condition is enough when the other is neutral (e.g. HH alone in early trend)
    if hh and not ll:
        return "UP"
    if ll and not hh:
        return "DOWN"

    # Fallback: EMA20 slope over 10 candles, no price-position requirement
    e = ema(c, 20)
    lookback = min(10, len(e) - 1)
    slope_pct = (e[-1] - e[-1 - lookback]) / e[-1] if e[-1] else 0.0
    if slope_pct > 0.0005:
        return "UP"
    if slope_pct < -0.0005:
        return "DOWN"
    return "RANGE"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_market_state(candles: Dict[str, pd.DataFrame], now_utc=None) -> Dict[str, Any]:
    """
    Compute all derived market data from raw candles.
    Called once per engine cycle; result passed as strat_data["market_state"].

    Keys returned:
      trend          — per-TF trend label (UP/DOWN/RANGE)
      structure      — per-TF HTF structure label (UP/DOWN/RANGE)
      ema            — per-TF EMA snapshots {value, slope}
      atr            — per-TF ATR14 value
      body_ratio     — per-TF avg body ratio (last 3 candles)
      supertrend     — M15 supertrend {direction, value, stable}
      trendlines     — M15 trendlines result
      weekly_range   — {high_7d, low_7d, mid_7d}
      pullback_depth — float (fraction of 7d range price has pulled back)
      asia_levels    — {asia_high, asia_low}
      prev_day       — {pdh, pdl}
      equal_highs    — list of equal high levels from M15
      equal_lows     — list of equal low levels from M15
      rsi            — per-TF RSI14 last value
      market_memory  — rolling 4h multi-TF directional memory
    """
    m1  = candles.get("M1")
    m5  = candles.get("M5")
    m15 = candles.get("M15")
    m30 = candles.get("M30")
    h1  = candles.get("H1")
    h4  = candles.get("H4")
    d1  = candles.get("D1")
    w1  = candles.get("W1")

    tick_price = 0.0
    if m1 is not None and len(m1):
        tick_price = float(m1["close"].iloc[-1])

    weekly = _weekly_range(d1, w1)

    state: Dict[str, Any] = {
        "trend": {
            "M1":  _trend_label(m1,  fast=9,  slow=20),
            "M5":  _trend_label(m5,  fast=20, slow=50),
            "M15": _trend_label(m15, fast=20, slow=50),
            "H1":  _trend_label(h1,  fast=20, slow=50),
            "H4":  _trend_label(h4,  fast=20, slow=50),
            "D1":  _trend_label(d1,  fast=20, slow=50),
        },
        "structure": {
            "H1":  _htf_structure(h1),
            "H4":  _htf_structure(h4),
            "D1":  _htf_structure(d1),
        },
        "ema": {
            "M1_20":  _ema_snapshot(m1,  20),
            "M5_20":  _ema_snapshot(m5,  20),
            "M15_20": _ema_snapshot(m15, 20),
            "M15_50": _ema_snapshot(m15, 50),
            "H1_20":  _ema_snapshot(h1,  20),
            "H1_50":  _ema_snapshot(h1,  50),
            "H4_50":  _ema_snapshot(h4,  50),
            "D1_20":  _ema_snapshot(d1,  20),
        },
        "atr": {
            "M1":  _atr_snapshot(m1),
            "M5":  _atr_snapshot(m5),
            "M15": _atr_snapshot(m15),
            "H1":  _atr_snapshot(h1),
            "H4":  _atr_snapshot(h4),
            "D1":  _atr_snapshot(d1),
        },
        "body_ratio": {
            "M1":  _body_ratio(m1),
            "M5":  _body_ratio(m5),
            "M15": _body_ratio(m15),
            "H4":  _body_ratio(h4),
            "D1":  _body_ratio(d1),
        },
        "supertrend": {
            "M15": _supertrend_snapshot(m15),
        },
        "trendlines": {
            "M15": detect_trendlines(m15, timeframe="M15") if _safe(m15) and len(m15) >= 50 else {},
        },
        "weekly_range": weekly,
        "pullback_depth": _pullback_depth(tick_price, weekly),
        "asia_levels": _asia_levels(m15, now_utc) if now_utc else {"asia_high": 0.0, "asia_low": 0.0},
        "prev_day": _prev_day_levels(d1),
        "equal_highs": _equal_levels(m15, "HIGH"),
        "equal_lows":  _equal_levels(m15, "LOW"),
        "rsi": {
            "M1":  _rsi_last(m1),
            "M5":  _rsi_last(m5),
            "M15": _rsi_last(m15),
        },
        "ohlc": {
            "H1": _last_ohlc(h1),
            "H4": _last_ohlc(h4),
            "D1": _last_ohlc(d1),
            "W1": _last_ohlc(w1),
        },
        "market_memory": compute_market_memory({"M1": m1, "M5": m5, "M15": m15}),
    }
    return state


def _rsi_last(df: Optional[pd.DataFrame], period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return 50.0
    c = df["close"].astype(float).values
    return round(float(rsi(c, period)[-1]), 2)


def _last_ohlc(df: Optional[pd.DataFrame]) -> Dict[str, float]:
    if df is None or len(df) == 0:
        return {}
    row = df.iloc[-1]
    return {col: float(row[col]) for col in ("open", "high", "low", "close") if col in row.index}
