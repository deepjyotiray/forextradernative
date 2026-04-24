"""
Technical indicators — fast numpy-based calculations.
"""
import numpy as np
import pandas as pd
from typing import Dict
import config as cfg


def ema(series: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1)
    out = np.empty_like(series, dtype=float)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    return out


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(highs)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
    return ema(tr, period)


def rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = ema(gains, period)
    avg_loss = ema(losses, period)
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    rsi_vals = 100.0 - (100.0 / (1.0 + rs))
    return np.concatenate([[50.0], rsi_vals])


def compute_indicators(df: pd.DataFrame) -> Dict:
    """Compute all indicators from a candle DataFrame. Returns dict of latest values."""
    if len(df) < 21:
        return {}
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    o = df["open"].values.astype(float)

    ema9 = ema(c, 9)
    ema15 = ema(c, 15)
    ema20 = ema(c, 20)
    ema50 = ema(c, 50) if len(c) >= 50 else ema9
    ema200 = ema(c, 200) if len(c) >= 200 else None
    atr7 = atr(h, l, c, 7)
    atr14 = atr(h, l, c, 14)
    atr20 = atr(h, l, c, 20)
    rsi14 = rsi(c, 14)

    last = len(c) - 1
    candle_range = h[last] - l[last]
    body = abs(c[last] - o[last])
    body_ratio = body / candle_range if candle_range > 0 else 0
    upper_wick = (h[last] - max(o[last], c[last])) / candle_range if candle_range > 0 else 0
    lower_wick = (min(o[last], c[last]) - l[last]) / candle_range if candle_range > 0 else 0

    # Calculate configured range lookback for compression gate.
    range_10 = 0
    range_lookback = max(2, int(getattr(cfg, "COMPRESSION_RANGE_LOOKBACK", 10)))
    if len(h) >= range_lookback:
        recent_high = h[-range_lookback:].max()
        recent_low = l[-range_lookback:].min()
        range_10 = recent_high - recent_low

    # Calculate ATR slope for compression gate
    atr_slope = 0
    if len(atr14) >= 4:
        atr_slope = atr14[-1] - atr14[-4]

    return {
        "ema9": round(float(ema9[last]), 2),
        "ema15": round(float(ema15[last]), 2),
        "ema20": round(float(ema20[last]), 2),
        "ema50": round(float(ema50[last]), 2) if len(c) >= 50 else None,
        "ema200": round(float(ema200[last]), 2) if ema200 is not None else None,
        "ema9_slope": round(float(ema9[last] - ema9[last - 1]), 4) if last > 0 else 0,
        "ema20_slope": round(float(ema20[last] - ema20[last - 1]), 4) if last > 0 else 0,
        "ema_gap": round(float(abs(ema9[last] - ema15[last])), 4),
        "atr": round(float(atr7[last]), 4),
        "atr14": round(float(atr14[last]), 4),
        "atr20": round(float(atr20[last]), 4),
        "atr_ratio": round(float(atr7[last] / atr20[last]), 4) if atr20[last] > 0 else 1.0,
        "atr_slope": round(float(atr_slope), 4),
        "rsi": round(float(rsi14[last]), 2),
        "body_ratio": round(float(body_ratio), 4),
        "upper_wick": round(float(upper_wick), 4),
        "lower_wick": round(float(lower_wick), 4),
        "range_10": round(float(range_10), 4),
        "close": float(c[last]),
        "open": float(o[last]),
        "high": float(h[last]),
        "low": float(l[last]),
        "is_bullish": bool(c[last] > o[last]),
        "candle_range": round(float(candle_range), 4),
    }


def compute_timeframe_context(m1_df: pd.DataFrame = None,
                              m5_df: pd.DataFrame = None,
                              h1_df: pd.DataFrame = None) -> Dict:
    """Add the required live context metrics without changing strategy architecture."""
    ctx: Dict = {}

    if m1_df is not None and len(m1_df) >= 21:
        c1 = m1_df["close"].values.astype(float)
        ema20_m1 = ema(c1, 20)
        ctx["m1_ema20"] = round(float(ema20_m1[-1]), 2)
        ctx["m1_ema20_slope"] = round(float(ema20_m1[-1] - ema20_m1[-3]), 4) if len(ema20_m1) >= 3 else 0.0

    if m5_df is not None and len(m5_df) >= 21:
        c5 = m5_df["close"].values.astype(float)
        ema20_m5 = ema(c5, 20)
        ctx["m5_ema20"] = round(float(ema20_m5[-1]), 2)
        ctx["m5_ema20_slope"] = round(float(ema20_m5[-1] - ema20_m5[-3]), 4) if len(ema20_m5) >= 3 else 0.0

    if h1_df is not None and len(h1_df) >= 50:
        c_h1 = h1_df["close"].values.astype(float)
        ema50_h1 = ema(c_h1, 50)
        ctx["h1_ema50"] = round(float(ema50_h1[-1]), 2)
        ctx["h1_ema50_slope"] = round(float(ema50_h1[-1] - ema50_h1[-3]), 4) if len(ema50_h1) >= 3 else 0.0

    return ctx
