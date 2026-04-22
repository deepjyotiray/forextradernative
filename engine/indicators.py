"""
Technical indicators — fast numpy-based calculations.
"""
import numpy as np
import pandas as pd
from typing import Dict


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

    return {
        "ema9": round(ema9[last], 2),
        "ema15": round(ema15[last], 2),
        "ema50": round(ema50[last], 2) if len(c) >= 50 else None,
        "ema200": round(ema200[last], 2) if ema200 is not None else None,
        "ema9_slope": round(ema9[last] - ema9[last - 1], 4) if last > 0 else 0,
        "ema_gap": round(abs(ema9[last] - ema15[last]), 4),
        "atr": round(atr7[last], 4),
        "atr14": round(atr14[last], 4),
        "atr20": round(atr20[last], 4),
        "atr_ratio": round(atr7[last] / atr20[last], 4) if atr20[last] > 0 else 1.0,
        "rsi": round(rsi14[last], 2),
        "body_ratio": round(body_ratio, 4),
        "upper_wick": round(upper_wick, 4),
        "lower_wick": round(lower_wick, 4),
        "close": c[last],
        "open": o[last],
        "high": h[last],
        "low": l[last],
        "is_bullish": c[last] > o[last],
        "candle_range": round(candle_range, 4),
    }
