"""
HTF Long Strategy — higher-timeframe bias-driven position trade (BUY only).

Macro trends (DXY, US10Y) are fetched automatically from Yahoo Finance
via the MacroFetcher background service — no manual config needed.

News blackout is read from the existing EconomicCalendar singleton.

Timeframes used:
  H1, H4, D1 — structure detection
  W1 (D1 last 5 candles as proxy when W1 unavailable) — weekly range

Entry: pullback completion or H1 structure confirmation.
Exit:  swing_trend profile (trailing SL, no timeout).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

import config as cfg
from ..strategies.base_strategy import BaseStrategy
from ..htf_bias_engine import evaluate as htf_evaluate
from ..macro_fetcher import macro_fetcher
from ..calendar import calendar


def _no(reason: str) -> Dict[str, Any]:
    return {"signal": "NO_TRADE", "reason": reason, "confidence": 0.0}


def _df_structure(df: Optional[pd.DataFrame]) -> str:
    """Classify a DataFrame as UP / DOWN / RANGE.

    Uses two independent signals — both must agree for a directional call:
      1. EMA20 slope over last 10 bars (normalised by price)
      2. Price position relative to EMA20

    This prevents false RANGE during consolidations within a trend.
    """
    if df is None or len(df) < 20:
        return "RANGE"
    c = df["close"].values.astype(float)
    k = 2.0 / 21
    ema_vals = []
    ema = c[0]
    for v in c[1:]:
        ema = v * k + ema * (1 - k)
        ema_vals.append(ema)
    # Slope: compare EMA now vs 10 bars ago, normalised by price
    lookback = min(10, len(ema_vals) - 1)
    slope_pct = (ema_vals[-1] - ema_vals[-1 - lookback]) / ema_vals[-1] if ema_vals[-1] else 0.0
    price_above = c[-1] > ema_vals[-1]
    price_below = c[-1] < ema_vals[-1]
    slope_up   = slope_pct > 0.0005   # 0.05% over 10 bars
    slope_down = slope_pct < -0.0005
    if slope_up and price_above:
        return "UP"
    if slope_down and price_below:
        return "DOWN"
    return "RANGE"


def _body_ratio(df: Optional[pd.DataFrame], n: int = 3) -> float:
    """Average body-to-range ratio over last n candles."""
    if df is None or len(df) < n:
        return 0.0
    sub = df.tail(n)
    bodies = (sub["close"] - sub["open"]).abs()
    ranges = (sub["high"] - sub["low"]).replace(0, np.nan)
    ratio = (bodies / ranges).dropna()
    return float(ratio.mean()) if len(ratio) else 0.0


def _weekly_range(d1_df: Optional[pd.DataFrame], w1_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    """Derive 7-day high/low/mid from W1 if available, else last 5 D1 candles."""
    src = w1_df if (w1_df is not None and len(w1_df) >= 1) else None
    if src is None and d1_df is not None and len(d1_df) >= 5:
        src = d1_df.tail(5)
    if src is None:
        return {"high_7d": 0.0, "low_7d": 0.0, "mid_7d": 0.0}
    high_7d = float(src["high"].max())
    low_7d = float(src["low"].min())
    return {
        "high_7d": high_7d,
        "low_7d": low_7d,
        "mid_7d": round((high_7d + low_7d) / 2, 2),
    }


def _pullback_depth(price: float, weekly: Dict[str, float]) -> float:
    """Retracement from recent 7d high as a fraction of the 7d range."""
    high = weekly.get("high_7d", 0.0)
    low = weekly.get("low_7d", 0.0)
    rng = high - low
    if rng <= 0 or high <= 0:
        return 0.0
    return round(max(0.0, (high - price) / rng), 4)


class HTFLongStrategy(BaseStrategy):
    name = "HTF_LONG"

    def generate_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not getattr(cfg, "HTF_LONG_ENABLED", True):
            return _no("HTF_LONG strategy disabled")

        tick = data.get("tick") or {}
        price = float(tick.get("ask", tick.get("bid", 0.0)))
        if price <= 0:
            return _no("No valid price")

        spread = float(tick.get("spread", 0.0))
        max_spread = float(getattr(cfg, "HTF_LONG_MAX_SPREAD", 0.80))
        if spread > max_spread:
            return _no(f"Spread {spread:.2f} > {max_spread}")

        h1_df = data.get("h1_df")
        h4_df = data.get("h4_df")
        d1_df = data.get("d1_df")
        w1_df = data.get("w1_df")
        account = data.get("account") or {}

        # ── Auto macro trends (no manual config needed) ───────────────────────
        macro_state = macro_fetcher.get()
        dxy_trend   = macro_state["dxy_trend"]
        us10y_trend = macro_state["us10y_trend"]

        # ── Auto news blackout from existing calendar singleton ───────────────
        calendar.poll()   # non-blocking; triggers background refresh if stale
        cal = calendar.check()
        high_impact_soon = bool(cal.get("blocked"))

        weekly = _weekly_range(d1_df, w1_df)
        pullback_pct = _pullback_depth(price, weekly)

        htf_data: Dict[str, Any] = {
            "price": price,
            "ohlc": {
                "H1": {col: float(h1_df[col].iloc[-1]) for col in ("open", "high", "low", "close")} if h1_df is not None and len(h1_df) else {},
                "H4": {col: float(h4_df[col].iloc[-1]) for col in ("open", "high", "low", "close")} if h4_df is not None and len(h4_df) else {},
                "D1": {col: float(d1_df[col].iloc[-1]) for col in ("open", "high", "low", "close")} if d1_df is not None and len(d1_df) else {},
                "W1": {col: float(w1_df[col].iloc[-1]) for col in ("open", "high", "low", "close")} if w1_df is not None and len(w1_df) else {},
            },
            "weekly_range": weekly,
            "structure": {
                "H1": _df_structure(h1_df),
                "H4": _df_structure(h4_df),
                "D1": _df_structure(d1_df),
            },
            "momentum": {
                "D1_body_ratio": _body_ratio(d1_df, 3),
                "H4_body_ratio": _body_ratio(h4_df, 3),
            },
            "macro": {
                "dxy_trend":   dxy_trend,
                "us10y_trend": us10y_trend,
            },
            "news": {
                "recent_events_bias": "NEUTRAL",
                "high_impact_soon":   high_impact_soon,
            },
            "pullback": {"depth_pct": pullback_pct},
            "account": {"balance": float(account.get("balance", 0.0))},
        }

        result = htf_evaluate(htf_data)

        if result["decision"] != "BUY":
            return _no(result["reason"])

        entry = float(result["entry_zone"])
        sl = float(result["sl"])
        sl_dist = round(abs(entry - sl), 2)
        if sl_dist < 5.0:
            return _no(f"SL distance too small: {sl_dist}")

        tps = [t for t in result["tp"] if t is not None]
        tp_primary = tps[0] if tps else round(entry + sl_dist * 2, 2)
        rr = round(abs(tp_primary - entry) / sl_dist, 2)

        min_rr = float(getattr(cfg, "HTF_LONG_MIN_RR", 1.8))
        if rr < min_rr:
            return _no(f"RR {rr:.2f} < {min_rr}")

        confidence = round(result["confidence"] / 100.0, 4)

        return {
            "signal":       "BUY",
            "entry":        entry,
            "sl":           sl,
            "tp":           tp_primary,
            "sl_distance":  sl_dist,
            "confidence":   confidence,
            "rr":           rr,
            "reason": (
                f"{result['reason']} | "
                f"DXY={dxy_trend} US10Y={us10y_trend} "
                f"[src={macro_state.get('source','?')}]"
            ),
            # Internal metadata
            "_exit_profile":               "swing_trend",
            "_signal_family":              "SWING",
            "_sweep_confirmed":            True,
            "_candle_confirmation":        True,
            "_setup_direction":            "LONG",
            "_htf_strategy":               result["strategy"],
            "_htf_confidence":             result["confidence"],
            "_htf_tp_levels":              tps,
            "_htf_dxy_trend":              dxy_trend,
            "_htf_us10y_trend":            us10y_trend,
            "_htf_macro_source":           macro_state.get("source", "unknown"),
            "_htf_macro_fetched_at":       macro_state.get("fetched_at_iso", ""),
            "_be_trigger_r":               cfg.EXIT_PROFILE_TREND_BE_TRIGGER_R,
            "_breakeven_min_hold_seconds": cfg.EXIT_PROFILE_TREND_BREAKEVEN_MIN_HOLD_SECONDS,
            "_breakeven_volume_hold_ratio":cfg.EXIT_PROFILE_TREND_BREAKEVEN_VOLUME_HOLD_RATIO,
            "_min_hold_seconds":           cfg.EXIT_PROFILE_TREND_MIN_HOLD_SECONDS,
            "_early_fail":                 0.0,
            "_tier1_min_ticks":            0,
            "_tier1_max_ticks":            0,
            "_timeout":                    0,
            "_timeout_min_progress_r":     0.0,
            "_reversal_arm_r":             cfg.EXIT_PROFILE_TREND_REVERSAL_ARM_R,
            "_reversal_drawdown_pct":      cfg.EXIT_PROFILE_TREND_REVERSAL_DRAWDOWN_PCT,
            "_reversal_floor_r":           cfg.EXIT_PROFILE_TREND_REVERSAL_FLOOR_R,
            "_trail_activate_r":           cfg.EXIT_PROFILE_TREND_TRAIL_ACTIVATE_R,
            "_trail_lock_r":               cfg.EXIT_PROFILE_TREND_TRAIL_LOCK_R,
            "_velocity_drop_enabled":      False,
            "_entry_spread":               spread,
        }
