"""
HTF Long Strategy — higher-timeframe bias-driven position trade (BUY only).

Macro trends (DXY, US10Y) come from the engine's correlation data.
News blackout comes from the engine's calendar data.
All per-TF structure, body ratio, weekly range, and OHLC come from market_state.

Timeframes used:
  H1, H4, D1 — structure detection
  W1 (D1 last 5 candles as proxy when W1 unavailable) — weekly range

Entry: pullback completion or H1 structure confirmation.
Exit:  swing_trend profile (trailing SL, no timeout).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import config as cfg
from ..strategies.base_strategy import BaseStrategy
from ..htf_bias_engine import evaluate as htf_evaluate


def _no(reason: str) -> Dict[str, Any]:
    return {"signal": "NO_TRADE", "reason": reason, "confidence": 0.0}


def _with_macro_context(reason: str, dxy_trend: str, us10y_trend: str, macro_source: str) -> str:
    return f"{reason} | DXY={dxy_trend} US10Y={us10y_trend} [src={macro_source}]"


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

        account = data.get("account") or {}
        ms = data.get("market_state") or {}

        # ── All derived data from market_state ────────────────────────────────
        structure = ms.get("structure") or {}
        body_ratio = ms.get("body_ratio") or {}
        weekly = ms.get("weekly_range") or {"high_7d": 0.0, "low_7d": 0.0, "mid_7d": 0.0}
        pullback_pct = float(ms.get("pullback_depth") or 0.0)
        ohlc = ms.get("ohlc") or {}

        # ── Macro from engine correlation ───────────────────────────────────
        correlation = data.get("correlation") or {}
        dxy_trend   = str((correlation.get("dxy") or {}).get("trend") or "NEUTRAL").upper()
        us10y_trend = str((correlation.get("us10y") or {}).get("trend") or "NEUTRAL").upper()
        macro_source = "engine_correlation"

        # ── News from engine calendar ───────────────────────────────────────
        cal = data.get("calendar") or {}
        high_impact_soon = bool(cal.get("blocked"))
        manual_long_bias = bool(getattr(cfg, "HTF_LONG_MANUAL_BIAS_ENABLED", False))

        htf_data: Dict[str, Any] = {
            "price": price,
            "ohlc": ohlc,
            "weekly_range": weekly,
            "structure": structure,
            "momentum": {
                "D1_body_ratio": float(body_ratio.get("D1", 0.0)),
                "H4_body_ratio": float(body_ratio.get("H4", 0.0)),
            },
            "macro": {"dxy_trend": dxy_trend, "us10y_trend": us10y_trend},
            "news": {"recent_events_bias": "NEUTRAL", "high_impact_soon": high_impact_soon},
            "pullback": {"depth_pct": pullback_pct},
            "account": {"balance": float(account.get("balance", 0.0))},
            "manual_long_bias": manual_long_bias,
        }

        result = htf_evaluate(htf_data)

        if result["decision"] != "BUY":
            return _no(_with_macro_context(result["reason"], dxy_trend, us10y_trend, macro_source))

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
            "reason": _with_macro_context(result["reason"], dxy_trend, us10y_trend, macro_source),
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
            "_htf_macro_source":           macro_source,
            "_htf_macro_fetched_at":       "",
            "_htf_manual_long_bias":       manual_long_bias,
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
