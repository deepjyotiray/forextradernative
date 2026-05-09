"""
Swing Engine Strategy — 1-3 day position trades on XAUUSD.

All trend labels, weekly levels, pullback depth, ATR, and body ratio
are consumed from data["market_state"] — computed once by the engine.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd

from engine.strategies.base_strategy import BaseStrategy
from engine import strategy_configs as _scfg
from engine.sl_streak_guard import sl_streak_guard, STRICT_MIN_CONFIDENCE, STRICT_PULLBACK_MIN, STRICT_PULLBACK_MAX


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _nearest_zone(levels: List[Dict], price: float) -> float:
    if not levels:
        return 0.0
    mids = [_safe_float(item.get("zone_mid")) for item in levels]
    return min(mids, key=lambda lv: abs(lv - price))


def _has_rejection(candle: pd.Series, direction: str) -> bool:
    high  = _safe_float(candle.get("high"))
    low   = _safe_float(candle.get("low"))
    open_ = _safe_float(candle.get("open"))
    close = _safe_float(candle.get("close"))
    rng   = max(0.01, high - low)
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    return (lower_wick / rng) >= 0.25 if direction == "BUY" else (upper_wick / rng) >= 0.25


def _candle_body_ratio(candle: pd.Series) -> float:
    rng  = max(0.01, _safe_float(candle.get("high")) - _safe_float(candle.get("low")))
    body = abs(_safe_float(candle.get("close")) - _safe_float(candle.get("open")))
    return round(body / rng, 3)


def _candle_is_directional(candle: pd.Series, direction: str) -> bool:
    return _safe_float(candle.get("close")) > _safe_float(candle.get("open")) \
        if direction == "BUY" \
        else _safe_float(candle.get("close")) < _safe_float(candle.get("open"))


def _macro_bias(correlation: Dict) -> str:
    dxy   = str((correlation.get("dxy")   or {}).get("trend") or "").upper()
    us10y = str((correlation.get("us10y") or {}).get("trend") or "").upper()
    if dxy == "UP"   and us10y == "UP":   return "STRONG_SELL_GOLD"
    if dxy == "DOWN" and us10y == "DOWN": return "STRONG_BUY_GOLD"
    return "NEUTRAL"


def _news_payload(calendar_state: Dict) -> Dict:
    event = calendar_state.get("next_event") or {}
    return {
        "high_impact": bool(calendar_state.get("blocked") or event.get("critical")),
        "actual":      _safe_float(event.get("actual")),
        "forecast":    _safe_float(event.get("forecast")),
    }


def _news_alignment(direction: str, news: Dict) -> bool:
    actual   = _safe_float(news.get("actual"))
    forecast = _safe_float(news.get("forecast"))
    if actual == 0.0 and forecast == 0.0:
        return False
    return actual < forecast if direction == "BUY" else actual > forecast


class SwingEngineStrategy(BaseStrategy):
    name = "SWING_ENGINE"

    def generate_signal(self, data: Dict) -> Dict:
        tick  = data.get("tick") or {}
        price = _safe_float(tick.get("bid") or tick.get("ask"))
        if price <= 0:
            return self._no("No tick data")

        h4 = data.get("h4_df")
        d1 = data.get("d1_df")
        if h4 is None or d1 is None or h4.empty or d1.empty:
            return self._no("Missing H4/D1 data")

        cfg    = _scfg.get(self.name)
        spread = _safe_float(tick.get("spread"))
        if spread > float(cfg.get("spread_max", 0.7)):
            return self._no(f"Spread {spread:.2f} > {cfg.get('spread_max', 0.7)}")

        open_count = int((data.get("strategy_trade_counts") or {}).get(self.name, 0) or 0)
        if open_count >= int(cfg.get("max_active_trades", 1)):
            return self._no(f"Max {cfg.get('max_active_trades', 1)} active swing trade(s)")

        cooldown_left = sl_streak_guard.cooldown_remaining(self.name)
        if cooldown_left > 0:
            return self._no(
                f"SL streak cooldown: {cooldown_left:.0f}s remaining after "
                f"{sl_streak_guard.status(self.name)['consecutive_sl_hits']} consecutive SL hits"
            )
        strict_mode = sl_streak_guard.is_strict_active(self.name)

        # ── 1. Trend from market_state ────────────────────────────────────
        # Use structure (10-candle EMA20 slope) as primary — more stable on HTF.
        # Fall back to trend label only when structure is also RANGE.
        ms        = data.get("market_state") or {}
        trend     = ms.get("trend") or {}
        structure = ms.get("structure") or {}

        def _resolve(tf: str) -> str:
            s = str(structure.get(tf, "RANGE")).upper()
            return s if s != "RANGE" else str(trend.get(tf, "RANGE")).upper()

        d1_trend = _resolve("D1")
        h4_trend = _resolve("H4")

        if d1_trend == "UP" and h4_trend in ("UP", "RANGE"):
            direction = "BUY"
        elif d1_trend == "DOWN" and h4_trend in ("DOWN", "RANGE"):
            direction = "SELL"
        else:
            return self._no(f"D1/H4 not aligned (D1:{d1_trend} H4:{h4_trend})")

        if strict_mode:
            h1_trend = _resolve("H1")
            need = "UP" if direction == "BUY" else "DOWN"
            if h1_trend != need:
                return self._no(
                    f"Strict gate (post SL streak): H1 not aligned (H1:{h1_trend}, need {need})"
                )

        confidence = 50

        # ── 1.5. Macro early block ────────────────────────────────────────
        macro_early = _macro_bias(data.get("correlation") or {})
        if macro_early == "STRONG_BUY_GOLD"  and direction == "SELL":
            return self._no("Macro strongly bullish (DXY+US10Y DOWN) — no SELL swing")
        if macro_early == "STRONG_SELL_GOLD" and direction == "BUY":
            return self._no("Macro strongly bearish (DXY+US10Y UP) — no BUY swing")
        if strict_mode and macro_early == "NEUTRAL":
            return self._no("Strict gate (post SL streak): macro must actively support direction (currently NEUTRAL)")

        # ── 2. Location: price near key level ────────────────────────────
        zones    = data.get("zones") or {}
        prev_day = ms.get("prev_day") or {}
        weekly   = ms.get("weekly_range") or {}
        atr_h4   = float((ms.get("atr") or {}).get("H4") or 0.0)
        if atr_h4 <= 0:
            atr_h4 = max(1.0, _safe_float((h4["high"] - h4["low"]).tail(14).mean(), 12.0))
        tolerance = max(3.0, min(8.0, atr_h4 * 0.35))

        if direction == "BUY":
            key_levels = [
                _nearest_zone(zones.get("support") or [], price),
                _safe_float(prev_day.get("pdl")),
                _safe_float(weekly.get("low_7d")),
            ]
        else:
            key_levels = [
                _nearest_zone(zones.get("resistance") or [], price),
                _safe_float(prev_day.get("pdh")),
                _safe_float(weekly.get("high_7d")),
            ]
        near_level = any(lv > 0 and abs(price - lv) <= tolerance for lv in key_levels)

        pullback = float(ms.get("pullback_depth") or 0.0)
        pullback_min = STRICT_PULLBACK_MIN if strict_mode else 0.30
        pullback_max = STRICT_PULLBACK_MAX if strict_mode else 0.70
        if not (near_level or (pullback_min <= pullback <= pullback_max)):
            gate_note = " (strict pullback range active)" if strict_mode else ""
            return self._no(f"Location filter failed: not near key level, not in pullback zone{gate_note}")

        # ── 3. Entry candle ───────────────────────────────────────────────
        h4_last  = h4.iloc[-1]
        candle_ok = _candle_body_ratio(h4_last) >= 0.5 and (
            _has_rejection(h4_last, direction) or _candle_is_directional(h4_last, direction)
        )
        if not candle_ok:
            return self._no("H4 entry candle not confirmed (body < 0.5 or no rejection/direction)")

        confidence += 30

        # ── 4. Macro + news boosters ──────────────────────────────────────
        macro = _macro_bias(data.get("correlation") or {})
        news  = _news_payload(data.get("calendar") or {})
        if (direction == "BUY"  and macro == "STRONG_BUY_GOLD") or \
           (direction == "SELL" and macro == "STRONG_SELL_GOLD"):
            confidence += 10
        if _news_alignment(direction, news):
            confidence += 10

        min_confidence = int(cfg.get("min_confidence_pct", 80) or 80)
        if confidence < min_confidence:
            return self._no(f"Confidence {confidence} < {min_confidence}")

        if strict_mode and confidence < STRICT_MIN_CONFIDENCE:
            return self._no(
                f"Strict gate (post SL streak): confidence {confidence} < {STRICT_MIN_CONFIDENCE} required"
            )

        # ── 5. SL / TP ────────────────────────────────────────────────────
        sl_min = float(cfg.get("sl_min", 10.0))
        sl_max = float(cfg.get("sl_max", 25.0))
        tp1_r  = float(cfg.get("tp1_r", 1.5))
        tp2_r  = float(cfg.get("tp2_r", 2.5))
        tp3_r  = float(cfg.get("tp3_r") or 3.0)

        swing_low  = _safe_float(h4["low"].tail(8).min())
        swing_high = _safe_float(h4["high"].tail(8).max())

        if direction == "BUY":
            anchor   = swing_low if 0 < swing_low < price else price - sl_min
            sl_dist  = min(max(sl_min, price - min(price - sl_min, anchor - 0.5)), sl_max)
            sl       = round(price - sl_dist, 2)
        else:
            anchor   = swing_high if swing_high > price else price + sl_min
            sl_dist  = min(max(sl_min, max(price + sl_min, anchor + 0.5) - price), sl_max)
            sl       = round(price + sl_dist, 2)

        balance     = _safe_float((data.get("account") or {}).get("balance"))
        if balance <= 0:
            return self._no("Account balance unavailable")
        risk_amount = balance * (float(cfg.get("risk_pct", 1.0)) / 100.0)
        lot         = max(0.01, min(0.03, risk_amount / max(1.0, sl_dist * 100.0)))

        sign      = 1 if direction == "BUY" else -1
        tp_levels = [
            round(price + sign * sl_dist * tp1_r, 2),
            round(price + sign * sl_dist * tp2_r, 2),
            round(price + sign * sl_dist * tp3_r, 2),
        ]

        h4_structure = str(structure.get("H4", "RANGE")).upper()
        reason = (
            f"{direction} swing | D1:{d1_trend} H4:{h4_trend}(struct:{h4_structure}) | "
            f"{'near level' if near_level else 'pullback'} | "
            f"body:{_candle_body_ratio(h4_last):.2f} | conf:{confidence}"
        )

        return {
            "signal": direction, "entry": round(price, 2),
            "sl": sl, "tp": tp_levels[-1], "tp_levels": tp_levels,
            "sl_distance": round(sl_dist, 2), "lot": round(lot, 2),
            "confidence": round(confidence / 100.0, 2), "confidence_pct": confidence,
            "reason": reason, "rr": round(tp3_r, 2),
            "_strategy_name": self.name, "strategy": self.name, "decision": direction,
            "_signal_family": "SWING", "_sweep_confirmed": True, "_candle_confirmation": True,
            "_exit_profile": "swing_engine", "_tp_levels": tp_levels,
            "_macro_bias": macro, "_news": news, "_strict_gate_active": strict_mode,
        }

    @staticmethod
    def _no(reason: str) -> Dict:
        return {"signal": "NO_TRADE", "reason": reason, "confidence": 0.0}
