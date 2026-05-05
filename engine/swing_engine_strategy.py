"""
Swing Engine Strategy — 1-3 day position trades on XAUUSD.

Entry logic (per spec):
  - D1 and H4 must both trend the same direction (UP/DOWN)
  - Price must be near a key level: support/PDL/weekly_low (BUY) or resistance/PDH/weekly_high (SELL)
  - H4 entry candle must confirm: rejection wick OR strong directional body (body_ratio >= 0.5)
  - RANGE market = NO TRADE, mid-range entry = NO TRADE

Risk (per spec):
  - Risk 1% per trade
  - SL: 10-25 pts (structure-based)
  - TP: 1.5R / 2.5R / 3R

Confidence model (per spec):
  +50  structure aligned (D1+H4)
  +30  valid setup (location + candle)
  +10  macro aligned (DXY + US10Y)
  +10  news aligned
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


def _trend_label(df: Optional[pd.DataFrame], fast: int = 20, slow: int = 50) -> str:
    if df is None or len(df) < max(fast, slow) + 3:
        return "RANGE"
    close = df["close"].astype(float)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    slope = _safe_float(ema_fast.iloc[-1] - ema_fast.iloc[-3])
    if ema_fast.iloc[-1] > ema_slow.iloc[-1] and slope > 0:
        return "UP"
    if ema_fast.iloc[-1] < ema_slow.iloc[-1] and slope < 0:
        return "DOWN"
    return "RANGE"


def _body_ratio(candle: pd.Series) -> float:
    rng = max(0.01, _safe_float(candle.get("high")) - _safe_float(candle.get("low")))
    body = abs(_safe_float(candle.get("close")) - _safe_float(candle.get("open")))
    return round(body / rng, 3)


def _has_rejection(candle: pd.Series, direction: str) -> bool:
    high = _safe_float(candle.get("high"))
    low = _safe_float(candle.get("low"))
    open_ = _safe_float(candle.get("open"))
    close = _safe_float(candle.get("close"))
    rng = max(0.01, high - low)
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    if direction == "BUY":
        return (lower_wick / rng) >= 0.25
    return (upper_wick / rng) >= 0.25


def _candle_is_directional(candle: pd.Series, direction: str) -> bool:
    open_ = _safe_float(candle.get("open"))
    close = _safe_float(candle.get("close"))
    return close > open_ if direction == "BUY" else close < open_


def _weekly_levels(d1_df: Optional[pd.DataFrame]):
    if d1_df is None or len(d1_df) < 5:
        return 0.0, 0.0
    recent = d1_df.tail(5)
    return _safe_float(recent["high"].max()), _safe_float(recent["low"].min())


def _nearest_zone(levels: List[Dict], price: float) -> float:
    if not levels:
        return 0.0
    mids = [_safe_float(item.get("zone_mid")) for item in levels]
    return min(mids, key=lambda level: abs(level - price))


def _pullback_depth(price: float, df: Optional[pd.DataFrame], direction: str) -> float:
    if df is None or len(df) < 20:
        return 0.0
    recent = df.tail(20)
    high = _safe_float(recent["high"].max())
    low = _safe_float(recent["low"].min())
    rng = max(0.01, high - low)
    if direction == "BUY":
        return round((high - price) / rng, 3)
    return round((price - low) / rng, 3)


def _macro_bias(correlation: Dict) -> str:
    dxy = str((correlation.get("dxy") or {}).get("trend") or "").upper()
    us10y = str((correlation.get("us10y") or {}).get("trend") or "").upper()
    if dxy == "UP" and us10y == "UP":
        return "STRONG_SELL_GOLD"
    if dxy == "DOWN" and us10y == "DOWN":
        return "STRONG_BUY_GOLD"
    return "NEUTRAL"


def _news_payload(calendar_state: Dict) -> Dict:
    event = calendar_state.get("next_event") or {}
    return {
        "high_impact": bool(calendar_state.get("blocked") or event.get("critical")),
        "actual": _safe_float(event.get("actual")),
        "forecast": _safe_float(event.get("forecast")),
    }


def _news_alignment(direction: str, news: Dict) -> bool:
    actual = _safe_float(news.get("actual"))
    forecast = _safe_float(news.get("forecast"))
    if actual == 0.0 and forecast == 0.0:
        return False
    return actual < forecast if direction == "BUY" else actual > forecast


def _htf_bias_aligned(direction: str, market_context: Dict) -> bool:
    """Return True if D1 + H4 weekly structure agrees with the trade direction.
    Used to suppress premature reversal exits on swing trades."""
    d1_df = market_context.get("d1_df")
    h4_df = market_context.get("h4_df")
    if d1_df is None or h4_df is None:
        return False
    d1_trend = _trend_label(d1_df, fast=20, slow=50)
    h4_trend = _trend_label(h4_df, fast=20, slow=50)
    if direction == "SELL":
        return d1_trend == "DOWN" or h4_trend == "DOWN"
    if direction == "BUY":
        return d1_trend == "UP" or h4_trend == "UP"
    return False


class SwingEngineStrategy(BaseStrategy):
    name = "SWING_ENGINE"

    def generate_signal(self, data: Dict) -> Dict:
        tick = data.get("tick") or {}
        price = _safe_float(tick.get("bid") or tick.get("ask"))
        if price <= 0:
            return self._no("No tick data")

        h4 = data.get("h4_df")
        d1 = data.get("d1_df")
        if h4 is None or d1 is None or h4.empty or d1.empty:
            return self._no("Missing H4/D1 data")

        cfg = _scfg.get(self.name)
        spread = _safe_float(tick.get("spread"))
        spread_max = float(cfg.get("spread_max", 0.7))
        if spread > spread_max:
            return self._no(f"Spread {spread:.2f} > {spread_max}")

        max_active = int(cfg.get("max_active_trades", 1))
        open_count = int((data.get("strategy_trade_counts") or {}).get(self.name, 0) or 0)
        if open_count >= max_active:
            return self._no(f"Max {max_active} active swing trade(s)")

        # ── SL streak guard: cooldown + strict gate ───────────────────────
        cooldown_left = sl_streak_guard.cooldown_remaining(self.name)
        if cooldown_left > 0:
            return self._no(
                f"SL streak cooldown: {cooldown_left:.0f}s remaining after "
                f"{sl_streak_guard.status(self.name)['consecutive_sl_hits']} consecutive SL hits"
            )
        strict_mode = sl_streak_guard.is_strict_active(self.name)

        # ── 1. Bias: D1 + H4 must both trend same direction ──────────────
        d1_trend = _trend_label(d1)
        h4_trend = _trend_label(h4)
        if d1_trend == "UP" and h4_trend == "UP":
            direction = "BUY"
        elif d1_trend == "DOWN" and h4_trend == "DOWN":
            direction = "SELL"
        else:
            return self._no(f"D1/H4 not aligned (D1:{d1_trend} H4:{h4_trend})")

        # Strict mode: H1 must also agree
        if strict_mode:
            h1 = data.get("h1_df")
            h1_trend = _trend_label(h1, fast=20, slow=50) if h1 is not None else "RANGE"
            if h1_trend != direction.replace("BUY", "UP").replace("SELL", "DOWN"):
                return self._no(
                    f"Strict gate (post SL streak): H1 not aligned (H1:{h1_trend}, need "
                    f"{'UP' if direction == 'BUY' else 'DOWN'})"
                )

        confidence = 50  # +50 structure aligned

        # ── 1.5. Block entry if macro opposes D1/H4 swing direction ──────
        # Strict mode: macro must actively support (not just not oppose)
        # For swing trades we only care about macro (DXY/US10Y) and D1/H4.
        # M15/H1 short-term bias is intentionally ignored here.
        macro_early = _macro_bias(data.get("correlation") or {})
        if macro_early == "STRONG_BUY_GOLD" and direction == "SELL":
            return self._no(
                f"Macro strongly bullish (DXY+US10Y DOWN) — no SELL swing"
            )
        if macro_early == "STRONG_SELL_GOLD" and direction == "BUY":
            return self._no(
                f"Macro strongly bearish (DXY+US10Y UP) — no BUY swing"
            )
        if strict_mode and macro_early == "NEUTRAL":
            return self._no(
                "Strict gate (post SL streak): macro must actively support direction (currently NEUTRAL)"
            )

        # ── 2. Location: price near key level ────────────────────────────
        zones = data.get("zones") or {}
        weekly_high, weekly_low = _weekly_levels(d1)
        prev_day = d1.iloc[-2] if len(d1) >= 2 else d1.iloc[-1]
        atr_h4 = max(1.0, _safe_float((h4["high"] - h4["low"]).tail(14).mean(), 12.0))
        tolerance = max(3.0, min(8.0, atr_h4 * 0.35))

        if direction == "BUY":
            key_levels = [
                _nearest_zone(zones.get("support") or [], price),
                _safe_float(prev_day.get("low")),
                weekly_low,
            ]
            near_level = any(lv > 0 and abs(price - lv) <= tolerance for lv in key_levels)
        else:
            key_levels = [
                _nearest_zone(zones.get("resistance") or [], price),
                _safe_float(prev_day.get("high")),
                weekly_high,
            ]
            near_level = any(lv > 0 and abs(price - lv) <= tolerance for lv in key_levels)

        pullback = _pullback_depth(price, h4, direction)
        pullback_min = STRICT_PULLBACK_MIN if strict_mode else 0.30
        pullback_max = STRICT_PULLBACK_MAX if strict_mode else 0.70
        valid_location = near_level or (pullback_min <= pullback <= pullback_max)
        if not valid_location:
            gate_note = " (strict pullback range active)" if strict_mode else ""
            return self._no(f"Location filter failed: not near key level, not in pullback zone{gate_note}")

        # ── 3. Entry candle: rejection OR strong directional, body >= 0.5 ─
        h4_last = h4.iloc[-1]
        candle_ok = _body_ratio(h4_last) >= 0.5 and (
            _has_rejection(h4_last, direction) or _candle_is_directional(h4_last, direction)
        )
        if not candle_ok:
            return self._no("H4 entry candle not confirmed (body < 0.5 or no rejection/direction)")

        confidence += 30  # +30 valid setup (location + candle)

        # Strict mode: enforce minimum confidence before optional boosters
        # (macro + news can still push it over the line)
        # ── 4. Macro + news (optional confidence boosters) ───────────────
        macro = _macro_bias(data.get("correlation") or {})
        news = _news_payload(data.get("calendar") or {})
        macro_aligned = (direction == "BUY" and macro == "STRONG_BUY_GOLD") or \
                        (direction == "SELL" and macro == "STRONG_SELL_GOLD")
        if macro_aligned:
            confidence += 10  # +10 macro aligned
        if _news_alignment(direction, news):
            confidence += 10  # +10 news aligned

        if strict_mode and confidence < STRICT_MIN_CONFIDENCE:
            return self._no(
                f"Strict gate (post SL streak): confidence {confidence} < {STRICT_MIN_CONFIDENCE} required"
            )

        # ── 5. SL / TP ───────────────────────────────────────────────────
        sl_min = float(cfg.get("sl_min", 10.0))
        sl_max = float(cfg.get("sl_max", 25.0))
        tp1_r = float(cfg.get("tp1_r", 1.5))
        tp2_r = float(cfg.get("tp2_r", 2.5))
        tp3_r = float(cfg.get("tp3_r") or 3.0)

        swing_low = _safe_float(h4["low"].tail(8).min())
        swing_high = _safe_float(h4["high"].tail(8).max())

        if direction == "BUY":
            swing_low_anchor = swing_low if 0 < swing_low < price else price - sl_min
            sl_raw = min(price - sl_min, swing_low_anchor - 0.5)
            sl_dist = min(max(sl_min, price - sl_raw), sl_max)
            sl = round(price - sl_dist, 2)
        else:
            swing_high_anchor = swing_high if swing_high > price else price + sl_min
            sl_raw = max(price + sl_min, swing_high_anchor + 0.5)
            sl_dist = min(max(sl_min, sl_raw - price), sl_max)
            sl = round(price + sl_dist, 2)

        risk_pct = float(cfg.get("risk_pct", 1.0))
        balance = _safe_float((data.get("account") or {}).get("balance"))
        if balance <= 0:
            return self._no("Account balance unavailable")
        risk_amount = balance * (risk_pct / 100.0)
        lot = max(0.01, min(0.03, risk_amount / max(1.0, sl_dist * 100.0)))

        sign = 1 if direction == "BUY" else -1
        tp_levels = [
            round(price + sign * sl_dist * tp1_r, 2),
            round(price + sign * sl_dist * tp2_r, 2),
            round(price + sign * sl_dist * tp3_r, 2),
        ]

        reason = (
            f"{direction} swing | D1:{d1_trend} H4:{h4_trend} | "
            f"{'near level' if near_level else 'pullback'} | "
            f"body:{_body_ratio(h4_last):.2f} | conf:{confidence}"
        )

        return {
            "signal": direction,
            "entry": round(price, 2),
            "sl": sl,
            "tp": tp_levels[-1],
            "tp_levels": tp_levels,
            "sl_distance": round(sl_dist, 2),
            "lot": round(lot, 2),
            "confidence": round(confidence / 100.0, 2),
            "confidence_pct": confidence,
            "reason": reason,
            "rr": round(sl_dist * tp1_r / sl_dist, 2),
            "_strategy_name": self.name,
            "strategy": self.name,
            "decision": direction,
            "_signal_family": "SWING",
            "_sweep_confirmed": True,
            "_candle_confirmation": True,
            "_exit_profile": "swing_engine",
            "_tp_levels": tp_levels,
            "_macro_bias": macro,
            "_news": news,
            "_strict_gate_active": strict_mode,
        }

    @staticmethod
    def _no(reason: str) -> Dict:
        return {"signal": "NO_TRADE", "reason": reason, "confidence": 0.0}
