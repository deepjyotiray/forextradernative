from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd

from engine.strategies.base_strategy import BaseStrategy
from engine import strategy_configs as _scfg


_UTC = timezone.utc


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
    high = _safe_float(candle.get("high"))
    low = _safe_float(candle.get("low"))
    spread = max(0.01, high - low)
    body = abs(_safe_float(candle.get("close")) - _safe_float(candle.get("open")))
    return round(body / spread, 3)


def _close_position(candle: pd.Series) -> str:
    high = _safe_float(candle.get("high"))
    low = _safe_float(candle.get("low"))
    close = _safe_float(candle.get("close"))
    rng = max(0.01, high - low)
    pct = (close - low) / rng
    if pct >= 0.67:
        return "HIGH"
    if pct <= 0.33:
        return "LOW"
    return "MID"


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


def _session(now_utc: datetime) -> str:
    hour = now_utc.astimezone(_UTC).hour
    if hour < 7:
        return "ASIA"
    if hour < 13:
        return "LONDON"
    if hour < 22:
        return "NEW_YORK"
    return "OFF"


def _weekly_levels(d1_df: Optional[pd.DataFrame]) -> tuple[float, float]:
    if d1_df is None or len(d1_df) < 5:
        return 0.0, 0.0
    recent = d1_df.tail(5)
    return _safe_float(recent["high"].max()), _safe_float(recent["low"].min())


def _asia_levels(m15_df: Optional[pd.DataFrame], now_utc: datetime) -> tuple[float, float]:
    if m15_df is None or m15_df.empty:
        return 0.0, 0.0
    session_start = now_utc.astimezone(_UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    london_start = session_start + timedelta(hours=7)
    window = m15_df[m15_df["datetime"] >= session_start]
    window = window[window["datetime"] < london_start]
    if window.empty:
        return 0.0, 0.0
    return _safe_float(window["high"].max()), _safe_float(window["low"].min())


def _equal_levels(df: Optional[pd.DataFrame], mode: str, tolerance: float = 0.5) -> List[float]:
    if df is None or len(df) < 20:
        return []
    values = df["high" if mode == "HIGH" else "low"].tail(40).astype(float).tolist()
    matches: List[float] = []
    for idx, value in enumerate(values):
        for other in values[idx + 1 :]:
            if abs(value - other) <= tolerance:
                matches.append(round((value + other) / 2.0, 2))
                break
    return sorted(set(matches))


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
    title = str(event.get("title") or "NONE").upper()
    event_name = "NONE"
    for candidate in ("CPI", "NFP", "FOMC"):
        if candidate in title:
            event_name = candidate
            break
    return {
        "high_impact": bool(calendar_state.get("blocked") or event.get("critical")),
        "actual": _safe_float(event.get("actual")),
        "forecast": _safe_float(event.get("forecast")),
        "event": event_name,
    }


def _news_alignment(direction: str, news: Dict) -> bool:
    actual = _safe_float(news.get("actual"))
    forecast = _safe_float(news.get("forecast"))
    if actual == 0.0 and forecast == 0.0:
        return False
    if direction == "BUY":
        return actual < forecast
    return actual > forecast


def _candle_is_directional(candle: pd.Series, direction: str) -> bool:
    open_ = _safe_float(candle.get("open"))
    close = _safe_float(candle.get("close"))
    return close > open_ if direction == "BUY" else close < open_


def _signal_payload(
    strategy: str,
    direction: str,
    entry: float,
    sl: float,
    tp_levels: List[float],
    lot: float,
    confidence: int,
    reason: str,
    extra: Dict | None = None,
) -> Dict:
    payload = {
        "signal": direction,
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "tp": round(tp_levels[-1], 2),
        "tp_levels": [round(level, 2) for level in tp_levels],
        "sl_distance": round(abs(entry - sl), 2),
        "lot": round(max(0.01, lot), 2),
        "confidence": round(max(0.0, min(confidence / 100.0, 1.0)), 2),
        "confidence_pct": int(max(0, min(confidence, 100))),
        "reason": reason,
        "_strategy_name": strategy,
    }
    if extra:
        payload.update(extra)
    return payload


def _no_trade(strategy: str, reason: str, confidence: int = 0, extra: Dict | None = None) -> Dict:
    payload = {
        "signal": "NO_TRADE",
        "confidence": round(max(0.0, min(confidence / 100.0, 1.0)), 2),
        "confidence_pct": int(max(0, min(confidence, 100))),
        "reason": reason,
        "_strategy_name": strategy,
    }
    if extra:
        payload.update(extra)
    return payload


@dataclass
class _CommonContext:
    now_utc: datetime
    price: float
    spread: float
    session: str
    m5: Optional[pd.DataFrame]
    m15: Optional[pd.DataFrame]
    h1: Optional[pd.DataFrame]
    h4: Optional[pd.DataFrame]
    d1: Optional[pd.DataFrame]
    zones: Dict
    liquidity: Dict
    correlation: Dict
    calendar: Dict
    account: Dict
    explicit_trade_counts: Dict


def _build_common(data: Dict) -> _CommonContext:
    tick = data.get("tick") or {}
    now_utc = data.get("now_utc")
    if not isinstance(now_utc, datetime):
        now_utc = datetime.now(_UTC)
    price = _safe_float(tick.get("bid") or tick.get("ask"))
    return _CommonContext(
        now_utc=now_utc.astimezone(_UTC),
        price=price,
        spread=_safe_float(tick.get("spread")),
        session=_session(now_utc),
        m5=data.get("m5_df"),
        m15=data.get("m15_df"),
        h1=data.get("h1_df"),
        h4=data.get("h4_df"),
        d1=data.get("d1_df"),
        zones=data.get("zones") or {},
        liquidity=data.get("liquidity") or {},
        correlation=data.get("correlation") or {},
        calendar=data.get("calendar") or {},
        account=data.get("account") or {},
        explicit_trade_counts=data.get("strategy_trade_counts") or {},
    )


class SwingEngineStrategy(BaseStrategy):
    name = "SWING_ENGINE"

    def generate_signal(self, data: Dict) -> Dict:
        ctx = _build_common(data)
        if ctx.price <= 0 or ctx.h4 is None or ctx.d1 is None or ctx.h4.empty or ctx.d1.empty:
            return _no_trade(self.name, "Missing H4/D1 data")

        _scfg_swing = _scfg.get(self.name)
        spread_max = float(_scfg_swing.get("spread_max", 0.7))
        if ctx.spread > spread_max:
            return _no_trade(self.name, f"Swing spread too high: {ctx.spread:.2f} > {spread_max}")
        max_active = int(_scfg_swing.get("max_active_trades", 1))
        swing_open = int(ctx.explicit_trade_counts.get(self.name, 0) or 0)
        if swing_open >= max_active:
            return _no_trade(self.name, f"Swing max {max_active} active trade(s)")

        d1_trend = _trend_label(ctx.d1)
        h4_trend = _trend_label(ctx.h4)
        if d1_trend == "UP" and h4_trend == "UP":
            direction = "BUY"
        elif d1_trend == "DOWN" and h4_trend == "DOWN":
            direction = "SELL"
        else:
            return _no_trade(self.name, "D1/H4 not aligned", 0, {"decision": "NO_TRADE"})

        confidence = 50  # +50 structure aligned
        h4_last = ctx.h4.iloc[-1]
        weekly_high, weekly_low = _weekly_levels(ctx.d1)
        prev_day = ctx.d1.iloc[-2] if len(ctx.d1) >= 2 else ctx.d1.iloc[-1]
        support_levels = [_nearest_zone(ctx.zones.get("support") or [], ctx.price), _safe_float(prev_day.get("low")), weekly_low]
        resistance_levels = [_nearest_zone(ctx.zones.get("resistance") or [], ctx.price), _safe_float(prev_day.get("high")), weekly_high]
        atr_h4 = max(1.0, _safe_float((ctx.h4["high"] - ctx.h4["low"]).tail(14).mean(), 12.0))
        level_tolerance = max(3.0, min(8.0, atr_h4 * 0.35))
        pullback = _pullback_depth(ctx.price, ctx.h4, direction)

        if direction == "BUY":
            near_level = any(level > 0 and abs(ctx.price - level) <= level_tolerance for level in support_levels)
        else:
            near_level = any(level > 0 and abs(ctx.price - level) <= level_tolerance for level in resistance_levels)
        valid_location = near_level or (0.30 <= pullback <= 0.70)
        if not valid_location:
            return _no_trade(self.name, "Location filter failed", confidence, {"decision": "NO_TRADE"})
        confidence += 30  # +30 valid setup

        candle_ok = _body_ratio(h4_last) >= 0.5 and (
            _has_rejection(h4_last, direction) or _candle_is_directional(h4_last, direction)
        )
        if not candle_ok:
            return _no_trade(self.name, "H4 entry candle not confirmed", confidence, {"decision": "NO_TRADE"})

        macro = _macro_bias(ctx.correlation)
        news = _news_payload(ctx.calendar)
        macro_aligned = (direction == "BUY" and macro == "STRONG_BUY_GOLD") or (
            direction == "SELL" and macro == "STRONG_SELL_GOLD"
        )
        if macro_aligned:
            confidence += 10  # +10 macro aligned
        if _news_alignment(direction, news):
            confidence += 10  # +10 news aligned

        entry = ctx.price
        swing_low = _safe_float(ctx.h4["low"].tail(8).min())
        swing_high = _safe_float(ctx.h4["high"].tail(8).max())
        sl_min = float(_scfg_swing.get("sl_min", 10.0))
        sl_max = float(_scfg_swing.get("sl_max", 25.0))
        tp1_r = float(_scfg_swing.get("tp1_r", 1.5))
        tp2_r = float(_scfg_swing.get("tp2_r", 2.5))
        tp3_r = float(_scfg_swing.get("tp3_r") or 3.0)
        if direction == "BUY":
            sl = min(entry - sl_min, swing_low - 0.5 if swing_low > 0 else entry - sl_min)
            sl = entry - min(max(sl_min, entry - sl), sl_max)
        else:
            sl = max(entry + sl_min, swing_high + 0.5 if swing_high > 0 else entry + sl_min)
            sl = entry + min(max(sl_min, sl - entry), sl_max)
        risk_pct = float(_scfg_swing.get("risk_pct", 1.0))
        risk_amount = _safe_float(ctx.account.get("balance"), 0.0) * (risk_pct / 100.0)
        sl_distance = max(sl_min, min(sl_max, abs(entry - sl)))
        lot = risk_amount / max(1.0, sl_distance * 100.0)
        tp_levels = [
            entry + (sl_distance * tp1_r if direction == "BUY" else -sl_distance * tp1_r),
            entry + (sl_distance * tp2_r if direction == "BUY" else -sl_distance * tp2_r),
            entry + (sl_distance * tp3_r if direction == "BUY" else -sl_distance * tp3_r),
        ]
        reason = f"{direction} swing: D1/H4 aligned, valid location, H4 candle confirmed"
        return _signal_payload(
            self.name,
            direction,
            entry,
            sl,
            tp_levels,
            lot,
            confidence,
            reason,
            {
                "strategy": self.name,
                "decision": direction,
                "_signal_family": "SWING",
                "_sweep_confirmed": True,
                "_candle_confirmation": True,
                "_exit_profile": "swing_engine",
                "_tp_levels": tp_levels,
                "_macro_bias": macro,
                "_news": news,
            },
        )


class IntradayEngineStrategy(BaseStrategy):
    name = "INTRADAY_ENGINE"

    def generate_signal(self, data: Dict) -> Dict:
        ctx = _build_common(data)
        if ctx.price <= 0 or ctx.m15 is None or ctx.h1 is None or ctx.m5 is None:
            return _no_trade(self.name, "Missing M5/M15/H1 data")
        _scfg_intra = _scfg.get(self.name)
        allowed_sessions = set(_scfg_intra.get("sessions", ["LONDON", "NEW_YORK"]))
        if ctx.session not in allowed_sessions:
            return _no_trade(self.name, "Intraday session closed")

        news = _news_payload(ctx.calendar)
        news_block = bool(_scfg_intra.get("news_block", True))
        if news_block and news["high_impact"]:
            return _no_trade(self.name, "High-impact news active")
        spread_max = float(_scfg_intra.get("spread_max", 0.5))
        if ctx.spread > spread_max:
            return _no_trade(self.name, f"Spread too high: {ctx.spread:.2f} > {spread_max}")
        max_trades_day = int(_scfg_intra.get("max_trades_day", 5))
        if int(ctx.explicit_trade_counts.get(self.name, 0) or 0) >= max_trades_day:
            return _no_trade(self.name, "Intraday daily trade cap reached")

        h1_trend = _trend_label(ctx.h1)
        m15_trend = _trend_label(ctx.m15)
        if h1_trend != m15_trend or h1_trend == "RANGE":
            return _no_trade(self.name, "H1/M15 trend not aligned")

        direction = "BUY" if h1_trend == "UP" else "SELL"
        confidence = 50  # +50 structure aligned
        m5_last = ctx.m5.iloc[-1]
        volume_ratio = 0.0
        if len(ctx.m5) >= 6:
            volume_ratio = _safe_float(ctx.m5["volume"].tail(1).iloc[-1]) / max(
                1.0, _safe_float(ctx.m5["volume"].tail(6).iloc[:-1].mean())
            )

        asia_high, asia_low = _asia_levels(ctx.m15, ctx.now_utc)
        prev_day = ctx.h1.iloc[-24:] if len(ctx.h1) >= 24 else ctx.h1
        pdh = _safe_float(prev_day["high"].max())
        pdl = _safe_float(prev_day["low"].min())
        eql = _equal_levels(ctx.m15, "LOW")
        eqh = _equal_levels(ctx.m15, "HIGH")
        recent_sweeps = list(ctx.liquidity.get("sweeps") or [])
        sweep = recent_sweeps[-1] if recent_sweeps else {}
        sweep_type = str(sweep.get("type") or "").upper()
        sweep_level = _safe_float(sweep.get("level"))

        if direction == "BUY":
            setup_ok = sweep_type == "BUY_SWEEP"
            level_ok = any(level > 0 and abs(sweep_level - level) <= 0.75 for level in [asia_low, pdl, *(eql or [])])
            close_ok = _close_position(m5_last) == "HIGH"
            sl_anchor = min(_safe_float(m5_last.get("low")), sweep_level or _safe_float(m5_last.get("low")))
        else:
            setup_ok = sweep_type == "SELL_SWEEP"
            level_ok = any(level > 0 and abs(sweep_level - level) <= 0.75 for level in [asia_high, pdh, *(eqh or [])])
            close_ok = _close_position(m5_last) == "LOW"
            sl_anchor = max(_safe_float(m5_last.get("high")), sweep_level or _safe_float(m5_last.get("high")))

        if not setup_ok or not level_ok:
            return _no_trade(self.name, "Mandatory liquidity setup missing", confidence)
        confidence += 30  # +30 valid setup

        if _body_ratio(m5_last) < 0.6 or volume_ratio < 1.2 or not close_ok:
            return _no_trade(self.name, "M5 confirmation failed", confidence)

        macro = _macro_bias(ctx.correlation)
        macro_aligned = (direction == "BUY" and macro == "STRONG_BUY_GOLD") or (
            direction == "SELL" and macro == "STRONG_SELL_GOLD"
        )
        if macro_aligned:
            confidence += 10  # +10 macro aligned
        if _news_alignment(direction, news):
            confidence += 10  # +10 news aligned

        sl_min = float(_scfg_intra.get("sl_min", 1.5))
        sl_max = float(_scfg_intra.get("sl_max", 3.0))
        tp1_r = float(_scfg_intra.get("tp1_r", 1.5))
        tp2_r = float(_scfg_intra.get("tp2_r", 2.0))
        risk_pct = float(_scfg_intra.get("risk_pct", 0.5))
        entry = ctx.price
        if direction == "BUY":
            sl_distance = max(sl_min, min(sl_max, entry - sl_anchor + 0.2))
            sl = entry - sl_distance
        else:
            sl_distance = max(sl_min, min(sl_max, sl_anchor - entry + 0.2))
            sl = entry + sl_distance
        risk_amount = _safe_float(ctx.account.get("balance"), 0.0) * (risk_pct / 100.0)
        lot = risk_amount / max(1.0, sl_distance * 100.0)
        tp_levels = [
            entry + (sl_distance * tp1_r if direction == "BUY" else -sl_distance * tp1_r),
            entry + (sl_distance * tp2_r if direction == "BUY" else -sl_distance * tp2_r),
        ]
        reason = f"{direction} intraday: {ctx.session} sweep reclaim with M5 confirmation"
        return _signal_payload(
            self.name,
            direction,
            entry,
            sl,
            tp_levels,
            lot,
            confidence,
            reason,
            {
                "strategy": self.name,
                "decision": direction,
                "_signal_family": "INTRADAY",
                "_sweep_confirmed": True,
                "_candle_confirmation": True,
                "_exit_profile": "intraday_engine",
                "_tp_levels": tp_levels,
                "_scalp": True,
                "_macro_bias": macro,
                "_news": news,
                "_entry_volume_ratio": round(volume_ratio, 3),
            },
        )
