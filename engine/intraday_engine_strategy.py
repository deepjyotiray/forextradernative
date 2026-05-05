"""
Intraday Engine Strategy — same-day scalping on XAUUSD.

Entry logic (per spec):
  - London or New York session only
  - H1 and M15 must trend the same direction (UP/DOWN) — RANGE = NO TRADE
  - NO TRADE on high-impact news
  - Mandatory liquidity sweep at a known level:
      BUY:  sweep DOWN at PDL / ASIA_LOW / EQL
      SELL: sweep UP   at PDH / ASIA_HIGH / EQH
  - M5 confirmation: body_ratio >= 0.6, volume >= 1.2x, close HIGH (BUY) or LOW (SELL)

Risk (per spec):
  - Risk 0.5% per trade
  - SL: 1.5-3 pts (below/above sweep level)
  - TP: 1.5R / 2R

Confidence model (per spec):
  +50  structure aligned (H1 == M15)
  +30  valid setup (sweep at level)
  +10  macro aligned (DXY + US10Y)
  +10  news aligned
"""
from __future__ import annotations

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
    rng = max(0.01, _safe_float(candle.get("high")) - _safe_float(candle.get("low")))
    body = abs(_safe_float(candle.get("close")) - _safe_float(candle.get("open")))
    return round(body / rng, 3)


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


def _session(now_utc: datetime) -> str:
    hour = now_utc.astimezone(_UTC).hour
    if hour < 7:
        return "ASIA"
    if hour < 13:
        return "LONDON"
    if hour < 22:
        return "NEW_YORK"
    return "OFF"


def _asia_levels(m15_df: Optional[pd.DataFrame], now_utc: datetime):
    if m15_df is None or m15_df.empty:
        return 0.0, 0.0
    session_start = now_utc.astimezone(_UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    london_start = session_start + timedelta(hours=7)
    times = m15_df["datetime"] if "datetime" in m15_df.columns else m15_df.index.to_series()
    window = m15_df[(times >= session_start) & (times < london_start)]
    if window.empty:
        return 0.0, 0.0
    return _safe_float(window["high"].max()), _safe_float(window["low"].min())


def _equal_levels(df: Optional[pd.DataFrame], mode: str, tolerance: float = 0.5) -> List[float]:
    if df is None or len(df) < 20:
        return []
    values = df["high" if mode == "HIGH" else "low"].tail(40).astype(float).tolist()
    matches: List[float] = []
    for idx, value in enumerate(values):
        for other in values[idx + 1:]:
            if abs(value - other) <= tolerance:
                matches.append(round((value + other) / 2.0, 2))
                break
    return sorted(set(matches))


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


class IntradayEngineStrategy(BaseStrategy):
    name = "INTRADAY_ENGINE"

    def generate_signal(self, data: Dict) -> Dict:
        tick = data.get("tick") or {}
        price = _safe_float(tick.get("bid") or tick.get("ask"))
        if price <= 0:
            return self._no("No tick data")

        m5 = data.get("m5_df")
        m15 = data.get("m15_df")
        h1 = data.get("h1_df")
        if m5 is None or m15 is None or h1 is None:
            return self._no("Missing M5/M15/H1 data")

        cfg = _scfg.get(self.name)
        now_utc = data.get("now_utc")
        if not isinstance(now_utc, datetime):
            now_utc = datetime.now(_UTC)
        elif now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=_UTC)

        # ── 1. Session: London or New York only ──────────────────────────
        session = _session(now_utc)
        allowed_sessions = set(cfg.get("sessions", ["LONDON", "NEW_YORK"]))
        if session not in allowed_sessions:
            return self._no(f"Session {session} not allowed")

        # ── 2. News: block on high-impact ────────────────────────────────
        news = _news_payload(data.get("calendar") or {})
        if bool(cfg.get("news_block", True)) and news["high_impact"]:
            return self._no("High-impact news active")

        # ── 3. Spread ────────────────────────────────────────────────────
        spread = _safe_float(tick.get("spread"))
        spread_max = float(cfg.get("spread_max", 0.5))
        if spread > spread_max:
            return self._no(f"Spread {spread:.2f} > {spread_max}")

        # ── 4. Active trade cap (max 1 open at a time) ─────────────────
        max_active = int(cfg.get("max_active_trades", 1))
        open_count = int((data.get("strategy_trade_counts") or {}).get(self.name, 0) or 0)
        if open_count >= max_active:
            return self._no(f"Max {max_active} active intraday trade(s)")

        # ── 5. Trend: H1 == M15, no RANGE ────────────────────────────────
        h1_trend = _trend_label(h1)
        m15_trend = _trend_label(m15)
        if h1_trend != m15_trend or h1_trend == "RANGE":
            return self._no(f"H1/M15 not aligned (H1:{h1_trend} M15:{m15_trend})")

        direction = "BUY" if h1_trend == "UP" else "SELL"
        confidence = 50  # +50 structure aligned

        # ── 6. Liquidity sweep at known level ────────────────────────────
        asia_high, asia_low = _asia_levels(m15, now_utc)
        prev_day = h1.iloc[-24:] if len(h1) >= 24 else h1
        pdh = _safe_float(prev_day["high"].max())
        pdl = _safe_float(prev_day["low"].min())
        eql = _equal_levels(m15, "LOW")
        eqh = _equal_levels(m15, "HIGH")

        liquidity = data.get("liquidity") or {}
        recent_sweeps = list(liquidity.get("sweeps") or [])
        sweep = recent_sweeps[-1] if recent_sweeps else {}
        sweep_type = str(sweep.get("type") or "").upper()
        sweep_level = _safe_float(sweep.get("level"))

        if direction == "BUY":
            # Spec: sweep DOWN at PDL / ASIA_LOW / EQL
            setup_ok = sweep_type == "BUY_SWEEP"
            level_ok = any(lv > 0 and abs(sweep_level - lv) <= 0.75
                           for lv in [pdl, asia_low, *(eql or [])])
            m5_closed = m5.iloc[-2] if len(m5) >= 2 else m5.iloc[-1]
            sl_anchor = min(_safe_float(m5_closed.get("low")),
                            sweep_level or _safe_float(m5_closed.get("low")))
        else:
            # Spec: sweep UP at PDH / ASIA_HIGH / EQH
            setup_ok = sweep_type == "SELL_SWEEP"
            level_ok = any(lv > 0 and abs(sweep_level - lv) <= 0.75
                           for lv in [pdh, asia_high, *(eqh or [])])
            m5_closed = m5.iloc[-2] if len(m5) >= 2 else m5.iloc[-1]
            sl_anchor = max(_safe_float(m5_closed.get("high")),
                            sweep_level or _safe_float(m5_closed.get("high")))

        if not setup_ok or not level_ok:
            return self._no(
                f"No sweep at key level (sweep:{sweep_type} level:{sweep_level:.2f} "
                f"PDH:{pdh:.2f} PDL:{pdl:.2f} AsiaH:{asia_high:.2f} AsiaL:{asia_low:.2f})"
            )

        confidence += 30  # +30 valid setup

        # ── 7. M5 confirmation: body >= 0.6, volume >= 1.2x, close HIGH/LOW ─
        m5_last = m5.iloc[-1]
        volume_ratio = 0.0
        if len(m5) >= 6:
            volume_ratio = _safe_float(m5["volume"].tail(1).iloc[-1]) / max(
                1.0, _safe_float(m5["volume"].tail(6).iloc[:-1].mean())
            )

        body = _body_ratio(m5_last)
        close_pos = _close_position(m5_last)
        expected_close = "HIGH" if direction == "BUY" else "LOW"

        if body < 0.6:
            return self._no(f"M5 body ratio {body:.2f} < 0.6")
        if volume_ratio < 1.2:
            return self._no(f"M5 volume ratio {volume_ratio:.2f} < 1.2")
        if close_pos != expected_close:
            return self._no(f"M5 close position {close_pos} != {expected_close}")

        # ── 8. Macro + news (optional confidence boosters) ───────────────
        macro = _macro_bias(data.get("correlation") or {})
        macro_aligned = (direction == "BUY" and macro == "STRONG_BUY_GOLD") or \
                        (direction == "SELL" and macro == "STRONG_SELL_GOLD")
        if macro_aligned:
            confidence += 10  # +10 macro aligned
        if _news_alignment(direction, news):
            confidence += 10  # +10 news aligned

        # ── 9. SL / TP ───────────────────────────────────────────────────
        sl_min = float(cfg.get("sl_min", 1.5))
        sl_max = float(cfg.get("sl_max", 3.0))
        tp1_r = float(cfg.get("tp1_r", 1.5))
        tp2_r = float(cfg.get("tp2_r", 2.0))

        if direction == "BUY":
            sl_dist = max(sl_min, min(sl_max, price - sl_anchor + 0.2))
            sl = round(price - sl_dist, 2)
        else:
            sl_dist = max(sl_min, min(sl_max, sl_anchor - price + 0.2))
            sl = round(price + sl_dist, 2)

        risk_pct = float(cfg.get("risk_pct", 0.5))
        balance = _safe_float((data.get("account") or {}).get("balance"))
        risk_amount = balance * (risk_pct / 100.0)
        lot = max(0.01, min(0.03, risk_amount / max(1.0, sl_dist * 100.0)))

        sign = 1 if direction == "BUY" else -1
        tp_levels = [
            round(price + sign * sl_dist * tp1_r, 2),
            round(price + sign * sl_dist * tp2_r, 2),
        ]

        reason = (
            f"{direction} intraday | {session} | sweep:{sweep_type}@{sweep_level:.2f} | "
            f"body:{body:.2f} vol:{volume_ratio:.2f} | conf:{confidence}"
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
            "rr": round(tp1_r, 2),
            "_strategy_name": self.name,
            "strategy": self.name,
            "decision": direction,
            "_signal_family": "INTRADAY",
            "_sweep_confirmed": True,
            "_candle_confirmation": True,
            "_exit_profile": "intraday_engine",
            "_scalp": True,
            "_tp_levels": tp_levels,
            "_macro_bias": macro,
            "_news": news,
            "_entry_volume_ratio": round(volume_ratio, 3),
        }

    @staticmethod
    def _no(reason: str) -> Dict:
        return {"signal": "NO_TRADE", "reason": reason, "confidence": 0.0}
