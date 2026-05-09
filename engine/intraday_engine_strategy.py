"""
Intraday Engine Strategy — same-day scalping on XAUUSD.

All trend labels, Asia levels, equal levels, prev-day levels, and body ratio
are consumed from data["market_state"] — computed once by the engine.
"""
from __future__ import annotations

from datetime import datetime, timezone
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


def _session(now_utc: datetime) -> str:
    hour = now_utc.astimezone(_UTC).hour
    if hour < 7:  return "ASIAN"
    if hour < 13: return "LONDON"
    if hour < 22: return "NEW_YORK"
    return "OFF"


def _body_ratio(candle: pd.Series) -> float:
    rng  = max(0.01, _safe_float(candle.get("high")) - _safe_float(candle.get("low")))
    body = abs(_safe_float(candle.get("close")) - _safe_float(candle.get("open")))
    return round(body / rng, 3)


def _close_position(candle: pd.Series) -> str:
    high  = _safe_float(candle.get("high"))
    low   = _safe_float(candle.get("low"))
    close = _safe_float(candle.get("close"))
    pct   = (close - low) / max(0.01, high - low)
    if pct >= 0.67: return "HIGH"
    if pct <= 0.33: return "LOW"
    return "MID"


def _macro_bias(correlation: Dict) -> str:
    dxy   = str((correlation.get("dxy")   or {}).get("trend") or "").upper()
    us10y = str((correlation.get("us10y") or {}).get("trend") or "").upper()
    if dxy == "UP"   and us10y == "UP":   return "STRONG_SELL_GOLD"
    if dxy == "DOWN" and us10y == "DOWN": return "STRONG_BUY_GOLD"
    return "NEUTRAL"


def _news_payload(calendar_state: Dict) -> Dict:
    event = calendar_state.get("next_event") or {}
    return {
        "high_impact": bool(calendar_state.get("blocked")),
        "actual":      _safe_float(event.get("actual")),
        "forecast":    _safe_float(event.get("forecast")),
    }


def _news_alignment(direction: str, news: Dict) -> bool:
    actual   = _safe_float(news.get("actual"))
    forecast = _safe_float(news.get("forecast"))
    if actual == 0.0 and forecast == 0.0:
        return False
    return actual < forecast if direction == "BUY" else actual > forecast


class IntradayEngineStrategy(BaseStrategy):
    name = "INTRADAY_ENGINE"

    def generate_signal(self, data: Dict) -> Dict:
        tick  = data.get("tick") or {}
        price = _safe_float(tick.get("bid") or tick.get("ask"))
        if price <= 0:
            return self._no("No tick data")

        m5  = data.get("m5_df")
        m15 = data.get("m15_df")
        h1  = data.get("h1_df")
        if m5 is None or m15 is None or h1 is None:
            return self._no("Missing M5/M15/H1 data")

        cfg     = _scfg.get(self.name)
        now_utc = data.get("now_utc")
        if not isinstance(now_utc, datetime):
            now_utc = datetime.now(_UTC)
        elif now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=_UTC)

        # ── 1. Session ────────────────────────────────────────────────────
        session = _session(now_utc)
        if session not in set(cfg.get("sessions", ["LONDON", "NEW_YORK"])):
            return self._no(f"Session {session} not allowed")

        # ── 2. News ───────────────────────────────────────────────────────
        news = _news_payload(data.get("calendar") or {})
        if bool(cfg.get("news_block", True)) and news["high_impact"]:
            return self._no("High-impact news active")

        # ── 3. Spread ─────────────────────────────────────────────────────
        spread = _safe_float(tick.get("spread"))
        spread_max = float(cfg.get("spread_max", 0.5))
        if spread > spread_max:
            return self._no(f"Spread {spread:.2f} > {spread_max}")

        # ── 4. Active trade cap ───────────────────────────────────────────
        open_count = int((data.get("strategy_trade_counts") or {}).get(self.name, 0) or 0)
        if open_count >= int(cfg.get("max_active_trades", 1)):
            return self._no(f"Max {cfg.get('max_active_trades', 1)} active intraday trade(s)")

        # ── 5. Trend from market_state ────────────────────────────────────
        ms    = data.get("market_state") or {}
        trend = ms.get("trend") or {}
        h1_trend  = str(trend.get("H1",  "RANGE")).upper()
        m15_trend = str(trend.get("M15", "RANGE")).upper()
        if h1_trend != m15_trend or h1_trend == "RANGE":
            return self._no(f"H1/M15 not aligned (H1:{h1_trend} M15:{m15_trend})")

        direction  = "BUY" if h1_trend == "UP" else "SELL"
        confidence = 50

        # ── 6. Liquidity sweep at known level ─────────────────────────────
        asia    = ms.get("asia_levels") or {}
        prev    = ms.get("prev_day") or {}
        eql     = ms.get("equal_lows")  or []
        eqh     = ms.get("equal_highs") or []
        asia_high = _safe_float(asia.get("asia_high"))
        asia_low  = _safe_float(asia.get("asia_low"))
        pdh       = _safe_float(prev.get("pdh"))
        pdl       = _safe_float(prev.get("pdl"))

        liquidity     = data.get("liquidity") or {}
        recent_sweeps = list(liquidity.get("sweeps") or [])
        sweep         = recent_sweeps[-1] if recent_sweeps else {}
        sweep_type    = str(sweep.get("type") or "").upper()
        sweep_level   = _safe_float(sweep.get("level"))

        if direction == "BUY":
            setup_ok  = sweep_type == "BUY_SWEEP"
            level_ok  = any(lv > 0 and abs(sweep_level - lv) <= 0.75
                            for lv in [pdl, asia_low, *eql])
            m5_closed = m5.iloc[-2] if len(m5) >= 2 else m5.iloc[-1]
            sl_anchor = min(_safe_float(m5_closed.get("low")),
                            sweep_level or _safe_float(m5_closed.get("low")))
        else:
            setup_ok  = sweep_type == "SELL_SWEEP"
            level_ok  = any(lv > 0 and abs(sweep_level - lv) <= 0.75
                            for lv in [pdh, asia_high, *eqh])
            m5_closed = m5.iloc[-2] if len(m5) >= 2 else m5.iloc[-1]
            sl_anchor = max(_safe_float(m5_closed.get("high")),
                            sweep_level or _safe_float(m5_closed.get("high")))

        if not setup_ok or not level_ok:
            return self._no(
                f"No sweep at key level (sweep:{sweep_type} level:{sweep_level:.2f} "
                f"PDH:{pdh:.2f} PDL:{pdl:.2f} AsiaH:{asia_high:.2f} AsiaL:{asia_low:.2f})"
            )

        confidence += 30

        # ── 7. M5 confirmation ────────────────────────────────────────────
        m5_last      = m5.iloc[-1]
        volume_ratio = 0.0
        if len(m5) >= 6:
            volume_ratio = _safe_float(m5["volume"].tail(1).iloc[-1]) / max(
                1.0, _safe_float(m5["volume"].tail(6).iloc[:-1].mean())
            )

        body      = _body_ratio(m5_last)
        close_pos = _close_position(m5_last)
        expected  = "HIGH" if direction == "BUY" else "LOW"

        body_min = 0.6
        volume_ratio_min = float(cfg.get("m5_volume_ratio_min", 1.2))

        if body < body_min:
            return self._no(f"M5 body ratio {body:.2f} < {body_min}")
        if volume_ratio < volume_ratio_min:
            return self._no(f"M5 volume ratio {volume_ratio:.2f} < {volume_ratio_min}")
        if close_pos != expected:
            return self._no(f"M5 close position {close_pos} != {expected}")

        # ── 8. Macro + news boosters ──────────────────────────────────────
        macro = _macro_bias(data.get("correlation") or {})
        if (direction == "BUY"  and macro == "STRONG_BUY_GOLD") or \
           (direction == "SELL" and macro == "STRONG_SELL_GOLD"):
            confidence += 10
        if _news_alignment(direction, news):
            confidence += 10

        min_confidence = int(cfg.get("min_confidence_pct", 80) or 80)
        if confidence < min_confidence:
            return self._no(f"Confidence {confidence} < {min_confidence}")

        # ── 9. SL / TP ────────────────────────────────────────────────────
        sl_min = float(cfg.get("sl_min", 1.5))
        sl_max = float(cfg.get("sl_max", 3.0))
        tp1_r  = float(cfg.get("tp1_r", 1.5))
        tp2_r  = float(cfg.get("tp2_r", 2.0))

        if direction == "BUY":
            sl_dist = max(sl_min, min(sl_max, price - sl_anchor + 0.2))
            sl      = round(price - sl_dist, 2)
        else:
            sl_dist = max(sl_min, min(sl_max, sl_anchor - price + 0.2))
            sl      = round(price + sl_dist, 2)

        balance     = _safe_float((data.get("account") or {}).get("balance"))
        risk_amount = balance * (float(cfg.get("risk_pct", 0.5)) / 100.0)
        lot         = max(0.01, min(0.03, risk_amount / max(1.0, sl_dist * 100.0)))

        sign      = 1 if direction == "BUY" else -1
        tp_levels = [
            round(price + sign * sl_dist * tp1_r, 2),
            round(price + sign * sl_dist * tp2_r, 2),
        ]

        reason = (
            f"{direction} intraday | {session} | sweep:{sweep_type}@{sweep_level:.2f} | "
            f"body:{body:.2f} vol:{volume_ratio:.2f} | conf:{confidence}"
        )

        return {
            "signal": direction, "entry": round(price, 2),
            "sl": sl, "tp": tp_levels[-1], "tp_levels": tp_levels,
            "sl_distance": round(sl_dist, 2), "lot": round(lot, 2),
            "confidence": round(confidence / 100.0, 2), "confidence_pct": confidence,
            "reason": reason, "rr": round(tp1_r, 2),
            "_strategy_name": self.name, "strategy": self.name, "decision": direction,
            "_signal_family": "INTRADAY", "_sweep_confirmed": True, "_candle_confirmation": True,
            "_exit_profile": "intraday_engine", "_scalp": True,
            "_tp_levels": tp_levels, "_macro_bias": macro, "_news": news,
            "_entry_volume_ratio": round(volume_ratio, 3),
        }

    @staticmethod
    def _no(reason: str) -> Dict:
        return {"signal": "NO_TRADE", "reason": reason, "confidence": 0.0}
