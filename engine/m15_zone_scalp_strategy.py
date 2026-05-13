"""
M15 zone scalp — XAUUSD: demand/supply touches on M15 clusters + HTF bias + reaction candle.

Demand long: bid ≥ zone_low, touch distance to band, D1/H4 bullish (or bias fallback),
entry after bullish rejection/displacement on last closed M15.

Supply short: bid ≤ zone_high, touch distance to band, D1/H4 bearish (or bias fallback),
entry after bearish rejection/displacement on last closed M15.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config as cfg
from engine.indicators import atr as atr_np
from engine.session_filter import get_session_at
from engine.strategies.base_strategy import BaseStrategy
from engine import strategy_configs as _scfg
from engine.zones import ZoneDetector

_UTC = timezone.utc


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _dist_to_interval(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return float(lo - x)
    if x > hi:
        return float(x - hi)
    return 0.0


def _m15_atr(m15: pd.DataFrame) -> float:
    if m15 is None or len(m15) < 16:
        return 0.0
    df = m15.rename(columns=lambda c: str(c).lower())
    hi = df["high"].astype(float).values
    lo = df["low"].astype(float).values
    cl = df["close"].astype(float).values
    out = atr_np(hi, lo, cl, 14)
    v = float(out[-1])
    return v if np.isfinite(v) and v > 0 else 0.0


def _session_allowed(now_utc: datetime, allowed: List[str]) -> Tuple[bool, str]:
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=_UTC)
    else:
        now_utc = now_utc.astimezone(_UTC)
    token = str(get_session_at(now_utc) or "ASIAN").upper()
    ok = token in {s.upper() for s in allowed}
    return ok, token


def _resolve_htf_tf(market_state: Dict, tf: str) -> str:
    trend = market_state.get("trend") or {}
    structure = market_state.get("structure") or {}
    s = str(structure.get(tf, "RANGE")).upper()
    return s if s != "RANGE" else str(trend.get(tf, "RANGE")).upper()


def _htf_bullish(market_state: Dict) -> bool:
    """Aligned with swing-style long: D1 UP and H4 UP or RANGE."""
    if not market_state:
        return False
    d1 = _resolve_htf_tf(market_state, "D1")
    h4 = _resolve_htf_tf(market_state, "H4")
    return d1 == "UP" and h4 in ("UP", "RANGE")


def _htf_bearish(market_state: Dict) -> bool:
    d1 = _resolve_htf_tf(market_state, "D1")
    h4 = _resolve_htf_tf(market_state, "H4")
    return d1 == "DOWN" and h4 in ("DOWN", "RANGE")


def _bias_fallback_long(data: Dict, min_conf: float) -> bool:
    b = data.get("bias") or {}
    return str(b.get("direction") or "").upper() == "LONG" and _safe_float(b.get("confidence")) >= min_conf


def _bias_fallback_short(data: Dict, min_conf: float) -> bool:
    b = data.get("bias") or {}
    return str(b.get("direction") or "").upper() == "SHORT" and _safe_float(b.get("confidence")) >= min_conf


def _bullish_htf_ok(data: Dict, s_cfg: Dict) -> bool:
    ms = data.get("market_state") or {}
    if _htf_bullish(ms):
        return True
    if _htf_bearish(ms):
        return False
    return bool(
        not s_cfg.get("require_htf_bias", True)
        or _bias_fallback_long(data, float(s_cfg.get("bias_fallback_min_confidence", 0.35) or 0.35))
    )


def _bearish_htf_ok(data: Dict, s_cfg: Dict) -> bool:
    ms = data.get("market_state") or {}
    if _htf_bearish(ms):
        return True
    if _htf_bullish(ms):
        return False
    return bool(
        not s_cfg.get("require_htf_bias", True)
        or _bias_fallback_short(data, float(s_cfg.get("bias_fallback_min_confidence", 0.35) or 0.35))
    )


def _ohlc_row(row: pd.Series) -> Tuple[float, float, float, float]:
    for keys in (
        ("open", "high", "low", "close"),
        ("Open", "High", "Low", "Close"),
    ):
        try:
            return tuple(float(row[k]) for k in keys)
        except Exception:
            continue
    raise KeyError("missing OHLC columns on M15 row")


def _bullish_reaction(
    candle: pd.Series,
    rej_wick: float,
    body_min: float,
    close_pos_min: float,
) -> bool:
    o, h, l, cl = _ohlc_row(candle)
    rng = max(0.01, h - l)
    lower_wick = min(o, cl) - l
    upper_wick = h - max(o, cl)
    body = abs(cl - o)
    close_pos = (cl - l) / rng
    rejection = lower_wick / rng >= rej_wick and cl > o
    displacement = (
        cl > o and body / rng >= body_min and close_pos >= close_pos_min and upper_wick / rng <= 0.45
    )
    return rejection or displacement


def _bearish_reaction(
    candle: pd.Series,
    rej_wick: float,
    body_min: float,
    close_pos_min: float,
) -> bool:
    o, h, l, cl = _ohlc_row(candle)
    rng = max(0.01, h - l)
    upper_wick = h - max(o, cl)
    lower_wick = min(o, cl) - l
    body = abs(cl - o)
    close_pos = (h - cl) / rng
    rejection = upper_wick / rng >= rej_wick and cl < o
    displacement = (
        cl < o and body / rng >= body_min and close_pos >= close_pos_min and lower_wick / rng <= 0.45
    )
    return rejection or displacement


class M15ZoneScalpStrategy(BaseStrategy):
    name = "M15_ZONE_SCALP"

    def generate_signal(self, data: Dict) -> Dict:
        symbol = str(data.get("symbol") or getattr(cfg, "SYMBOL", "XAUUSD")).upper()
        if "XAU" not in symbol:
            return self._no(f"Strategy tuned for gold (got {symbol})")

        tick = data.get("tick") or {}
        bid = _safe_float(tick.get("bid"))
        ask = _safe_float(tick.get("ask") or tick.get("bid"))
        spread = _safe_float(tick.get("spread"))
        if bid <= 0:
            return self._no("No tick data")

        m15 = data.get("m15_df")
        if m15 is None or len(m15) < 24:
            return self._no("Insufficient M15 data")

        s_cfg = _scfg.get(self.name)
        if spread > float(s_cfg.get("spread_max", 0.5) or 0.5):
            return self._no(f"Spread {spread:.2f} > {s_cfg.get('spread_max', 0.5)}")

        now_utc = data.get("now_utc")
        if not isinstance(now_utc, datetime):
            now_utc = datetime.now(_UTC)
        elif now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=_UTC)

        sess_ok, sess_lbl = _session_allowed(now_utc, list(s_cfg.get("sessions") or ["LONDON", "NEW_YORK"]))
        if not sess_ok:
            return self._no(f"Session {sess_lbl} not allowed")

        if bool(s_cfg.get("news_block", True)):
            cal = data.get("calendar") or {}
            if bool(cal.get("blocked")):
                return self._no("High-impact news (calendar blocked)")

        open_count = int((data.get("strategy_trade_counts") or {}).get(self.name, 0) or 0)
        if open_count >= int(s_cfg.get("max_active_trades", 1) or 1):
            return self._no(f"Max {s_cfg.get('max_active_trades', 1)} active scalp(s)")

        atr_m15 = _m15_atr(m15)
        if atr_m15 <= 0:
            return self._no("Invalid M15 ATR")

        pip = max(0.01, float(s_cfg.get("pip_size", 0.1) or 0.1))
        tp_pips = float(s_cfg.get("tp_pips", 25) or 25)

        proximity = max(
            float(s_cfg.get("zone_touch_floor_pts", 0.8) or 0.8),
            atr_m15 * float(s_cfg.get("zone_touch_atr_mult", 0.25) or 0.25),
        )

        zd = ZoneDetector(
            eps=float(s_cfg.get("zone_detector_eps", 1.2) or 1.2),
            min_samples=int(s_cfg.get("zone_detector_min_samples", 2) or 2),
            max_width=float(s_cfg.get("zone_detector_max_width", 5.0) or 5.0),
        )
        frame = m15.copy()
        frame.columns = [str(c).lower() for c in frame.columns]
        clusters = zd.scored_clusters(frame)
        if not clusters:
            return self._no("No M15 zones")

        min_str = float(s_cfg.get("min_zone_strength", 0.15) or 0.15)

        buy_zone = self._pick_demand_zone(bid, clusters, proximity, min_str)
        sell_zone = self._pick_supply_zone(bid, clusters, proximity, min_str)

        direction: Optional[str] = None
        active: Optional[Tuple[int, Dict]] = None

        demand_htf = buy_zone is not None and _bullish_htf_ok(data, s_cfg)
        supply_htf = sell_zone is not None and _bearish_htf_ok(data, s_cfg)

        if buy_zone and demand_htf and not (sell_zone and supply_htf):
            direction, active = "BUY", buy_zone
        elif sell_zone and supply_htf and not (buy_zone and demand_htf):
            direction, active = "SELL", sell_zone
        elif buy_zone and demand_htf and sell_zone and supply_htf:
            bd = buy_zone[1]
            sd = sell_zone[1]
            q_buy = float(bd.get("strength", 0)) + max(0.0, proximity - _dist_to_interval(bid, bd["zone_low"], bd["zone_high"]))
            q_sel = float(sd.get("strength", 0)) + max(0.0, proximity - _dist_to_interval(bid, sd["zone_low"], sd["zone_high"]))
            if q_buy >= q_sel:
                direction, active = "BUY", buy_zone
            else:
                direction, active = "SELL", sell_zone

        if not direction or not active:
            if buy_zone and not demand_htf:
                return self._no("Demand touch but HTF / bias not bullish")
            if sell_zone and not supply_htf:
                return self._no("Supply touch but HTF / bias not bearish")
            return self._no("No qualifying demand/supply touch with HTF alignment")

        _zi, zn = active

        rej = float(s_cfg.get("rejection_wick_ratio", 0.32) or 0.32)
        body_min = float(s_cfg.get("displacement_body_ratio", 0.52) or 0.52)
        close_near = float(s_cfg.get("close_near_extreme_ratio", 0.62) or 0.62)

        closed = frame.iloc[-2] if len(frame) >= 2 else frame.iloc[-1]
        body_ratio = 0.0
        if direction == "BUY":
            if not _bullish_reaction(closed, rej, body_min, close_near):
                return self._no("No bullish rejection/displacement on last closed M15")
        else:
            if not _bearish_reaction(closed, rej, body_min, close_near):
                return self._no("No bearish rejection/displacement on last closed M15")
        try:
            o, h, l, cl = _ohlc_row(closed)
            body_ratio = abs(cl - o) / max(0.01, h - l)
        except Exception:
            body_ratio = 0.0

        zl = float(zn["zone_low"])
        zh = float(zn["zone_high"])
        zm = float(zn["zone_mid"])

        sl_buf = atr_m15 * float(s_cfg.get("sl_buffer_atr_mult", 0.35) or 0.35)
        sl_floor = float(s_cfg.get("sl_floor_pips", 12) or 12) * pip
        sl_cap = float(s_cfg.get("sl_ceiling_pips", 35) or 35) * pip

        if direction == "BUY":
            entry = bid
            sl_raw = zl - sl_buf
            sl_dist = max(sl_floor, min(sl_cap, entry - sl_raw))
            sl = round(entry - sl_dist, 2)
            tp_dist = tp_pips * pip
            tp = round(entry + tp_dist, 2)
        else:
            entry = ask if ask > 0 else bid
            sl_raw = zh + sl_buf
            sl_dist = max(sl_floor, min(sl_cap, sl_raw - entry))
            sl = round(entry + sl_dist, 2)
            tp_dist = tp_pips * pip
            tp = round(entry - tp_dist, 2)

        rr = tp_dist / max(sl_dist, 1e-6)
        confidence = round(0.55 + float(zn.get("strength", 0.0)) * 0.35, 2)

        balance = _safe_float((data.get("account") or {}).get("balance"))
        if balance <= 0:
            return self._no("Account balance unavailable")

        fl_raw = s_cfg.get("fixed_lot")
        fixed_lot = _safe_float(fl_raw) if fl_raw not in (None, "", False) else 0.0
        if fixed_lot >= 0.01:
            lot = min(0.1, fixed_lot)
        else:
            risk_amount = balance * (_safe_float(s_cfg.get("risk_pct"), 0.5) / 100.0)
            lot = max(0.01, min(0.05, risk_amount / max(1.0, sl_dist * 100.0)))

        tp_levels = [tp]
        ms = data.get("market_state") or {}
        htf_note = f"D1:{_resolve_htf_tf(ms,'D1')} H4:{_resolve_htf_tf(ms,'H4')}" if ms else "bias-only"
        reason = (
            f"{direction} M15 zone scalp | demand/supply | {htf_note} | zone_mid:{zm} [{zl}-{zh}] "
            f"str:{zn.get('strength')} touch≤{proximity:.2f} | TP {tp_pips} pips ({tp_dist:.2f})"
        )

        return {
            "signal": direction,
            "entry": round(entry, 2),
            "sl": sl,
            "tp": tp,
            "tp_levels": tp_levels,
            "sl_distance": round(sl_dist, 2),
            "lot": round(lot, 2),
            "confidence": min(0.99, confidence),
            "confidence_pct": int(round(confidence * 100)),
            "reason": reason,
            "rr": round(rr, 2),
            "_strategy_name": self.name,
            "strategy": self.name,
            "decision": direction,
            "_signal_family": "M15",
            "_setup_direction": "LONG" if direction == "BUY" else "SHORT",
            "_bias_direction": "LONG" if direction == "BUY" else "SHORT",
            "_sweep_confirmed": False,
            "_m15_zone_confirmed": True,
            "_candle_confirmation": True,
            "_body_ratio": round(body_ratio, 3),
            "_exit_profile": "m15_zone_scalp",
            "_scalp": True,
            "_tp_levels": tp_levels,
            "_zone_mid": zm,
            "_tp_pips": tp_pips,
            "_pip_size": pip,
            "_session": sess_lbl,
        }

    @staticmethod
    def _pick_demand_zone(
        bid: float,
        clusters: List[Dict],
        proximity: float,
        min_str: float,
    ) -> Optional[Tuple[int, Dict]]:
        """Demand: bid is in the upper half of the zone or above it, within touch distance."""
        best: Optional[Tuple[int, Dict, float]] = None
        for i, z in enumerate(clusters):
            st = float(z.get("strength", 0.0))
            if st < min_str:
                continue
            zl = float(z["zone_low"])
            zh = float(z["zone_high"])
            zm = float(z.get("zone_mid", (zl + zh) / 2.0))
            if bid < zl:
                continue
            if bid < zm:
                continue
            dist = _dist_to_interval(bid, zl, zh)
            if dist > proximity:
                continue
            quality = st + (proximity - dist) / max(proximity, 1e-6) * 0.15
            if best is None or quality > best[2]:
                best = (i, z, quality)
        if best is None:
            return None
        return best[0], best[1]

    @staticmethod
    def _pick_supply_zone(
        bid: float,
        clusters: List[Dict],
        proximity: float,
        min_str: float,
    ) -> Optional[Tuple[int, Dict]]:
        """Supply: bid is in the lower half of the zone or below it, within touch distance."""
        best: Optional[Tuple[int, Dict, float]] = None
        for i, z in enumerate(clusters):
            st = float(z.get("strength", 0.0))
            if st < min_str:
                continue
            zl = float(z["zone_low"])
            zh = float(z["zone_high"])
            zm = float(z.get("zone_mid", (zl + zh) / 2.0))
            if bid > zh:
                continue
            if bid >= zm:
                continue
            dist = _dist_to_interval(bid, zl, zh)
            if dist > proximity:
                continue
            quality = st + (proximity - dist) / max(proximity, 1e-6) * 0.15
            if best is None or quality > best[2]:
                best = (i, z, quality)
        if best is None:
            return None
        return best[0], best[1]

    @staticmethod
    def _no(reason: str) -> Dict:
        return {"signal": "NO_TRADE", "reason": reason, "confidence": 0.0}
