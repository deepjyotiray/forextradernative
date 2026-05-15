"""
M15 scalp deep - stricter M15 zone scalp for XAUUSD.

Key differences from M15_ZONE_SCALP:
- requires true higher-timeframe agreement instead of mixed-state bias fallback
- requires stronger M15 structure and micro execution quality
- enforces a configurable minimum reward-to-risk
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Tuple

import pandas as pd

import config as cfg
from engine import strategy_configs as _scfg
from engine.m15_zone_scalp_strategy import (
    M15ZoneScalpStrategy,
    _dist_to_interval,
    _m15_atr,
    _ohlc_row,
    _resolve_htf_tf,
    _safe_float,
    _session_allowed,
)
from engine.zones import ZoneDetector

_UTC = timezone.utc
_SIDE_SIGN = {"BUY": 1.0, "SELL": -1.0}
_SIDE_BIAS = {"BUY": "LONG", "SELL": "SHORT"}
_SIDE_TREND = {"BUY": "UP", "SELL": "DOWN"}


def _side_sign(direction: str) -> float:
    return _SIDE_SIGN[str(direction or "").upper()]


def _body_ratio(candle: pd.Series) -> float:
    o, h, l, cl = _ohlc_row(candle)
    return abs(cl - o) / max(0.01, h - l)


def _closed_candle_hits_zone(direction: str, candle: pd.Series, zone_low: float, zone_high: float, buffer_pts: float) -> bool:
    _o, h, l, _cl = _ohlc_row(candle)
    if str(direction).upper() == "BUY":
        return l <= (zone_high + buffer_pts)
    return h >= (zone_low - buffer_pts)


def _ema_alignment(m15: pd.DataFrame, direction: str, min_slope: float) -> Tuple[bool, float, float]:
    frame = m15.rename(columns=lambda c: str(c).lower())
    closes = frame["close"].astype(float)
    if len(closes) < 24:
        return False, 0.0, 0.0

    ema_fast = closes.ewm(span=9, adjust=False).mean()
    ema_slow = closes.ewm(span=21, adjust=False).mean()
    fast_now = float(ema_fast.iloc[-2])
    fast_prev = float(ema_fast.iloc[-5])
    slow_now = float(ema_slow.iloc[-2])
    slope = (fast_now - fast_prev) / 3.0
    gap = fast_now - slow_now
    sign = _side_sign(direction)
    ok = (sign * gap) > 0 and (sign * slope) >= min_slope
    return ok, slope, gap


def _strict_htf_alignment(data: Dict, direction: str, min_bias_conf: float) -> Tuple[bool, str]:
    market_state = data.get("market_state") or {}
    d1 = _resolve_htf_tf(market_state, "D1")
    h4 = _resolve_htf_tf(market_state, "H4")
    bias = data.get("bias") or {}
    bias_dir = str(bias.get("direction") or "").upper()
    bias_conf = _safe_float(bias.get("confidence"), 0.0)
    htf_note = f"D1:{d1} H4:{h4}"
    trend = _SIDE_TREND[str(direction or "").upper()]
    bias_target = _SIDE_BIAS[str(direction or "").upper()]

    if d1 == trend and h4 == trend:
        return True, htf_note
    if d1 == "RANGE" and h4 == trend and bias_dir == bias_target and bias_conf >= min_bias_conf:
        return True, htf_note
    return False, htf_note


def _reaction_ok(direction: str, candle: pd.Series, rej_wick: float, body_min: float, close_pos_min: float) -> bool:
    o, h, l, cl = _ohlc_row(candle)
    rng = max(0.01, h - l)
    sign = _side_sign(direction)
    favorable_close = (sign * (cl - o)) > 0
    reaction_wick = (min(o, cl) - l) if sign > 0 else (h - max(o, cl))
    opposite_wick = (h - max(o, cl)) if sign > 0 else (min(o, cl) - l)
    close_pos = ((cl - l) / rng) if sign > 0 else ((h - cl) / rng)
    body = abs(cl - o)
    rejection = (reaction_wick / rng) >= rej_wick and favorable_close
    displacement = (
        favorable_close
        and (body / rng) >= body_min
        and close_pos >= close_pos_min
        and (opposite_wick / rng) <= 0.45
    )
    return rejection or displacement


def _sequence_ok(direction: str, prior_close: float, current_close: float) -> bool:
    return (_side_sign(direction) * (current_close - prior_close)) > 0


def _pick_zone(direction: str, bid: float, clusters: List[Dict], proximity: float, min_str: float) -> Tuple[int, Dict] | None:
    sign = _side_sign(direction)
    best: Tuple[int, Dict, float] | None = None
    for i, z in enumerate(clusters):
        strength = float(z.get("strength", 0.0))
        if strength < min_str:
            continue
        zone_low = float(z["zone_low"])
        zone_high = float(z["zone_high"])
        zone_mid = float(z.get("zone_mid", (zone_low + zone_high) / 2.0))
        outer_boundary = zone_low if sign > 0 else zone_high
        if (sign * (bid - outer_boundary)) < 0:
            continue
        if (sign * (bid - zone_mid)) < 0:
            continue
        dist = _dist_to_interval(bid, zone_low, zone_high)
        if dist > proximity:
            continue
        quality = strength + ((proximity - dist) / max(proximity, 1e-6) * 0.15)
        if best is None or quality > best[2]:
            best = (i, z, quality)
    if best is None:
        return None
    return best[0], best[1]


def _micro_ok(direction: str, data: Dict, s_cfg: Dict) -> Tuple[bool, str, float, float, float | None]:
    tick_snapshot = data.get("tick_snapshot") or {}
    tick_pressure = data.get("tick_pressure") or {}
    velocity = _safe_float(tick_snapshot.get("velocity"), _safe_float(tick_pressure.get("velocity"), 0.0))
    burst = _safe_float(tick_pressure.get("burst_rate"), 0.0)
    pressure_raw = tick_pressure.get("pressure_score")
    pressure = None if pressure_raw is None else _safe_float(pressure_raw, 0.0)
    pressure_bias = str(
        tick_pressure.get("directional_bias")
        or tick_pressure.get("bias")
        or ""
    ).upper()

    if velocity < float(s_cfg.get("min_entry_velocity", 5.0) or 5.0):
        return False, f"Velocity {velocity:.2f} < {s_cfg.get('min_entry_velocity', 5.0)}", velocity, burst, pressure
    if burst < float(s_cfg.get("min_entry_burst_rate", 3.0) or 3.0):
        return False, f"Burst {burst:.2f} < {s_cfg.get('min_entry_burst_rate', 3.0)}", velocity, burst, pressure

    sign = _side_sign(direction)
    signed_pressure = None if pressure is None else (sign * pressure)
    min_signed_pressure = float(s_cfg.get("min_signed_pressure", -0.02) or -0.02)
    if signed_pressure is not None and signed_pressure < min_signed_pressure:
        return False, f"Signed pressure {signed_pressure:.3f} < {min_signed_pressure:.3f}", velocity, burst, pressure

    opposing_bias_threshold = float(s_cfg.get("opposing_bias_pressure_threshold", 0.05) or 0.05)
    aligned_bias = _SIDE_BIAS[str(direction or "").upper()]
    opposing_bias = "SHORT" if aligned_bias == "LONG" else "LONG"
    if pressure_bias == opposing_bias and signed_pressure is not None and signed_pressure <= -opposing_bias_threshold:
        return False, f"Opposing {opposing_bias} pressure bias", velocity, burst, pressure

    return True, "MICRO_OK", velocity, burst, pressure


class M15ScalpDeepStrategy(M15ZoneScalpStrategy):
    name = "M15_SCALP_DEEP"

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
        if m15 is None or len(m15) < 40:
            return self._no("Insufficient M15 data")

        s_cfg = _scfg.get(self.name)
        if spread > float(s_cfg.get("spread_max", 0.35) or 0.35):
            return self._no(f"Spread {spread:.2f} > {s_cfg.get('spread_max', 0.35)}")

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

        regime_state = str(((data.get("regime") or {}).get("state") or "")).upper()
        if regime_state == "RANGING" and not bool(s_cfg.get("allow_ranging", False)):
            return self._no("RANGING regime blocked")

        open_count = int((data.get("strategy_trade_counts") or {}).get(self.name, 0) or 0)
        if open_count >= int(s_cfg.get("max_active_trades", 1) or 1):
            return self._no(f"Max {s_cfg.get('max_active_trades', 1)} active scalp(s)")

        atr_m15 = _m15_atr(m15)
        if atr_m15 <= 0:
            return self._no("Invalid M15 ATR")

        pip = max(0.01, float(s_cfg.get("pip_size", 0.1) or 0.1))
        tp_min_dist = float(s_cfg.get("tp_min_pips", 28) or 28) * pip
        tp_max_dist = float(s_cfg.get("tp_max_pips", 60) or 60) * pip
        target_rr = float(s_cfg.get("target_rr", 1.3) or 1.3)
        min_rr = float(s_cfg.get("min_rr", 1.15) or 1.15)

        proximity = max(
            float(s_cfg.get("zone_touch_floor_pts", 0.6) or 0.6),
            atr_m15 * float(s_cfg.get("zone_touch_atr_mult", 0.18) or 0.18),
        )
        reaction_zone_buffer = max(
            float(s_cfg.get("reaction_zone_buffer_pts", 0.25) or 0.25),
            atr_m15 * float(s_cfg.get("reaction_zone_buffer_atr_mult", 0.10) or 0.10),
        )

        zd = ZoneDetector(
            eps=float(s_cfg.get("zone_detector_eps", 1.0) or 1.0),
            min_samples=int(s_cfg.get("zone_detector_min_samples", 2) or 2),
            max_width=float(s_cfg.get("zone_detector_max_width", 4.0) or 4.0),
        )
        frame = m15.copy()
        frame.columns = [str(c).lower() for c in frame.columns]
        clusters = zd.scored_clusters(frame)
        if not clusters:
            return self._no("No M15 zones")

        min_strength = float(s_cfg.get("min_zone_strength", 0.55) or 0.55)
        max_zone_width = atr_m15 * float(s_cfg.get("max_zone_width_atr_mult", 0.85) or 0.85)

        zones_by_direction = {
            "BUY": _pick_zone("BUY", bid, clusters, proximity, min_strength),
            "SELL": _pick_zone("SELL", bid, clusters, proximity, min_strength),
        }

        candidates: List[Tuple[float, str, Tuple[int, Dict], str, float, float, float | None]] = []
        closed = frame.iloc[-2]
        prior = frame.iloc[-3]
        min_body_ratio = float(s_cfg.get("min_signal_body_ratio", 0.32) or 0.32)
        rejection_wick = float(s_cfg.get("rejection_wick_ratio", 0.36) or 0.36)
        displacement_body = float(s_cfg.get("displacement_body_ratio", 0.58) or 0.58)
        close_near = float(s_cfg.get("close_near_extreme_ratio", 0.68) or 0.68)
        ema_slope_min = float(s_cfg.get("ema_slope_min", 0.05) or 0.05)

        for direction in ("BUY", "SELL"):
            active = zones_by_direction[direction]
            if not active:
                continue
            _micro_allowed, micro_reason, velocity, burst, pressure = _micro_ok(direction, data, s_cfg)

            _i, zone = active
            zone_low = float(zone["zone_low"])
            zone_high = float(zone["zone_high"])
            zone_width = max(0.01, zone_high - zone_low)
            if zone_width > max_zone_width:
                continue

            htf_ok, note = _strict_htf_alignment(
                data,
                direction,
                float(s_cfg.get("bias_fallback_min_confidence", 0.70) or 0.70),
            )
            if not htf_ok:
                continue

            reaction_ok = _reaction_ok(direction, closed, rejection_wick, displacement_body, close_near)
            trend_ok, ema_slope, ema_gap = _ema_alignment(frame, direction, ema_slope_min)
            prior_close = float(prior["close"])
            closed_close = float(closed["close"])
            sequence_ok = _sequence_ok(direction, prior_close, closed_close)

            if not reaction_ok or not trend_ok or not sequence_ok:
                continue
            if not _closed_candle_hits_zone(direction, closed, zone_low, zone_high, reaction_zone_buffer):
                continue

            closed_body_ratio = _body_ratio(closed)
            if closed_body_ratio < min_body_ratio:
                continue
            if micro_reason != "MICRO_OK":
                continue

            strength = float(zone.get("strength", 0.0))
            quality = strength + min(0.25, abs(ema_gap) * 0.05) + min(0.20, abs(ema_slope) * 0.10)
            candidates.append((quality, direction, active, note, velocity, burst, pressure))

        if not candidates:
            return self._no("No deep-confluence M15 zone setup")

        candidates.sort(key=lambda item: item[0], reverse=True)
        _quality, direction, active, htf_note, velocity, burst, pressure = candidates[0]
        _zi, zone = active

        zone_low = float(zone["zone_low"])
        zone_high = float(zone["zone_high"])
        zone_mid = float(zone.get("zone_mid", (zone_low + zone_high) / 2.0))
        sl_buffer = atr_m15 * float(s_cfg.get("sl_buffer_atr_mult", 0.25) or 0.25)
        sl_floor = float(s_cfg.get("sl_floor_pips", 10) or 10) * pip
        sl_cap = float(s_cfg.get("sl_ceiling_pips", 28) or 28) * pip
        sign = _side_sign(direction)
        entry = (ask if ask > 0 else bid) if direction == "BUY" else (bid if bid > 0 else ask)
        sl_anchor = zone_low if direction == "BUY" else zone_high
        sl_raw = sl_anchor - (sign * sl_buffer)
        raw_sl_distance = sign * (entry - sl_raw)
        sl_dist = max(sl_floor, min(sl_cap, raw_sl_distance))
        sl = round(entry - (sign * sl_dist), 2)
        tp_dist = max(tp_min_dist, min(tp_max_dist, sl_dist * target_rr))
        tp = round(entry + (sign * tp_dist), 2)

        rr = tp_dist / max(sl_dist, 1e-6)
        if rr < min_rr:
            return self._no(f"RR {rr:.2f} < {min_rr:.2f}")

        balance = _safe_float((data.get("account") or {}).get("balance"))
        if balance <= 0:
            return self._no("Account balance unavailable")

        fixed_lot_raw = s_cfg.get("fixed_lot")
        fixed_lot = _safe_float(fixed_lot_raw) if fixed_lot_raw not in (None, "", False) else 0.0
        if fixed_lot >= 0.01:
            lot = min(0.10, fixed_lot)
        else:
            risk_amount = balance * (_safe_float(s_cfg.get("risk_pct"), 0.35) / 100.0)
            lot = max(0.01, min(0.05, risk_amount / max(1.0, sl_dist * 100.0)))

        strength = float(zone.get("strength", 0.0))
        pressure_score = 0.0 if pressure is None else float(pressure)
        confidence = 0.60 + min(0.24, strength * 0.20) + min(0.10, max(rr - 1.0, 0.0) * 0.15)
        confidence += min(0.03, max(0.0, sign * pressure_score) * 0.10)

        tp_levels = [tp]
        reason = (
            f"{direction} M15 scalp deep | {htf_note} | zone_mid:{zone_mid:.2f} "
            f"[{zone_low:.2f}-{zone_high:.2f}] str:{strength:.2f} touch<={proximity:.2f} "
            f"| rr {rr:.2f} | vel {velocity:.2f} burst {burst:.2f} pressure {pressure_score:.3f}"
        )

        return {
            "signal": direction,
            "entry": round(entry, 2),
            "sl": sl,
            "tp": tp,
            "tp_levels": tp_levels,
            "sl_distance": round(sl_dist, 2),
            "lot": round(lot, 2),
            "confidence": min(0.99, round(confidence, 3)),
            "confidence_pct": int(round(min(0.99, confidence) * 100)),
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
            "_body_ratio": round(_body_ratio(closed), 3),
            "_exit_profile": "m15_scalp_deep",
            "_scalp": True,
            "_tp_levels": tp_levels,
            "_zone_mid": zone_mid,
            "_tp_pips": round(tp_dist / pip, 2),
            "_pip_size": pip,
            "_session": sess_lbl,
        }
