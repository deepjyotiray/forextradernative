"""
M15 scalp deep - Strategy 2 implementation.

Model:
- H1 strong trend with at least two consecutive BOS
- trace the latest expansion origin to an unmitigated supply / demand zone
- wait for price to tap the HTF zone
- confirm on M5 with market-structure shift and a breaker block inside the HTF zone
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

import config as cfg
from engine import strategy_configs as _scfg
from engine.m15_zone_scalp_strategy import (
    M15ZoneScalpStrategy,
    _ohlc_row,
    _safe_float,
    _session_allowed,
)

_UTC = timezone.utc
_SIDE_SIGN = {"BUY": 1.0, "SELL": -1.0}


def _side_sign(direction: str) -> float:
    return _SIDE_SIGN[str(direction or "").upper()]


def _body_ratio(candle: pd.Series) -> float:
    o, h, l, cl = _ohlc_row(candle)
    return abs(cl - o) / max(0.01, h - l)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.astype(float).ewm(span=span, adjust=False).mean()


def _atr(frame: pd.DataFrame, period: int = 14) -> float:
    if frame is None or len(frame) < period + 2:
        return 0.0
    rows = frame.rename(columns=lambda c: str(c).lower())
    high = rows["high"].astype(float)
    low = rows["low"].astype(float)
    close = rows["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1] or 0.0)


def _zones_overlap(a_low: float, a_high: float, b_low: float, b_high: float, tolerance: float = 0.0) -> bool:
    return not (a_high < b_low - tolerance or b_high < a_low - tolerance)


def _price_in_zone(price: float, zone_low: float, zone_high: float, tolerance: float = 0.0) -> bool:
    return zone_low - tolerance <= price <= zone_high + tolerance


def _swing_points(highs: List[float], lows: List[float], window: int = 2) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    swing_highs: List[Tuple[int, float]] = []
    swing_lows: List[Tuple[int, float]] = []
    for i in range(window, len(highs) - window):
        if all(highs[i] >= highs[i - j] for j in range(1, window + 1)) and all(highs[i] >= highs[i + j] for j in range(1, window + 1)):
            swing_highs.append((i, float(highs[i])))
        if all(lows[i] <= lows[i - j] for j in range(1, window + 1)) and all(lows[i] <= lows[i + j] for j in range(1, window + 1)):
            swing_lows.append((i, float(lows[i])))
    return swing_highs, swing_lows


def _micro_ok(direction: str, data: Dict, s_cfg: Dict) -> Tuple[bool, str, float, float, float]:
    tick_snapshot = data.get("tick_snapshot") or {}
    tick_pressure = data.get("tick_pressure") or {}
    velocity = _safe_float(tick_snapshot.get("velocity"), _safe_float(tick_pressure.get("velocity"), 0.0))
    burst = _safe_float(tick_pressure.get("burst_rate"), 0.0)
    pressure = _safe_float(tick_pressure.get("pressure_score"), 0.0)
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
    min_signed_pressure = float(s_cfg.get("min_signed_pressure", -0.02) or -0.02)
    signed_pressure = sign * pressure
    if signed_pressure < min_signed_pressure:
        return False, f"Signed pressure {signed_pressure:.3f} < {min_signed_pressure:.3f}", velocity, burst, pressure

    opposing_bias = "SHORT" if direction == "BUY" else "LONG"
    opposing_threshold = float(s_cfg.get("opposing_bias_pressure_threshold", 0.05) or 0.05)
    if pressure_bias == opposing_bias and signed_pressure <= -opposing_threshold:
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

        s_cfg = _scfg.get(self.name)
        h1 = data.get("h1_df")
        m5 = data.get("m5_df")
        min_h1_bars = int(s_cfg.get("min_h1_bars", 40) or 40)
        min_m5_bars = int(s_cfg.get("min_m5_bars", 20) or 20)
        if h1 is None or len(h1) < min_h1_bars:
            return self._no("Insufficient H1 data")
        if m5 is None or len(m5) < min_m5_bars:
            return self._no("Insufficient M5 data")

        if spread > float(s_cfg.get("spread_max", 0.45) or 0.45):
            return self._no(f"Spread {spread:.2f} > {s_cfg.get('spread_max', 0.45)}")

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
            if bool(cal.get("blocked")) or bool(cal.get("high_impact")):
                return self._no("High-impact news active")

        regime_state = str(((data.get("regime") or {}).get("state") or "")).upper()
        if regime_state == "RANGING" and not bool(s_cfg.get("allow_ranging", False)):
            return self._no("RANGING regime blocked")

        open_count = int((data.get("strategy_trade_counts") or {}).get(self.name, 0) or 0)
        if open_count >= int(s_cfg.get("max_active_trades", 1) or 1):
            return self._no(f"Max {s_cfg.get('max_active_trades', 1)} active scalp(s)")

        htf_setup = self._find_htf_setup(h1, s_cfg)
        if not htf_setup:
            return self._no("No H1 unmitigated trend zone")

        direction = str(htf_setup["direction"])
        entry = ask if direction == "BUY" else bid
        zone = dict(htf_setup["zone"])
        zone_tolerance = max(
            float(s_cfg.get("htf_zone_touch_pts", 0.35) or 0.35),
            _atr(m5) * float(s_cfg.get("htf_zone_touch_atr_mult", 0.08) or 0.08),
        )

        if not self._htf_zone_is_active(entry, m5, zone, zone_tolerance):
            return self._no("Price has not tapped the H1 zone")

        breaker = self._find_m5_breaker_confirmation(m5, direction, zone, s_cfg)
        if not breaker:
            return self._no("No M5 structure-shift breaker confirmation")

        micro_ok, micro_reason, velocity, burst, pressure = _micro_ok(direction, data, s_cfg)
        if not micro_ok:
            return self._no(micro_reason)

        entry_tolerance = max(
            float(s_cfg.get("breaker_entry_tolerance_pts", 0.20) or 0.20),
            _atr(m5) * float(s_cfg.get("breaker_entry_tolerance_atr_mult", 0.04) or 0.04),
        )
        if not _price_in_zone(entry, breaker["zone_low"], breaker["zone_high"], entry_tolerance):
            return self._no("Price not retraced into breaker block")

        pip = max(0.01, float(s_cfg.get("pip_size", 0.1) or 0.1))
        sl = self._compute_stop(entry, direction, m5, zone, breaker, s_cfg)
        if sl is None:
            return self._no("Cannot compute valid stop")
        sl_dist = round(abs(entry - sl), 2)
        if sl_dist <= 0:
            return self._no("Invalid stop distance")

        tp = self._compute_take_profit(entry, direction, sl_dist, data, htf_setup, s_cfg)
        rr = round(abs(tp - entry) / sl_dist, 2) if sl_dist > 0 else 0.0
        min_rr = float(s_cfg.get("min_rr", 1.3) or 1.3)
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

        strength = float(htf_setup.get("strength", 0.0))
        pressure_score = float(pressure)
        confidence = 0.62 + min(0.18, strength * 0.06) + min(0.10, max(rr - 1.0, 0.0) * 0.14)
        confidence += min(0.05, max(0.0, _side_sign(direction) * pressure_score) * 0.12)

        reason = (
            f"{direction} M15 scalp deep | H1 {htf_setup['trend_note']} | "
            f"zone [{zone['zone_low']:.2f}-{zone['zone_high']:.2f}] | "
            f"M5 breaker [{breaker['zone_low']:.2f}-{breaker['zone_high']:.2f}] "
            f"| rr {rr:.2f} | vel {velocity:.2f} burst {burst:.2f} pressure {pressure_score:.3f}"
        )

        return {
            "signal": direction,
            "entry": round(entry, 2),
            "sl": sl,
            "tp": tp,
            "tp_levels": [tp],
            "sl_distance": sl_dist,
            "lot": round(lot, 2),
            "confidence": min(0.99, round(confidence, 3)),
            "confidence_pct": int(round(min(0.99, confidence) * 100)),
            "reason": reason,
            "rr": rr,
            "_strategy_name": self.name,
            "strategy": self.name,
            "decision": direction,
            "_signal_family": "M15",
            "_setup_direction": "LONG" if direction == "BUY" else "SHORT",
            "_bias_direction": "LONG" if direction == "BUY" else "SHORT",
            "_sweep_confirmed": False,
            "_m15_zone_confirmed": False,
            "_breaker_block_confirmed": True,
            "_candle_confirmation": True,
            "_body_ratio": round(float(breaker.get("body_ratio", 0.0)), 3),
            "_exit_profile": "m15_scalp_deep",
            "_scalp": True,
            "_tp_levels": [tp],
            "_htf_zone_mid": round((zone["zone_low"] + zone["zone_high"]) / 2.0, 2),
            "_breaker_mid": round((breaker["zone_low"] + breaker["zone_high"]) / 2.0, 2),
            "_tp_pips": round(abs(tp - entry) / pip, 2),
            "_pip_size": pip,
            "_session": sess_lbl,
        }

    def _find_htf_setup(self, h1: pd.DataFrame, s_cfg: Dict) -> Optional[Dict]:
        frame = h1.rename(columns=lambda c: str(c).lower()).copy()
        highs = frame["high"].astype(float).tolist()
        lows = frame["low"].astype(float).tolist()
        closes = frame["close"].astype(float).tolist()
        window = max(2, int(s_cfg.get("h1_swing_window", 2) or 2))
        min_bos = max(2, int(s_cfg.get("min_consecutive_bos", 2) or 2))
        swing_highs, swing_lows = _swing_points(highs, lows, window=window)
        if len(swing_highs) < min_bos + 1 or len(swing_lows) < min_bos + 1:
            return None

        direction = self._trend_direction(swing_highs, swing_lows, min_bos)
        if not direction:
            return None

        if direction == "BUY":
            ref_idx, ref_level = swing_highs[-2]
            bos_idx = self._find_bos_idx(closes, ref_idx, ref_level, "BUY")
            anchor_idx = max((idx for idx, _ in swing_lows if idx < bos_idx), default=max(0, bos_idx - 8))
            zone = self._find_origin_zone(frame, anchor_idx, bos_idx, "BUY")
        else:
            ref_idx, ref_level = swing_lows[-2]
            bos_idx = self._find_bos_idx(closes, ref_idx, ref_level, "SELL")
            anchor_idx = max((idx for idx, _ in swing_highs if idx < bos_idx), default=max(0, bos_idx - 8))
            zone = self._find_origin_zone(frame, anchor_idx, bos_idx, "SELL")

        if bos_idx is None or not zone:
            return None

        if not self._zone_is_unmitigated(frame, zone, len(frame) - 2):
            return None

        atr_h1 = _atr(frame)
        impulse = abs(closes[bos_idx] - closes[anchor_idx]) if bos_idx > anchor_idx else 0.0
        min_impulse = atr_h1 * float(s_cfg.get("h1_impulse_atr_mult", 1.2) or 1.2)
        if impulse < min_impulse:
            return None

        trend_note = f"{direction} 2x BOS @ {ref_level:.2f}"
        return {
            "direction": direction,
            "zone": zone,
            "bos_idx": bos_idx,
            "bos_level": round(float(ref_level), 2),
            "trend_note": trend_note,
            "strength": round(impulse / max(atr_h1, 0.01), 2),
        }

    @staticmethod
    def _trend_direction(swing_highs: List[Tuple[int, float]], swing_lows: List[Tuple[int, float]], min_bos: int) -> str:
        highs = [price for _, price in swing_highs[-(min_bos + 1):]]
        lows = [price for _, price in swing_lows[-(min_bos + 1):]]
        if all(highs[i] > highs[i - 1] for i in range(1, len(highs))) and all(lows[i] > lows[i - 1] for i in range(1, len(lows))):
            return "BUY"
        if all(highs[i] < highs[i - 1] for i in range(1, len(highs))) and all(lows[i] < lows[i - 1] for i in range(1, len(lows))):
            return "SELL"
        return ""

    @staticmethod
    def _find_bos_idx(closes: List[float], ref_idx: int, ref_level: float, direction: str) -> Optional[int]:
        for idx in range(ref_idx + 1, len(closes)):
            if direction == "BUY" and float(closes[idx]) > ref_level:
                return idx
            if direction == "SELL" and float(closes[idx]) < ref_level:
                return idx
        return None

    @staticmethod
    def _find_origin_zone(frame: pd.DataFrame, anchor_idx: int, bos_idx: int, direction: str) -> Optional[Dict]:
        start = max(0, anchor_idx)
        for idx in range(bos_idx - 1, start - 1, -1):
            candle = frame.iloc[idx]
            open_price, high, low, close = _ohlc_row(candle)
            if direction == "BUY" and close < open_price:
                return {
                    "idx": idx,
                    "zone_low": round(low, 2),
                    "zone_high": round(max(open_price, close), 2),
                }
            if direction == "SELL" and close > open_price:
                return {
                    "idx": idx,
                    "zone_low": round(min(open_price, close), 2),
                    "zone_high": round(high, 2),
                }
        return None

    @staticmethod
    def _zone_is_unmitigated(frame: pd.DataFrame, zone: Dict, last_closed_idx: int) -> bool:
        zone_low = float(zone["zone_low"])
        zone_high = float(zone["zone_high"])
        zone_idx = int(zone["idx"])
        if last_closed_idx <= zone_idx + 1:
            return True
        for idx in range(zone_idx + 1, last_closed_idx + 1):
            candle = frame.iloc[idx]
            high = _safe_float(candle.get("high"))
            low = _safe_float(candle.get("low"))
            if _zones_overlap(low, high, zone_low, zone_high):
                return False
        return True

    @staticmethod
    def _htf_zone_is_active(entry: float, m5: pd.DataFrame, zone: Dict, tolerance: float) -> bool:
        zone_low = float(zone["zone_low"])
        zone_high = float(zone["zone_high"])
        if _price_in_zone(entry, zone_low, zone_high, tolerance):
            return True
        recent = m5.tail(6).rename(columns=lambda c: str(c).lower())
        for _, candle in recent.iterrows():
            high = _safe_float(candle.get("high"))
            low = _safe_float(candle.get("low"))
            if _zones_overlap(low, high, zone_low, zone_high, tolerance):
                return True
        return False

    def _find_m5_breaker_confirmation(self, m5: pd.DataFrame, direction: str, htf_zone: Dict, s_cfg: Dict) -> Optional[Dict]:
        frame = m5.tail(max(20, int(s_cfg.get("m5_lookback_bars", 24) or 24))).copy()
        frame = frame.rename(columns=lambda c: str(c).lower()).reset_index(drop=True)
        zone_low = float(htf_zone["zone_low"])
        zone_high = float(htf_zone["zone_high"])
        overlap_tol = max(
            float(s_cfg.get("breaker_overlap_pts", 0.15) or 0.15),
            _atr(frame) * float(s_cfg.get("breaker_overlap_atr_mult", 0.03) or 0.03),
        )
        for touch_idx in self._touch_indices(frame, zone_low, zone_high, overlap_tol):
            if touch_idx >= len(frame) - 2:
                continue
            pre_start = max(0, touch_idx - int(s_cfg.get("m5_structure_reference_bars", 4) or 4))
            if direction == "BUY":
                struct_level = float(frame.iloc[pre_start:touch_idx + 1]["high"].max())
                shift_idx = self._find_shift_idx(frame, touch_idx, struct_level, "BUY")
            else:
                struct_level = float(frame.iloc[pre_start:touch_idx + 1]["low"].min())
                shift_idx = self._find_shift_idx(frame, touch_idx, struct_level, "SELL")
            if shift_idx is None:
                continue

            breaker = self._find_breaker_block(frame, touch_idx, shift_idx, direction, zone_low, zone_high, overlap_tol)
            if not breaker:
                continue
            breaker["shift_idx"] = shift_idx
            breaker["struct_level"] = round(struct_level, 2)
            return breaker
        return None

    @staticmethod
    def _touch_indices(frame: pd.DataFrame, zone_low: float, zone_high: float, tolerance: float) -> List[int]:
        indices: List[int] = []
        for idx in range(len(frame) - 2, max(-1, len(frame) - 16), -1):
            candle = frame.iloc[idx]
            high = _safe_float(candle.get("high"))
            low = _safe_float(candle.get("low"))
            if _zones_overlap(low, high, zone_low, zone_high, tolerance):
                indices.append(idx)
        return indices

    @staticmethod
    def _find_shift_idx(frame: pd.DataFrame, touch_idx: int, struct_level: float, direction: str) -> Optional[int]:
        for idx in range(touch_idx + 1, len(frame)):
            close = _safe_float(frame.iloc[idx].get("close"))
            if direction == "BUY" and close > struct_level:
                return idx
            if direction == "SELL" and close < struct_level:
                return idx
        return None

    @staticmethod
    def _find_breaker_block(
        frame: pd.DataFrame,
        touch_idx: int,
        shift_idx: int,
        direction: str,
        zone_low: float,
        zone_high: float,
        tolerance: float,
    ) -> Optional[Dict]:
        for idx in range(shift_idx - 1, touch_idx - 1, -1):
            candle = frame.iloc[idx]
            open_price, high, low, close = _ohlc_row(candle)
            if direction == "BUY" and close < open_price:
                breaker_low = round(low, 2)
                breaker_high = round(max(open_price, close), 2)
            elif direction == "SELL" and close > open_price:
                breaker_low = round(min(open_price, close), 2)
                breaker_high = round(high, 2)
            else:
                continue
            if not _zones_overlap(breaker_low, breaker_high, zone_low, zone_high, tolerance):
                continue
            return {
                "idx": idx,
                "zone_low": breaker_low,
                "zone_high": breaker_high,
                "body_ratio": _body_ratio(candle),
            }
        return None

    @staticmethod
    def _compute_stop(entry: float, direction: str, m5: pd.DataFrame, htf_zone: Dict, breaker: Dict, s_cfg: Dict) -> Optional[float]:
        recent = m5.tail(max(4, int(s_cfg.get("m5_stop_lookback", 6) or 6))).rename(columns=lambda c: str(c).lower())
        pip = max(0.01, float(s_cfg.get("pip_size", 0.1) or 0.1))
        sl_buffer = max(
            float(s_cfg.get("sl_buffer_atr_mult", 0.20) or 0.20) * max(_atr(m5), 0.01),
            float(s_cfg.get("sl_buffer_pips", 3) or 3) * pip,
        )
        sl_floor = float(s_cfg.get("sl_floor_pips", 12) or 12) * pip
        sl_cap = float(s_cfg.get("sl_ceiling_pips", 40) or 40) * pip

        if direction == "BUY":
            anchor = min(
                float(recent["low"].min()),
                float(htf_zone["zone_low"]),
                float(breaker["zone_low"]),
            ) - sl_buffer
            sl = round(anchor, 2)
            sl_dist = entry - sl
            if sl_dist < sl_floor:
                sl = round(entry - sl_floor, 2)
                sl_dist = entry - sl
        else:
            anchor = max(
                float(recent["high"].max()),
                float(htf_zone["zone_high"]),
                float(breaker["zone_high"]),
            ) + sl_buffer
            sl = round(anchor, 2)
            sl_dist = sl - entry
            if sl_dist < sl_floor:
                sl = round(entry + sl_floor, 2)
                sl_dist = sl - entry

        if sl_dist <= 0 or sl_dist > sl_cap:
            return None
        return sl

    def _compute_take_profit(self, entry: float, direction: str, sl_dist: float, data: Dict, htf_setup: Dict, s_cfg: Dict) -> float:
        target_rr = float(s_cfg.get("target_rr", 1.8) or 1.8)
        fallback = entry + (sl_dist * target_rr) if direction == "BUY" else entry - (sl_dist * target_rr)

        levels: List[float] = []
        key_levels = ((data.get("liquidity") or {}).get("key_levels") or {})
        for key in ("session_high", "prev_day_high", "session_low", "prev_day_low"):
            value = _safe_float(key_levels.get(key))
            if value > 0:
                levels.append(value)
        bos_level = _safe_float(htf_setup.get("bos_level"))
        if bos_level > 0:
            levels.append(bos_level)

        h1 = data.get("h1_df")
        if h1 is not None and len(h1) >= 10:
            rows = h1.rename(columns=lambda c: str(c).lower())
            highs = rows["high"].astype(float).tolist()
            lows = rows["low"].astype(float).tolist()
            swing_highs, swing_lows = _swing_points(highs, lows, window=2)
            if direction == "BUY":
                levels.extend(price for _, price in swing_highs if price > entry)
            else:
                levels.extend(price for _, price in swing_lows if price < entry)

        min_rr = float(s_cfg.get("min_rr", 1.3) or 1.3)
        if direction == "BUY":
            for candidate in sorted(v for v in levels if v > entry):
                rr = (candidate - entry) / max(sl_dist, 1e-6)
                if rr >= min_rr:
                    return round(candidate, 2)
        else:
            for candidate in sorted((v for v in levels if v < entry), reverse=True):
                rr = (entry - candidate) / max(sl_dist, 1e-6)
                if rr >= min_rr:
                    return round(candidate, 2)
        return round(fallback, 2)
