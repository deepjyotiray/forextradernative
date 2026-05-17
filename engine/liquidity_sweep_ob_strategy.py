"""
Liquidity Sweep + Order Block strategy.

Setup model:
1. HTF liquidity sweep
2. HTF break of structure away from the sweep
3. Return into the originating order block
4. LTF violated FVG -> IFVG confirmation inside / around the block
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

from . import strategy_configs as _scfg
from .indicators import atr
from .liquidity import detect_fvg, detect_liquidity_sweeps
from .session_filter import get_session_at
from .strategies.base_strategy import BaseStrategy


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _candle_body_ratio(candle: pd.Series) -> float:
    high = _safe_float(candle.get("high"))
    low = _safe_float(candle.get("low"))
    open_price = _safe_float(candle.get("open"))
    close = _safe_float(candle.get("close"))
    rng = max(0.0, high - low)
    if rng <= 0:
        return 0.0
    return abs(close - open_price) / rng


def _normalize_now(now_utc) -> datetime:
    if not isinstance(now_utc, datetime):
        return datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        return now_utc.replace(tzinfo=timezone.utc)
    return now_utc.astimezone(timezone.utc)


def _zone_contains(price: float, zone_low: float, zone_high: float, tolerance: float) -> bool:
    return zone_low - tolerance <= price <= zone_high + tolerance


def _zones_overlap(a_low: float, a_high: float, b_low: float, b_high: float, tolerance: float) -> bool:
    return not (a_high < b_low - tolerance or b_high < a_low - tolerance)


def _touches_zone(df: pd.DataFrame, zone_low: float, zone_high: float, tolerance: float) -> bool:
    if df is None or df.empty:
        return False
    low = float(df["low"].min())
    high = float(df["high"].max())
    return _zones_overlap(low, high, zone_low, zone_high, tolerance)


def _swing_points(h: List[float], l: List[float], window: int = 2) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    highs: List[Tuple[int, float]] = []
    lows: List[Tuple[int, float]] = []
    for i in range(window, len(h) - window):
        if all(h[i] >= h[i - j] for j in range(1, window + 1)) and all(h[i] >= h[i + j] for j in range(1, window + 1)):
            highs.append((i, float(h[i])))
        if all(l[i] <= l[i - j] for j in range(1, window + 1)) and all(l[i] <= l[i + j] for j in range(1, window + 1)):
            lows.append((i, float(l[i])))
    return highs, lows


class LiquiditySweepOrderBlockStrategy(BaseStrategy):
    name = "LIQUIDITY_SWEEP_OB"

    def generate_signal(self, data: Dict) -> Dict:
        s_cfg = _scfg.get(self.name)
        tick = data.get("tick") or {}
        bid = _safe_float(tick.get("bid"))
        ask = _safe_float(tick.get("ask") or tick.get("bid"))
        spread = _safe_float(tick.get("spread"))
        if bid <= 0:
            return self._no("No tick data")

        spread_max = float(s_cfg.get("spread_max", 0.6) or 0.6)
        if spread > spread_max:
            return self._no(f"Spread {spread:.2f} > {spread_max}")

        now_utc = _normalize_now(data.get("now_utc"))
        session = get_session_at(now_utc)
        allowed_sessions = [str(s).upper() for s in (s_cfg.get("sessions") or ["LONDON", "NEW_YORK"])]
        if session not in allowed_sessions:
            return self._no(f"Session {session} not allowed")

        if bool(s_cfg.get("news_block", True)):
            cal = data.get("calendar") or {}
            if bool(cal.get("blocked")) or bool(cal.get("high_impact")):
                return self._no("High-impact news active")

        open_count = int((data.get("strategy_trade_counts") or {}).get(self.name, 0) or 0)
        max_active = int(s_cfg.get("max_active_trades", 1) or 1)
        if open_count >= max_active:
            return self._no(f"Max {max_active} active trade(s)")

        htf_df, htf_label = self._pick_htf_frame(data, s_cfg)
        if htf_df is None:
            return self._no("No HTF frame available")
        ltf_df, ltf_label = self._pick_ltf_frame(data, s_cfg)
        if ltf_df is None:
            return self._no("No LTF frame available")

        setup = self._find_recent_htf_setup(htf_df, s_cfg)
        if not setup:
            return self._no(f"No HTF sweep+BOS setup on {htf_label}")

        direction = str(setup["direction"])
        entry_price = bid if direction == "LONG" else (ask if ask > 0 else bid)
        ob = dict(setup["order_block"] or {})

        ltf_atr = self._atr_value(ltf_df)
        zone_tolerance = max(
            float(s_cfg.get("ob_retest_tolerance", 0.25) or 0.25),
            ltf_atr * float(s_cfg.get("ob_retest_atr_mult", 0.10) or 0.10),
        )

        ifvg = self._find_ifvg_confirmation(ltf_df, direction, ob, s_cfg)
        if not ifvg:
            return self._no(f"No {ltf_label} IFVG confirmation inside order block")

        if not self._entry_zone_ok(entry_price, direction, ob, ifvg, zone_tolerance):
            return self._no("Price not in executable OB / IFVG area")

        sl = self._compute_stop(entry_price, direction, ltf_df, ob, ifvg, s_cfg)
        if sl is None:
            return self._no("Cannot compute valid stop")
        sl_dist = round(abs(entry_price - sl), 2)
        if sl_dist <= 0:
            return self._no("Invalid stop distance")

        tp = self._compute_take_profit(entry_price, direction, sl_dist, data, setup, s_cfg)
        rr = round(abs(tp - entry_price) / sl_dist, 2) if sl_dist > 0 else 0.0
        min_rr = float(s_cfg.get("min_rr", 1.5) or 1.5)
        if rr < min_rr:
            return self._no(f"RR {rr:.2f} < {min_rr:.2f}")

        confidence = self._confidence(direction, data.get("bias") or {}, setup, ifvg)
        signal = "BUY" if direction == "LONG" else "SELL"

        reason_parts = [
            f"{htf_label} {direction} sweep",
            f"BOS @ {setup['bos_level']:.2f}",
            f"OB [{ob['zone_low']:.2f}-{ob['zone_high']:.2f}]",
            f"{ltf_label} {ifvg['label']}",
            f"SL {sl_dist:.2f}",
            f"RR {rr:.2f}",
        ]

        return {
            "signal": signal,
            "entry": round(entry_price, 2),
            "sl": sl,
            "tp": tp,
            "sl_distance": sl_dist,
            "confidence": confidence,
            "reason": " | ".join(reason_parts),
            "rr": rr,
            "strategy": self.name,
            "_strategy_name": self.name,
            "_signal_family": "SMC",
            "_setup_direction": direction,
            "_bias_direction": str((data.get("bias") or {}).get("direction") or "NEUTRAL").upper(),
            "_sweep_confirmed": True,
            "_candle_confirmation": True,
            "_liquidity_sweep_ob": {
                "htf": htf_label,
                "ltf": ltf_label,
                "sweep_idx": setup["sweep_idx"],
                "bos_idx": setup["bos_idx"],
                "ifvg_idx": ifvg["confirm_idx"],
            },
            "_comment": "FT_LSOB",
            "_high_conf": confidence >= 0.82,
        }

    @staticmethod
    def _pick_htf_frame(data: Dict, s_cfg: Dict) -> Tuple[Optional[pd.DataFrame], str]:
        for tf in s_cfg.get("htf_timeframes", ["M30", "M15"]):
            key = f"{str(tf).lower()}_df"
            frame = data.get(key)
            if frame is not None and len(frame) >= int(s_cfg.get("min_htf_bars", 40) or 40):
                return frame, str(tf).upper()
        return None, ""

    @staticmethod
    def _pick_ltf_frame(data: Dict, s_cfg: Dict) -> Tuple[Optional[pd.DataFrame], str]:
        for tf in s_cfg.get("ltf_timeframes", ["M5", "M1"]):
            key = f"{str(tf).lower()}_df"
            frame = data.get(key)
            if frame is not None and len(frame) >= int(s_cfg.get("min_ltf_bars", 20) or 20):
                return frame, str(tf).upper()
        return None, ""

    def _find_recent_htf_setup(self, df: pd.DataFrame, s_cfg: Dict) -> Optional[Dict]:
        if df is None or len(df) < 12:
            return None

        sweep_lookback = min(len(df) - 1, int(s_cfg.get("sweep_lookback", 40) or 40))
        if sweep_lookback < 6:
            return None
        sweeps = detect_liquidity_sweeps(df, lookback=sweep_lookback)
        if not sweeps:
            return None

        highs = df["high"].astype(float).tolist()
        lows = df["low"].astype(float).tolist()
        closes = df["close"].astype(float).tolist()
        swing_highs, swing_lows = _swing_points(highs, lows, window=2)
        max_bars_after_sweep = int(s_cfg.get("max_bars_after_sweep", 12) or 12)
        structure_lookback = int(s_cfg.get("structure_lookback", 12) or 12)

        for sweep in reversed(sweeps):
            sweep_idx = int(sweep.get("idx", -1))
            if sweep_idx < 3 or sweep_idx >= len(df) - 1:
                continue
            if len(df) - 1 - sweep_idx > max_bars_after_sweep:
                continue

            if str(sweep.get("type") or "").upper() == "BUY_SWEEP":
                direction = "LONG"
                structure_level = self._structure_level_high(df, swing_highs, sweep_idx, structure_lookback)
                bos_idx = self._find_bos_idx(closes, sweep_idx, structure_level, "LONG", max_bars_after_sweep)
            else:
                direction = "SHORT"
                structure_level = self._structure_level_low(df, swing_lows, sweep_idx, structure_lookback)
                bos_idx = self._find_bos_idx(closes, sweep_idx, structure_level, "SHORT", max_bars_after_sweep)

            if structure_level is None or bos_idx is None:
                continue

            order_block = self._find_origin_order_block(df, sweep_idx, bos_idx, direction)
            if not order_block:
                continue

            return {
                "direction": direction,
                "sweep_idx": sweep_idx,
                "bos_idx": bos_idx,
                "bos_level": round(float(structure_level), 2),
                "order_block": order_block,
            }
        return None

    @staticmethod
    def _structure_level_high(df: pd.DataFrame, swing_highs: List[Tuple[int, float]], sweep_idx: int, lookback: int) -> Optional[float]:
        candidates = [price for idx, price in swing_highs if idx < sweep_idx]
        if candidates:
            return float(candidates[-1])
        start = max(0, sweep_idx - lookback)
        if sweep_idx <= start:
            return None
        return float(df.iloc[start:sweep_idx]["high"].astype(float).max())

    @staticmethod
    def _structure_level_low(df: pd.DataFrame, swing_lows: List[Tuple[int, float]], sweep_idx: int, lookback: int) -> Optional[float]:
        candidates = [price for idx, price in swing_lows if idx < sweep_idx]
        if candidates:
            return float(candidates[-1])
        start = max(0, sweep_idx - lookback)
        if sweep_idx <= start:
            return None
        return float(df.iloc[start:sweep_idx]["low"].astype(float).min())

    @staticmethod
    def _find_bos_idx(closes: List[float], sweep_idx: int, structure_level: float, direction: str, max_lookahead: int) -> Optional[int]:
        end = min(len(closes), sweep_idx + max_lookahead + 1)
        for idx in range(sweep_idx + 1, end):
            close = float(closes[idx])
            if direction == "LONG" and close > structure_level:
                return idx
            if direction == "SHORT" and close < structure_level:
                return idx
        return None

    @staticmethod
    def _find_origin_order_block(df: pd.DataFrame, sweep_idx: int, bos_idx: int, direction: str) -> Optional[Dict]:
        start = max(0, sweep_idx - 1)
        for idx in range(bos_idx - 1, start - 1, -1):
            candle = df.iloc[idx]
            open_price = _safe_float(candle.get("open"))
            close = _safe_float(candle.get("close"))
            high = _safe_float(candle.get("high"))
            low = _safe_float(candle.get("low"))
            if direction == "LONG" and close < open_price:
                return {
                    "idx": idx,
                    "zone_low": round(low, 2),
                    "zone_high": round(max(open_price, close), 2),
                }
            if direction == "SHORT" and close > open_price:
                return {
                    "idx": idx,
                    "zone_low": round(min(open_price, close), 2),
                    "zone_high": round(high, 2),
                }
        return None

    def _find_ifvg_confirmation(self, df: pd.DataFrame, direction: str, order_block: Dict, s_cfg: Dict) -> Optional[Dict]:
        if df is None or len(df) < 6:
            return None

        lookback = min(len(df) - 1, int(s_cfg.get("fvg_lookback", 30) or 30))
        if lookback < 4:
            return None
        fvgs = detect_fvg(df, lookback=lookback)
        if not fvgs:
            return None

        zone_low = float(order_block["zone_low"])
        zone_high = float(order_block["zone_high"])
        ltf_atr = self._atr_value(df)
        overlap_tol = max(
            float(s_cfg.get("ifvg_overlap_tolerance", 0.20) or 0.20),
            ltf_atr * float(s_cfg.get("ifvg_overlap_atr_mult", 0.05) or 0.05),
        )
        recent_bars = int(s_cfg.get("ifvg_recent_bars", 4) or 4)
        min_body_ratio = float(s_cfg.get("ifvg_confirm_body_ratio", 0.45) or 0.45)
        last_idx = len(df) - 1

        for fvg in reversed(fvgs):
            fvg_low = float(fvg["zone_low"])
            fvg_high = float(fvg["zone_high"])
            gap_idx = int(fvg["idx"])
            if gap_idx >= last_idx:
                continue
            if not _zones_overlap(fvg_low, fvg_high, zone_low, zone_high, overlap_tol):
                continue

            if direction == "LONG" and str(fvg["type"]) == "BEARISH_FVG":
                confirm_idx = self._find_gap_violation(df, gap_idx, fvg_high, "LONG")
                if confirm_idx is None or confirm_idx < last_idx - recent_bars:
                    continue
                segment = df.iloc[max(0, gap_idx - 2):confirm_idx + 1]
                confirm = df.iloc[confirm_idx]
                if not _touches_zone(segment, zone_low, zone_high, overlap_tol):
                    continue
                if _safe_float(confirm.get("close")) <= _safe_float(confirm.get("open")):
                    continue
                if _candle_body_ratio(confirm) < min_body_ratio:
                    continue
                return {
                    "label": "bearish FVG reclaimed -> bullish IFVG",
                    "zone_low": round(fvg_low, 2),
                    "zone_high": round(fvg_high, 2),
                    "confirm_idx": confirm_idx,
                }

            if direction == "SHORT" and str(fvg["type"]) == "BULLISH_FVG":
                confirm_idx = self._find_gap_violation(df, gap_idx, fvg_low, "SHORT")
                if confirm_idx is None or confirm_idx < last_idx - recent_bars:
                    continue
                segment = df.iloc[max(0, gap_idx - 2):confirm_idx + 1]
                confirm = df.iloc[confirm_idx]
                if not _touches_zone(segment, zone_low, zone_high, overlap_tol):
                    continue
                if _safe_float(confirm.get("close")) >= _safe_float(confirm.get("open")):
                    continue
                if _candle_body_ratio(confirm) < min_body_ratio:
                    continue
                return {
                    "label": "bullish FVG broken -> bearish IFVG",
                    "zone_low": round(fvg_low, 2),
                    "zone_high": round(fvg_high, 2),
                    "confirm_idx": confirm_idx,
                }
        return None

    @staticmethod
    def _find_gap_violation(df: pd.DataFrame, gap_idx: int, threshold: float, direction: str) -> Optional[int]:
        for idx in range(gap_idx + 1, len(df)):
            close = _safe_float(df.iloc[idx].get("close"))
            if direction == "LONG" and close > threshold:
                return idx
            if direction == "SHORT" and close < threshold:
                return idx
        return None

    @staticmethod
    def _entry_zone_ok(price: float, direction: str, order_block: Dict, ifvg: Dict, tolerance: float) -> bool:
        ob_low = float(order_block["zone_low"])
        ob_high = float(order_block["zone_high"])
        ifvg_low = float(ifvg["zone_low"])
        ifvg_high = float(ifvg["zone_high"])
        trade_low = min(ob_low, ifvg_low)
        trade_high = max(ob_high, ifvg_high)
        if direction == "LONG":
            return price <= trade_high + tolerance and price >= trade_low - tolerance
        return price >= trade_low - tolerance and price <= trade_high + tolerance

    def _compute_stop(self, entry: float, direction: str, ltf_df: pd.DataFrame, order_block: Dict, ifvg: Dict, s_cfg: Dict) -> Optional[float]:
        if ltf_df is None or ltf_df.empty:
            return None

        lookback = min(len(ltf_df), int(s_cfg.get("swing_lookback", 8) or 8))
        recent = ltf_df.tail(lookback)
        buffer_points = float(s_cfg.get("sl_buffer_points", 0.20) or 0.20)
        sl_min = float(s_cfg.get("sl_min_points", 0.60) or 0.60)
        sl_max = float(s_cfg.get("sl_max_points", 6.00) or 6.00)

        if direction == "LONG":
            raw = min(
                float(recent["low"].astype(float).min()),
                float(order_block["zone_low"]),
                float(ifvg["zone_low"]),
            ) - buffer_points
            sl = round(raw, 2)
            sl_dist = round(entry - sl, 2)
            if sl_dist < sl_min:
                sl = round(entry - sl_min, 2)
                sl_dist = round(entry - sl, 2)
        else:
            raw = max(
                float(recent["high"].astype(float).max()),
                float(order_block["zone_high"]),
                float(ifvg["zone_high"]),
            ) + buffer_points
            sl = round(raw, 2)
            sl_dist = round(sl - entry, 2)
            if sl_dist < sl_min:
                sl = round(entry + sl_min, 2)
                sl_dist = round(sl - entry, 2)

        if sl_dist <= 0 or sl_dist > sl_max:
            return None
        return sl

    @staticmethod
    def _compute_take_profit(entry: float, direction: str, sl_dist: float, data: Dict, setup: Dict, s_cfg: Dict) -> float:
        rr_target = float(s_cfg.get("target_rr", 2.0) or 2.0)
        fallback_tp = entry + sl_dist * rr_target if direction == "LONG" else entry - sl_dist * rr_target

        levels = []
        key_levels = ((data.get("liquidity") or {}).get("key_levels") or {})
        for key in ("session_high", "prev_day_high", "session_low", "prev_day_low"):
            value = _safe_float(key_levels.get(key))
            if value > 0:
                levels.append(value)
        bos_level = _safe_float(setup.get("bos_level"))
        if bos_level > 0:
            levels.append(bos_level)

        if direction == "LONG":
            candidates = sorted(v for v in levels if v > entry)
            for candidate in candidates:
                rr = (candidate - entry) / max(sl_dist, 1e-6)
                if rr >= float(s_cfg.get("min_rr", 1.5) or 1.5):
                    return round(candidate, 2)
        else:
            candidates = sorted((v for v in levels if v < entry), reverse=True)
            for candidate in candidates:
                rr = (entry - candidate) / max(sl_dist, 1e-6)
                if rr >= float(s_cfg.get("min_rr", 1.5) or 1.5):
                    return round(candidate, 2)
        return round(fallback_tp, 2)

    @staticmethod
    def _confidence(direction: str, bias: Dict, setup: Dict, ifvg: Dict) -> float:
        confidence = 0.72
        bias_dir = str(bias.get("direction") or "").upper()
        if bias_dir == direction:
            confidence += 0.08
        if setup.get("bos_idx", 0) - setup.get("sweep_idx", 0) <= 3:
            confidence += 0.04
        if "IFVG" in str(ifvg.get("label") or "").upper():
            confidence += 0.04
        return round(min(0.92, confidence), 2)

    @staticmethod
    def _atr_value(df: pd.DataFrame, period: int = 14) -> float:
        if df is None or len(df) < period + 2:
            return 0.0
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        closes = df["close"].values.astype(float)
        try:
            return float(atr(highs, lows, closes, period)[-1])
        except Exception:
            return 0.0

    @staticmethod
    def _no(reason: str) -> Dict:
        return {"signal": "NO_TRADE", "reason": reason, "confidence": 0.0}
