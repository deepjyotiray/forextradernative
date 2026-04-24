"""
Liquidity Sweep Scalper with execution-quality filters.
"""
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .anti_starvation import anti_starvation
from .decision_logger import log_scalper_decision
from .indicators import atr, ema
from .signal_quality import quality_score
from .strategies.base_strategy import BaseStrategy
from .tick_processor import TickProcessor

_QUALITY_THRESHOLD = 0.65
_SPREAD_MAX = 0.25
_ATR_MIN = 0.30
_ATR_MAX = 5.00
_BODY_MIN = 0.40
_TICK_DIR_PCT = 0.65
_SWEEP_LOOKBACK = 15
_SWEEP_TOL = 0.30
_MAX_TRADES_SESSION = 5
_BE_TRIGGER = 0.30
_TIMEOUT = 60
_EARLY_FAIL_PTS = 0.20
_LEVEL_COOLDOWN = 1200


class SweepScalper(BaseStrategy):
    name = "SWEEP_SCALPER"

    def __init__(self):
        self.tick_proc = TickProcessor(buffer_size=500)
        self._session_trades = 0
        self._session_date = ""
        self._last_session = ""
        self._traded_levels: Dict[float, float] = {}

    def generate_signal(self, data: Dict) -> Dict:
        tick = data.get("tick")
        if not tick:
            return _no("No tick")

        self.tick_proc.feed(tick)
        tick_snap = self.tick_proc.snapshot()

        m1 = data.get("m1_df")
        m5 = data.get("m5_df")

        price = tick["bid"]
        spread = tick.get("spread", 0)
        spread_mean = tick_snap.get("spread_mean", spread)
        spread_std = tick_snap.get("spread_std", 0)

        def skip(reason: str,
                 setup_direction: Optional[str] = None,
                 quality_score_value: float = 0.0,
                 compression_ok: bool = False,
                 sweep_level: Optional[float] = None) -> Dict:
            log_scalper_decision(
                setup_direction=setup_direction,
                quality_score=quality_score_value,
                spread_mean=spread_mean,
                spread_std=spread_std,
                compression_ok=compression_ok,
                decision="TRADE_SKIPPED",
                reason=reason,
                price=price,
                sweep_level=sweep_level,
            )
            return {"signal": "NO_TRADE", "reason": reason, "score": quality_score_value}

        if m1 is None or len(m1) < 30:
            return skip("Insufficient M1 data")

        now = datetime.now(timezone.utc)
        hour = now.hour
        in_window = (7 <= hour < 9) or (12 <= hour < 13) or (13 <= hour < 15)
        if not in_window and not cfg.SESSION_OVERRIDE_ENABLED:
            return skip(f"Outside trade window (UTC {hour}:xx)")

        session = "LONDON" if hour < 12 else "OVERLAP" if hour < 13 else "NY"
        today = now.strftime("%Y-%m-%d")
        if session != self._last_session or today != self._session_date:
            self._session_trades = 0
            self._last_session = session
            self._session_date = today

        if self._session_trades >= _MAX_TRADES_SESSION:
            return skip(f"Max {_MAX_TRADES_SESSION} trades this session")

        spread_ok, spread_reason = self._check_spread_quality(spread, tick_snap)
        if not spread_ok:
            return skip(f"Spread: {spread_reason}")

        closes = m1["close"].values.astype(float)
        highs = m1["high"].values.astype(float)
        lows = m1["low"].values.astype(float)
        opens = m1["open"].values.astype(float)
        atr_val = atr(highs, lows, closes, 14)[-1]
        if atr_val < _ATR_MIN:
            return skip(f"ATR {atr_val:.3f} < {_ATR_MIN}")
        if atr_val > _ATR_MAX:
            return skip(f"ATR {atr_val:.3f} > {_ATR_MAX}")

        compression_ok, comp_reason = self._check_compression_gate(m1, atr_val)
        if not compression_ok:
            return skip(f"Compression: {comp_reason}")

        ema20 = ema(closes, 20)
        ema20_val = ema20[-1]
        ema20_slope = ema20[-1] - ema20[-3] if len(ema20) >= 3 else 0
        if abs(ema20_slope) < 0.05:
            return skip(f"EMA20 flat ({ema20_slope:.3f})")

        sweep = self._detect_sweep(highs, lows, closes, opens)
        if not sweep:
            return skip("No sweep detected", compression_ok=True)

        direction = sweep["direction"]
        sweep_level = sweep["level"]
        self._cleanup_levels()
        level_key = round(sweep_level * 2) / 2
        if level_key in self._traded_levels:
            return skip(f"Level {sweep_level:.2f} already traded", setup_direction=direction,
                        compression_ok=True, sweep_level=sweep_level)

        m1_aligned = self._check_timeframe_alignment(m1, direction)
        m5_aligned = self._check_timeframe_alignment(m5, direction) if m5 is not None else False

        last = len(closes) - 1
        body = abs(closes[last] - opens[last])
        candle_range = highs[last] - lows[last]
        body_ratio = body / candle_range if candle_range > 0 else 0
        avg_body = np.abs(closes[last - 10:last] - opens[last - 10:last]).mean()
        if body_ratio < _BODY_MIN:
            return skip(f"Body ratio {body_ratio:.2f} < {_BODY_MIN}", setup_direction=direction,
                        compression_ok=True, sweep_level=sweep_level)
        if body < avg_body:
            return skip(f"Body {body:.3f} < avg {avg_body:.3f}", setup_direction=direction,
                        compression_ok=True, sweep_level=sweep_level)
        if direction == "LONG" and (closes[last] <= opens[last] or price < ema20_val):
            return skip("LONG displacement invalid", setup_direction=direction,
                        compression_ok=True, sweep_level=sweep_level)
        if direction == "SHORT" and (closes[last] >= opens[last] or price > ema20_val):
            return skip("SHORT displacement invalid", setup_direction=direction,
                        compression_ok=True, sweep_level=sweep_level)

        if not tick_snap.get("ready"):
            return skip("Tick buffer not ready", setup_direction=direction,
                        compression_ok=True, sweep_level=sweep_level)
        eq_ok, eq_reason = self.tick_proc.check_execution_quality()
        if not eq_ok:
            return skip(f"Exec quality: {eq_reason}", setup_direction=direction,
                        compression_ok=True, sweep_level=sweep_level)

        relaxed_params = anti_starvation.get_relaxed_params()
        tick_threshold = 0.65
        if relaxed_params["active"] and relaxed_params["type"] == "tick_ratio":
            tick_threshold = relaxed_params["tick_ratio_threshold"]

        dir_pct = tick_snap.get("dir_pct", 0.5)
        directional_ok = (
            (direction == "LONG" and dir_pct >= tick_threshold)
            or (direction == "SHORT" and dir_pct <= (1.0 - tick_threshold))
        )
        if not directional_ok:
            target = tick_threshold if direction == "LONG" else (1.0 - tick_threshold)
            return skip(
                f"Tick direction {dir_pct:.0%} not aligned with {direction} threshold {target:.0%}",
                setup_direction=direction,
                compression_ok=True,
                sweep_level=sweep_level,
            )

        q_score, q_reasons = self._quality_score(
            direction=direction,
            body_ratio=body_ratio,
            dir_pct=dir_pct,
            m1_aligned=m1_aligned,
            m5_aligned=m5_aligned,
            compression_ok=compression_ok,
            tick_snap=tick_snap,
        )
        if q_score < _QUALITY_THRESHOLD:
            return skip(
                f"Quality {q_score:.0%} < {_QUALITY_THRESHOLD:.0%}",
                setup_direction=direction,
                quality_score_value=q_score,
                compression_ok=compression_ok,
                sweep_level=sweep_level,
            )

        signal = "BUY" if direction == "LONG" else "SELL"
        sl, tp, sl_dist = self._compute_cost_aware_sl_tp(price, signal, atr_val, spread, sweep_level)
        if sl is None or tp is None:
            return skip("Cannot compute valid SL/TP", setup_direction=direction,
                        quality_score_value=q_score, compression_ok=compression_ok,
                        sweep_level=sweep_level)

        tp_dist = abs(tp - price)
        rr = tp_dist / sl_dist if sl_dist > 0 else 0
        if self.tick_proc.spread_changed(spread, max_delta=0.03):
            return skip("Spread widened since signal", setup_direction=direction,
                        quality_score_value=q_score, compression_ok=compression_ok,
                        sweep_level=sweep_level)

        self._traded_levels[level_key] = time.time()
        self._session_trades += 1

        reasons = [
            f"Sweep {direction} @ {sweep_level:.2f}",
            f"Body {body_ratio:.0%}",
            f"EMA20 {ema20_slope:+.3f}",
            f"Vel {tick_snap.get('velocity', 0):.1f}",
            f"ATR {atr_val:.3f}",
            f"SL:{sl_dist:.2f} TP:{tp_dist:.2f}",
            f"[Q:{q_score:.0%}]",
        ] + q_reasons

        signal_result = {
            "signal": signal,
            "entry": price,
            "sl": sl,
            "tp": tp,
            "sl_distance": sl_dist,
            "confidence": q_score,
            "reason": " | ".join(reasons),
            "reasons": reasons,
            "rr": round(rr, 2),
            "sweep_level": sweep_level,
            "indicators": {
                "ema20": round(ema20_val, 2),
                "ema20_slope": round(ema20_slope, 4),
                "atr14": round(atr_val, 4),
                "spread": spread,
                "spread_mean": spread_mean,
                "tick_velocity": tick_snap.get("velocity", 0),
            },
            "_scalp": True,
            "_be_trigger": _BE_TRIGGER,
            "_timeout": _TIMEOUT,
            "_early_fail": _EARLY_FAIL_PTS,
            "_setup_direction": direction,
            "_quality_score": q_score,
            "_threshold": _QUALITY_THRESHOLD,
            "_entry_tick_velocity": tick_snap.get("velocity", 0),
            "_entry_spread": spread,
        }

        log_scalper_decision(
            setup_direction=direction,
            quality_score=q_score,
            spread_mean=spread_mean,
            spread_std=spread_std,
            compression_ok=compression_ok,
            decision="TRADE_TAKEN",
            reason=" | ".join(reasons),
            price=price,
            sweep_level=sweep_level,
            signal_data=signal_result,
        )
        return signal_result

    def _quality_score(
        self,
        direction: str,
        body_ratio: float,
        dir_pct: float,
        m1_aligned: bool,
        m5_aligned: bool,
        compression_ok: bool,
        tick_snap: Dict,
    ) -> Tuple[float, list]:
        relaxed_params = anti_starvation.get_relaxed_params()
        tick_threshold = 0.65
        if relaxed_params["active"] and relaxed_params["type"] == "tick_ratio":
            tick_threshold = relaxed_params["tick_ratio_threshold"]
        score, reasons = quality_score(
            direction=direction,
            sweep_present=True,
            body_ratio=body_ratio,
            tick_ratio=dir_pct,
            tick_velocity_increasing=tick_snap.get("vel_increasing", False),
            m1_aligned=m1_aligned,
            m5_aligned=m5_aligned,
            compression_ok=compression_ok,
            tick_ratio_threshold=tick_threshold,
        )
        if relaxed_params["active"] and relaxed_params["type"] == "tick_ratio":
            reasons.append(f"Anti-starvation tick ratio {tick_threshold:.0%}")
        return score, reasons

    def _check_timeframe_alignment(self, df: pd.DataFrame, direction: str) -> bool:
        if df is None or len(df) < 25:
            return False

        closes = df["close"].values.astype(float)
        ema20 = ema(closes, 20)
        price = closes[-1]
        ema_val = ema20[-1]
        slope = ema20[-1] - ema20[-3] if len(ema20) >= 3 else 0
        if direction == "LONG":
            return price >= ema_val and slope > 0.1
        return price <= ema_val and slope < -0.1

    def _check_spread_quality(self, spread: float, tick_snap: Dict) -> Tuple[bool, str]:
        relaxed_params = anti_starvation.get_relaxed_params()
        spread_limit = _SPREAD_MAX
        if relaxed_params["active"] and relaxed_params["type"] == "spread_tolerance":
            spread_limit += relaxed_params["spread_tolerance_bonus"]

        if not tick_snap.get("ready"):
            if spread > spread_limit:
                return False, f"Spread {spread:.3f} > {spread_limit:.3f}"
            return True, "OK"

        spread_mean = tick_snap.get("spread_mean", spread)
        spread_std = tick_snap.get("spread_std", 0)
        spread_pctl = tick_snap.get("spread_pctl", 0.5)
        if spread_mean > spread_limit:
            return False, f"Mean spread {spread_mean:.3f} > {spread_limit:.3f}"
        if spread_std > 0.04:
            return False, f"Spread volatility {spread_std:.3f} > 0.04"
        if spread_pctl > 0.50:
            return False, f"Spread percentile {spread_pctl:.0%} > 50%"
        if spread > spread_mean + 0.03:
            return False, f"Current spread {spread:.3f} > baseline+0.03"
        return True, "Spread OK"

    def _check_compression_gate(self, m1: pd.DataFrame, atr_val: float) -> Tuple[bool, str]:
        if m1 is None or len(m1) < 17:
            return False, "Insufficient M1 data for compression check"

        last_10 = m1.iloc[-10:]
        range_10 = float(last_10["high"].max() - last_10["low"].min())
        atr_threshold = 1.8 * atr_val
        if range_10 >= atr_threshold:
            return False, f"Range10 {range_10:.1f} >= {atr_threshold:.1f} (too wide)"

        highs = m1["high"].values.astype(float)
        lows = m1["low"].values.astype(float)
        closes = m1["close"].values.astype(float)
        atr_series = atr(highs, lows, closes, 14)
        atr_prev = atr_series[-4] if len(atr_series) >= 4 else atr_series[-1]
        if atr_series[-1] <= atr_prev:
            return False, f"ATR not rising ({atr_series[-1]:.3f} <= {atr_prev:.3f})"
        return True, f"Compression OK (R10:{range_10:.1f} < {atr_threshold:.1f}, ATR+)"

    def _compute_cost_aware_sl_tp(
        self,
        price: float,
        signal: str,
        atr_val: float,
        spread: float,
        sweep_level: float,
    ) -> tuple:
        sl_dist = max(0.80, min(1.50, atr_val * 0.6))
        tp_dist = max(atr_val * 0.8, spread * 2.2)
        tp_dist = min(tp_dist, atr_val * 1.5)
        tp_dist = max(tp_dist, spread * 2.2)

        if signal == "BUY":
            sl_candidate = max(sweep_level - 0.10, price - sl_dist)
            sl = round(sl_candidate, 2)
            tp = round(price + tp_dist, 2)
            sl_dist = price - sl
        else:
            sl_candidate = min(sweep_level + 0.10, price + sl_dist)
            sl = round(sl_candidate, 2)
            tp = round(price - tp_dist, 2)
            sl_dist = sl - price

        sl_dist = round(abs(sl_dist), 2)
        if sl_dist < 0.5:
            return None, None, 0
        return sl, tp, sl_dist

    def _detect_sweep(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        opens: np.ndarray,
    ) -> Optional[Dict]:
        n = len(highs)
        if n < _SWEEP_LOOKBACK + 2:
            return None

        start = n - _SWEEP_LOOKBACK - 1
        end = n - 2
        curr = n - 1

        for i in range(start, end - 2):
            for j in range(i + 2, end):
                if abs(lows[i] - lows[j]) < _SWEEP_TOL:
                    eq_low = min(lows[i], lows[j])
                    for k in range(j + 1, curr + 1):
                        if lows[k] < eq_low - _SWEEP_TOL and closes[curr] > eq_low and closes[curr] > opens[curr]:
                            return {
                                "direction": "LONG",
                                "level": round(float(lows[k]), 2),
                                "eq_level": round(float(eq_low), 2),
                            }

        for i in range(start, end - 2):
            for j in range(i + 2, end):
                if abs(highs[i] - highs[j]) < _SWEEP_TOL:
                    eq_high = max(highs[i], highs[j])
                    for k in range(j + 1, curr + 1):
                        if highs[k] > eq_high + _SWEEP_TOL and closes[curr] < eq_high and closes[curr] < opens[curr]:
                            return {
                                "direction": "SHORT",
                                "level": round(float(highs[k]), 2),
                                "eq_level": round(float(eq_high), 2),
                            }
        return None

    def _cleanup_levels(self):
        now = time.time()
        expired = [level for level, ts in self._traded_levels.items() if now - ts > _LEVEL_COOLDOWN]
        for level in expired:
            del self._traded_levels[level]


def _no(reason: str) -> Dict:
    return {"signal": "NO_TRADE", "reason": reason}
