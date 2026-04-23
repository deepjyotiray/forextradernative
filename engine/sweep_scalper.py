"""
Liquidity Sweep Scalper — Tier 1 upgraded.

Changes from baseline:
  1. Spread model (mean/std/percentile) replaces simple threshold
  2. Dynamic SL = clamp(ATR*0.6, 0.80, 1.50)
  3. First-sweep-only — tracks swept levels, blocks re-entry within ±$0.50 for 20min
  4. Early failure exit params passed to trade manager
  5. Session windows tightened: London 07-09, Overlap 12-13, NY 13-15
  6. Dynamic TP = max(ATR*0.8, spread*2.2) capped at ATR*1.5
"""
import time
import numpy as np
import pandas as pd
from typing import Dict, Optional
from datetime import datetime, timezone
from .strategies.base_strategy import BaseStrategy
from .indicators import ema, atr
from .tick_processor import TickProcessor
import config as cfg

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
_TIMEOUT = 60               # speed exit: 60s
_EARLY_FAIL_PTS = 0.20      # $0.20 adverse in first 15 ticks = exit
_LEVEL_COOLDOWN = 1200       # 20 min before re-entering same level


class SweepScalper(BaseStrategy):
    name = "SWEEP_SCALPER"

    def __init__(self):
        self.tick_proc = TickProcessor(buffer_size=500)
        self._session_trades = 0
        self._session_date = ""
        self._last_session = ""
        # Sweep level tracking: {level_rounded: timestamp}
        self._traded_levels: Dict[float, float] = {}

    def generate_signal(self, data: Dict) -> Dict:
        tick = data.get("tick")
        if not tick:
            return _no("No tick")

        self.tick_proc.feed(tick)
        ts = self.tick_proc.snapshot()

        m1 = data.get("m1_df")
        m5 = data.get("m5_df")
        if m1 is None or len(m1) < 30:
            return _no("Insufficient M1 data")

        price = tick["bid"]
        spread = tick.get("spread", 0)

        # === 1. Session windows ===
        now = datetime.now(timezone.utc)
        h = now.hour
        if cfg.TIER1_ENABLED:
            # Tier 1: tightened windows
            in_window = (7 <= h < 9) or (12 <= h < 13) or (13 <= h < 15)
        else:
            # Baseline: wider windows
            in_window = (7 <= h < 12) or (13 <= h < 17)
        if not in_window:
            return _no(f"Outside trade window (UTC {h}:xx)")

        session = "LONDON" if h < 12 else "OVERLAP" if h < 13 else "NY"
        today = now.strftime("%Y-%m-%d")
        if session != self._last_session or today != self._session_date:
            self._session_trades = 0
            self._last_session = session
            self._session_date = today

        if self._session_trades >= _MAX_TRADES_SESSION:
            return _no(f"Max {_MAX_TRADES_SESSION} trades this session")

        # === 2. Spread check ===
        if cfg.TIER1_ENABLED:
            spread_ok, spread_reason = self.tick_proc.check_spread_ok(_SPREAD_MAX, max_pctl=0.5)
            if not spread_ok:
                return _no(f"Spread: {spread_reason}")
        else:
            if spread > 0.40:
                return _no(f"Spread {spread:.2f} > 0.40")

        # === 3. Volatility filter ===
        c = m1["close"].values.astype(float)
        h_arr = m1["high"].values.astype(float)
        l_arr = m1["low"].values.astype(float)
        o_arr = m1["open"].values.astype(float)
        atr14 = atr(h_arr, l_arr, c, 14)
        atr_val = atr14[-1]

        if atr_val < _ATR_MIN:
            return _no(f"ATR {atr_val:.3f} < {_ATR_MIN}")
        if atr_val > _ATR_MAX:
            return _no(f"ATR {atr_val:.3f} > {_ATR_MAX}")

        # === 4. EMA20 bias ===
        ema20 = ema(c, 20)
        ema20_val = ema20[-1]
        ema20_slope = ema20[-1] - ema20[-3] if len(ema20) >= 3 else 0

        if abs(ema20_slope) < 0.05:
            return _no(f"EMA20 flat ({ema20_slope:.3f})")

        # === 5. M5 bias ===
        m5_bias = None
        if m5 is not None and len(m5) >= 20:
            c5 = m5["close"].values.astype(float)
            ema20_m5 = ema(c5, 20)
            if c5[-1] > ema20_m5[-1]:
                m5_bias = "LONG"
            elif c5[-1] < ema20_m5[-1]:
                m5_bias = "SHORT"

        # === 6. Detect sweep ===
        sweep = self._detect_sweep(h_arr, l_arr, c, o_arr, price)
        if not sweep:
            return _no("No sweep detected")

        direction = sweep["direction"]
        sweep_level = sweep["level"]

        # === 7. First-sweep-only: block re-entry at same level ===
        if cfg.TIER1_ENABLED:
            self._cleanup_levels()
            level_key = round(sweep_level * 2) / 2  # round to nearest $0.50
            if level_key in self._traded_levels:
                return _no(f"Level {sweep_level:.2f} already traded")

        # Bias agreement
        if m5_bias and m5_bias != direction:
            return _no(f"M5 bias {m5_bias} conflicts with sweep {direction}")
        if direction == "LONG" and ema20_slope < 0:
            return _no("EMA20 bearish vs bullish sweep")
        if direction == "SHORT" and ema20_slope > 0:
            return _no("EMA20 bullish vs bearish sweep")

        # === 8. Displacement candle ===
        last = len(c) - 1
        body = abs(c[last] - o_arr[last])
        rng = h_arr[last] - l_arr[last]
        body_ratio = body / rng if rng > 0 else 0
        bodies = np.abs(c[last-10:last] - o_arr[last-10:last])
        avg_body = bodies.mean() if len(bodies) > 0 else 0

        if body_ratio < _BODY_MIN:
            return _no(f"Body ratio {body_ratio:.2f} < {_BODY_MIN}")
        if body < avg_body:
            return _no(f"Body {body:.3f} < avg {avg_body:.3f}")
        if direction == "LONG" and c[last] <= o_arr[last]:
            return _no("Displacement not bullish")
        if direction == "SHORT" and c[last] >= o_arr[last]:
            return _no("Displacement not bearish")
        if direction == "LONG" and price < ema20_val:
            return _no("Price below EMA20")
        if direction == "SHORT" and price > ema20_val:
            return _no("Price above EMA20")

        # === 9. Execution quality gate ===
        if not ts.get("ready"):
            return _no("Tick buffer not ready")
        eq_ok, eq_reason = self.tick_proc.check_execution_quality()
        if not eq_ok:
            return _no(f"Exec quality: {eq_reason}")

        # Directional tick dominance
        dir_pct = ts.get("dir_pct", 0.5)
        if dir_pct < _TICK_DIR_PCT:
            return _no(f"Tick direction {dir_pct:.0%} < {_TICK_DIR_PCT:.0%}")

        # === 10. SL/TP ===
        signal = "BUY" if direction == "LONG" else "SELL"

        if cfg.TIER1_ENABLED:
            # Tier 1: Dynamic SL = clamp(ATR * 0.6, 0.80, 1.50)
            sl_dist = max(0.80, min(1.50, atr_val * 0.6))
        else:
            # Baseline: fixed SL range $0.60-$1.00
            sl_dist = max(0.60, min(1.00, atr_val * 0.5))

        if signal == "BUY":
            sl = round(max(sweep_level - 0.10, price - sl_dist), 2)
            sl_dist = price - sl
        else:
            sl = round(min(sweep_level + 0.10, price + sl_dist), 2)
            sl_dist = sl - price

        if cfg.TIER1_ENABLED:
            sl_dist = round(max(0.80, min(1.50, sl_dist)), 2)
        else:
            sl_dist = round(max(0.60, min(1.00, sl_dist)), 2)
        if signal == "BUY":
            sl = round(price - sl_dist, 2)
        else:
            sl = round(price + sl_dist, 2)

        spread_mean = ts.get("spread_mean", spread)
        if cfg.TIER1_ENABLED:
            # Tier 1: TP = max(ATR * 0.8, spread * 2.2) capped at ATR * 1.5
            tp_dist = max(atr_val * 0.8, spread_mean * 2.2)
            tp_dist = min(tp_dist, atr_val * 1.5)
            tp_dist = round(max(tp_dist, spread_mean * 2.2), 2)
        else:
            # Baseline: simpler TP
            tp_dist = round(atr_val * 1.0, 2)

        if signal == "BUY":
            tp = round(price + tp_dist, 2)
        else:
            tp = round(price - tp_dist, 2)

        rr = tp_dist / sl_dist if sl_dist > 0 else 0

        # === 11. Final spread re-check before returning signal ===
        if cfg.TIER1_ENABLED and self.tick_proc.spread_changed(spread, max_delta=0.03):
            return _no("Spread widened since signal")

        # Record level as traded
        if cfg.TIER1_ENABLED:
            level_key = round(sweep_level * 2) / 2
            self._traded_levels[level_key] = time.time()
        self._session_trades += 1

        # === Quality score ===
        q_score = 0.0
        q_score += 0.25  # sweep detected
        q_score += min(0.20, body_ratio * 0.25)  # displacement
        q_score += min(0.15, (dir_pct - 0.5) * 0.5) if dir_pct > 0.5 else 0  # tick direction
        q_score += min(0.10, abs(ema20_slope) * 0.5)  # M1 EMA momentum
        if m5_bias == direction:
            q_score += 0.15  # M5 alignment
        if ts.get("velocity", 0) > 10:
            q_score += 0.10  # tick velocity
        q_score = round(min(1.0, q_score), 3)

        if q_score < _QUALITY_THRESHOLD:
            return _no(f"Quality {q_score:.0%} < {_QUALITY_THRESHOLD:.0%}")

        # === LTF pullback blocker ===
        pullback_against = 0
        if m5_bias and m5_bias != direction:
            pullback_against += 1
        if m1 is not None and len(m1) >= 5:
            c1 = m1["close"].values.astype(float)
            o1 = m1["open"].values.astype(float)
            bearish_seq = sum(1 for i in range(-3, 0) if c1[i] < o1[i])
            bullish_seq = sum(1 for i in range(-3, 0) if c1[i] > o1[i])
            if direction == "LONG" and bearish_seq >= 3:
                pullback_against += 1
            elif direction == "SHORT" and bullish_seq >= 3:
                pullback_against += 1
        if dir_pct < 0.40 and direction == "LONG":
            pullback_against += 1
        elif dir_pct > 0.60 and direction == "SHORT":
            pullback_against += 1
        if pullback_against >= 2:
            return _no(f"Pullback active against {direction} (signals:{pullback_against})")

        reasons = [
            f"Sweep {direction} @ {sweep_level:.2f}",
            f"Body {body_ratio:.0%}",
            f"EMA20 {ema20_slope:+.3f}",
            f"Vel {ts['velocity']:.1f}",
            f"ATR {atr_val:.3f}",
            f"Spread {spread_mean:.3f}",
            f"SL:{sl_dist:.2f} TP:{tp_dist:.2f}",
            f"[Q:{q_score:.0%}]",
        ]

        return {
            "signal": signal,
            "entry": price,
            "sl": sl,
            "tp": tp,
            "sl_distance": sl_dist,
            "confidence": 0.80,
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
            },
            "_scalp": True,
            "_be_trigger": _BE_TRIGGER,
            "_timeout": _TIMEOUT if cfg.TIER1_ENABLED else 150,
            "_early_fail": _EARLY_FAIL_PTS if cfg.TIER1_ENABLED else 0,
        }

    def _detect_sweep(self, h, l, c, o, price) -> Optional[Dict]:
        n = len(h)
        if n < _SWEEP_LOOKBACK + 2:
            return None
        start = n - _SWEEP_LOOKBACK - 1
        end = n - 2
        curr = n - 1

        for i in range(start, end - 2):
            for j in range(i + 2, end):
                if abs(l[i] - l[j]) < _SWEEP_TOL:
                    eq_low = min(l[i], l[j])
                    for k in range(j + 1, curr + 1):
                        if l[k] < eq_low - _SWEEP_TOL:
                            if c[curr] > eq_low and c[curr] > o[curr]:
                                return {"direction": "LONG", "level": round(float(l[k]), 2),
                                        "eq_level": round(float(eq_low), 2)}

        for i in range(start, end - 2):
            for j in range(i + 2, end):
                if abs(h[i] - h[j]) < _SWEEP_TOL:
                    eq_high = max(h[i], h[j])
                    for k in range(j + 1, curr + 1):
                        if h[k] > eq_high + _SWEEP_TOL:
                            if c[curr] < eq_high and c[curr] < o[curr]:
                                return {"direction": "SHORT", "level": round(float(h[k]), 2),
                                        "eq_level": round(float(eq_high), 2)}
        return None

    def _cleanup_levels(self):
        """Remove levels older than cooldown."""
        now = time.time()
        expired = [k for k, t in self._traded_levels.items() if now - t > _LEVEL_COOLDOWN]
        for k in expired:
            del self._traded_levels[k]


def _no(reason: str) -> Dict:
    return {"signal": "NO_TRADE", "reason": reason}
