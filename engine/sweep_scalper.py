"""
Liquidity Sweep Scalper — fast $3-$6 scalps on XAUUSD.

Only strategy: detect liquidity sweep on M1, confirm with displacement
candle + tick momentum, enter immediately, tight SL/TP.

Trade windows: London open first 2h (07-09 UTC), NY open first 2h (13-15 UTC).
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone
from .strategies.base_strategy import BaseStrategy
from .indicators import ema, atr
from .tick_processor import TickProcessor


# Scalper-specific constants
_SPREAD_MAX = 0.35          # $0.35 max spread
_SL_MIN = 0.60              # $0.60 min SL (must be > spread + buffer)
_SL_MAX = 1.00              # $1.00 max SL
_TP_MIN = 0.80              # $0.80 min TP
_TP_MAX = 1.50              # $1.50 max TP
_ATR_MIN = 0.30             # min M1 ATR
_ATR_MAX = 5.00             # max M1 ATR (news)
_BODY_MIN = 0.40            # min body ratio for displacement
_TICK_DIR_PCT = 0.65        # 65% directional ticks
_SWEEP_LOOKBACK = 15        # M1 candles to scan for equal levels
_SWEEP_TOL = 0.30           # $0.30 tolerance for "equal" highs/lows
_MAX_TRADES_SESSION = 5
_BE_TRIGGER = 0.30          # move SL to BE after +$0.30
_TIMEOUT = 150              # force exit after 150s


class SweepScalper(BaseStrategy):
    name = "SWEEP_SCALPER"

    def __init__(self):
        self.tick_proc = TickProcessor(buffer_size=500)
        self._session_trades = 0
        self._session_date = ""
        self._last_session = ""

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

        # === 1. Session window — London 07-12 and NY 13-17 UTC ===
        now = datetime.now(timezone.utc)
        h = now.hour
        in_window = (7 <= h < 12) or (13 <= h < 17)
        if not in_window:
            return _no(f"Outside trade window (UTC {h}:xx)")

        # Reset session trade count on new session
        session = "LONDON" if h < 13 else "NY"
        today = now.strftime("%Y-%m-%d")
        if session != self._last_session or today != self._session_date:
            self._session_trades = 0
            self._last_session = session
            self._session_date = today

        if self._session_trades >= _MAX_TRADES_SESSION:
            return _no(f"Max {_MAX_TRADES_SESSION} trades this session")

        # === 2. Spread filter ===
        if spread > _SPREAD_MAX:
            return _no(f"Spread {spread:.2f} > {_SPREAD_MAX}")

        # === 3. Volatility filter (ATR on M1) ===
        c = m1["close"].values.astype(float)
        h_arr = m1["high"].values.astype(float)
        l_arr = m1["low"].values.astype(float)
        o_arr = m1["open"].values.astype(float)
        atr14 = atr(h_arr, l_arr, c, 14)
        atr_val = atr14[-1]

        if atr_val < _ATR_MIN:
            return _no(f"ATR {atr_val:.3f} < {_ATR_MIN} (dead market)")
        if atr_val > _ATR_MAX:
            return _no(f"ATR {atr_val:.3f} > {_ATR_MAX} (news spike)")

        # === 4. EMA20 on M1 (bias filter) ===
        ema20 = ema(c, 20)
        ema20_val = ema20[-1]
        ema20_slope = ema20[-1] - ema20[-3] if len(ema20) >= 3 else 0

        # EMA must not be flat
        if abs(ema20_slope) < 0.05:
            return _no(f"EMA20 flat (slope {ema20_slope:.3f})")

        # === 5. M5 bias filter ===
        m5_bias = None
        if m5 is not None and len(m5) >= 20:
            c5 = m5["close"].values.astype(float)
            ema20_m5 = ema(c5, 20)
            if c5[-1] > ema20_m5[-1]:
                m5_bias = "LONG"
            elif c5[-1] < ema20_m5[-1]:
                m5_bias = "SHORT"

        # === 6. Detect liquidity sweep on M1 ===
        sweep = self._detect_sweep(h_arr, l_arr, c, o_arr, price)
        if not sweep:
            return _no("No sweep detected")

        direction = sweep["direction"]  # "LONG" or "SHORT"
        sweep_level = sweep["level"]

        # M5 bias must agree (or be None)
        if m5_bias and m5_bias != direction:
            return _no(f"M5 bias {m5_bias} conflicts with sweep {direction}")

        # EMA20 must agree
        if direction == "LONG" and ema20_slope < 0:
            return _no("EMA20 slope bearish, sweep is bullish")
        if direction == "SHORT" and ema20_slope > 0:
            return _no("EMA20 slope bullish, sweep is bearish")

        # === 7. Displacement candle confirmation ===
        last = len(c) - 1
        body = abs(c[last] - o_arr[last])
        rng = h_arr[last] - l_arr[last]
        body_ratio = body / rng if rng > 0 else 0

        # Average body of previous 10 candles
        bodies = np.abs(c[last-10:last] - o_arr[last-10:last])
        avg_body = bodies.mean() if len(bodies) > 0 else 0

        if body_ratio < _BODY_MIN:
            return _no(f"Body ratio {body_ratio:.2f} < {_BODY_MIN}")
        if body < avg_body * 1.0:
            return _no(f"Body {body:.3f} < avg {avg_body:.3f}")

        # Displacement direction must match
        if direction == "LONG" and c[last] <= o_arr[last]:
            return _no("Displacement candle not bullish")
        if direction == "SHORT" and c[last] >= o_arr[last]:
            return _no("Displacement candle not bearish")

        # Price must be back above/below EMA20
        if direction == "LONG" and price < ema20_val:
            return _no("Price still below EMA20")
        if direction == "SHORT" and price > ema20_val:
            return _no("Price still above EMA20")

        # === 8. Tick momentum confirmation ===
        if not ts.get("ready"):
            return _no("Tick buffer not ready")

        if ts.get("velocity", 0) < 2:
            return _no(f"Tick velocity low ({ts['velocity']})")

        if not ts.get("spread_stable", True):
            return _no("Spread unstable")

        # Directional tick dominance
        dir_ticks = ts.get("dir_ticks", 0)
        tick_count = min(20, ts.get("tick_count", 1))
        if tick_count > 5:
            dir_up = (tick_count + dir_ticks) / 2
            dir_pct = dir_up / tick_count if direction == "LONG" else (tick_count - dir_up) / tick_count
            if dir_pct < _TICK_DIR_PCT:
                return _no(f"Tick direction {dir_pct:.0%} < {_TICK_DIR_PCT:.0%}")

        # === 9. Compute SL/TP ===
        signal = "BUY" if direction == "LONG" else "SELL"

        if signal == "BUY":
            sl = round(sweep_level - 0.10, 2)  # just below sweep wick
            sl_dist = price - sl
            # Clamp SL distance
            if sl_dist < _SL_MIN:
                sl = round(price - _SL_MIN, 2)
                sl_dist = _SL_MIN
            elif sl_dist > _SL_MAX:
                sl = round(price - _SL_MAX, 2)
                sl_dist = _SL_MAX
            tp = round(price + max(_TP_MIN, min(_TP_MAX, sl_dist * 1.5)), 2)
        else:
            sl = round(sweep_level + 0.10, 2)
            sl_dist = sl - price
            if sl_dist < _SL_MIN:
                sl = round(price + _SL_MIN, 2)
                sl_dist = _SL_MIN
            elif sl_dist > _SL_MAX:
                sl = round(price + _SL_MAX, 2)
                sl_dist = _SL_MAX
            tp = round(price - max(_TP_MIN, min(_TP_MAX, sl_dist * 1.5)), 2)

        # TP must be >= 2x spread to net positive
        tp_dist = abs(tp - price)
        if tp_dist < spread * 2:
            return _no(f"TP {tp_dist:.2f} < 2x spread {spread*2:.2f}")

        rr = tp_dist / sl_dist if sl_dist > 0 else 0

        self._session_trades += 1

        reasons = [
            f"Sweep {direction} @ {sweep_level:.2f}",
            f"Displacement body {body_ratio:.0%}",
            f"EMA20 slope {ema20_slope:+.3f}",
            f"Tick vel {ts['velocity']:.1f}",
            f"ATR {atr_val:.3f}",
        ]

        return {
            "signal": signal,
            "entry": price,
            "sl": sl,
            "tp": tp,
            "sl_distance": round(sl_dist, 2),
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
            },
            # Scalp-specific management params (read by trade manager)
            "_scalp": True,
            "_be_trigger": _BE_TRIGGER,
            "_timeout": _TIMEOUT,
        }

    def _detect_sweep(self, h: np.ndarray, l: np.ndarray, c: np.ndarray,
                      o: np.ndarray, price: float) -> Optional[Dict]:
        """Detect liquidity sweep in last N M1 candles."""
        n = len(h)
        if n < _SWEEP_LOOKBACK + 2:
            return None

        start = n - _SWEEP_LOOKBACK - 1
        end = n - 2  # exclude current candle (it's the displacement)
        curr = n - 1  # current candle

        # --- LONG sweep: equal lows swept then bullish displacement ---
        for i in range(start, end - 2):
            for j in range(i + 2, end):
                if abs(l[i] - l[j]) < _SWEEP_TOL:  # equal lows
                    eq_low = min(l[i], l[j])
                    # Check if a candle between j and curr wicked below
                    for k in range(j + 1, curr + 1):
                        if l[k] < eq_low - _SWEEP_TOL:
                            # And current candle closes above the equal low (reversal)
                            if c[curr] > eq_low and c[curr] > o[curr]:
                                return {
                                    "direction": "LONG",
                                    "level": round(float(l[k]), 2),
                                    "eq_level": round(float(eq_low), 2),
                                }

        # --- SHORT sweep: equal highs swept then bearish displacement ---
        for i in range(start, end - 2):
            for j in range(i + 2, end):
                if abs(h[i] - h[j]) < _SWEEP_TOL:  # equal highs
                    eq_high = max(h[i], h[j])
                    for k in range(j + 1, curr + 1):
                        if h[k] > eq_high + _SWEEP_TOL:
                            if c[curr] < eq_high and c[curr] < o[curr]:
                                return {
                                    "direction": "SHORT",
                                    "level": round(float(h[k]), 2),
                                    "eq_level": round(float(eq_high), 2),
                                }

        return None


def _no(reason: str) -> Dict:
    return {"signal": "NO_TRADE", "reason": reason}
