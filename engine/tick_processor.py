"""
Tick Processor — rolling tick buffer with spread model.
Tracks velocity, momentum, spread (mean/std/percentile), direction stability.
"""
import time
import math
from collections import deque
from typing import Dict, Optional


def _tick_field(raw_ticks, idx: int, name: str, default=0.0):
    try:
        tick = raw_ticks[idx]
        if isinstance(tick, dict):
            return tick.get(name, default)
        if hasattr(tick, name):
            return getattr(tick, name)
        return tick[name]
    except Exception:
        return default


def analyze_tick_pressure(raw_ticks) -> Dict:
    if raw_ticks is None:
        return {"ready": False}
    try:
        count = len(raw_ticks)
    except Exception:
        return {"ready": False}
    if count < 5:
        return {"ready": False, "tick_count": count}

    mids = []
    spreads = []
    times = []
    for idx in range(count):
        bid = float(_tick_field(raw_ticks, idx, "bid", 0.0) or 0.0)
        ask = float(_tick_field(raw_ticks, idx, "ask", 0.0) or 0.0)
        ts_msc = float(_tick_field(raw_ticks, idx, "time_msc", 0.0) or 0.0)
        ts = float(_tick_field(raw_ticks, idx, "time", 0.0) or 0.0)
        mid = (bid + ask) / 2.0 if ask or bid else 0.0
        mids.append(mid)
        spreads.append(max(0.0, ask - bid))
        times.append(ts_msc if ts_msc > 0 else ts * 1000.0)

    duration_ms = max(1.0, times[-1] - times[0])
    duration_s = duration_ms / 1000.0
    burst_rate = count / duration_s if duration_s > 0 else 0.0

    up_events = 0
    down_events = 0
    bid_up = 0
    bid_down = 0
    ask_up = 0
    ask_down = 0
    for idx in range(1, count):
        bid_delta = float(_tick_field(raw_ticks, idx, "bid", 0.0) or 0.0) - float(_tick_field(raw_ticks, idx - 1, "bid", 0.0) or 0.0)
        ask_delta = float(_tick_field(raw_ticks, idx, "ask", 0.0) or 0.0) - float(_tick_field(raw_ticks, idx - 1, "ask", 0.0) or 0.0)
        if bid_delta > 0:
            bid_up += 1
        elif bid_delta < 0:
            bid_down += 1
        if ask_delta > 0:
            ask_up += 1
        elif ask_delta < 0:
            ask_down += 1
        if bid_delta > 0 or ask_delta > 0:
            up_events += 1
        elif bid_delta < 0 or ask_delta < 0:
            down_events += 1

    directional_events = max(1, up_events + down_events)
    pressure_score = (up_events - down_events) / directional_events
    price_move = mids[-1] - mids[0]
    price_move_per_second = price_move / duration_s if duration_s > 0 else 0.0
    spread_mean = sum(spreads) / len(spreads)
    spread_std = math.sqrt(sum((s - spread_mean) ** 2 for s in spreads) / len(spreads)) if spreads else 0.0
    baseline_spread = sum(spreads[:-5]) / max(1, len(spreads[:-5])) if len(spreads) > 5 else spread_mean
    spread_shock = spreads[-1] - baseline_spread

    first_half = max(2, count // 2)
    first_window_ms = max(1.0, times[first_half - 1] - times[0])
    second_window_ms = max(1.0, times[-1] - times[first_half])
    first_burst = first_half / (first_window_ms / 1000.0)
    second_burst = (count - first_half) / (second_window_ms / 1000.0) if count - first_half > 0 else 0.0
    burst_decay = (second_burst / first_burst) if first_burst > 0 else 1.0

    bias = "NEUTRAL"
    if pressure_score >= 0.15 or price_move > 0.05:
        bias = "LONG"
    elif pressure_score <= -0.15 or price_move < -0.05:
        bias = "SHORT"

    return {
        "ready": True,
        "tick_count": count,
        "burst_rate": round(burst_rate, 2),
        "pressure_score": round(pressure_score, 3),
        "directional_bias": bias,
        "price_move": round(price_move, 3),
        "price_move_per_second": round(price_move_per_second, 4),
        "spread_mean": round(spread_mean, 3),
        "spread_std": round(spread_std, 4),
        "spread_shock": round(spread_shock, 3),
        "burst_decay": round(burst_decay, 3),
        "quote_up_ratio": round(up_events / directional_events, 3),
        "quote_down_ratio": round(down_events / directional_events, 3),
        "bid_up_ratio": round(bid_up / max(1, bid_up + bid_down), 3),
        "ask_down_ratio": round(ask_down / max(1, ask_up + ask_down), 3),
        "favorable_long": pressure_score >= 0.20 and price_move >= -0.02,
        "favorable_short": pressure_score <= -0.20 and price_move <= 0.02,
        "pressure_fading": burst_decay < 0.80,
    }


def entry_pressure_block_reason(
    direction: str,
    tick_pressure: Dict,
    *,
    strong_threshold: float = 0.35,
    short_positive_veto_threshold: float = 0.05,
) -> Optional[str]:
    if not tick_pressure or not tick_pressure.get("ready"):
        return None

    trade_direction = str(direction or "").upper()
    if trade_direction == "BUY":
        trade_direction = "LONG"
    elif trade_direction == "SELL":
        trade_direction = "SHORT"
    if trade_direction not in {"LONG", "SHORT"}:
        return None

    score = float(tick_pressure.get("pressure_score", 0.0) or 0.0)
    bias = str(tick_pressure.get("directional_bias", "NEUTRAL") or "NEUTRAL").upper()
    strong_limit = abs(float(strong_threshold or 0.0))
    short_veto_limit = max(0.0, float(short_positive_veto_threshold or 0.0))

    if trade_direction == "LONG" and bias == "SHORT" and score <= -strong_limit:
        return "Tick pressure strongly SHORT against LONG setup"

    if trade_direction == "SHORT":
        if (short_veto_limit > 0 and score >= short_veto_limit) or (short_veto_limit <= 0 and score > 0):
            return f"Tick pressure positive ({score:+.2f}) against SHORT setup"
        if bias == "LONG" and score >= strong_limit:
            return "Tick pressure strongly LONG against SHORT setup"

    return None


class TickProcessor:
    def __init__(self, buffer_size: int = 500):
        self._ticks: deque = deque(maxlen=buffer_size)

    def feed(self, tick: Dict):
        ts = tick.get("ts")
        if ts is None:
            ts = time.time()
        self._ticks.append({
            "bid": tick["bid"], "ask": tick["ask"],
            "spread": tick.get("spread", tick["ask"] - tick["bid"]),
            "ts": float(ts),
            "volume": float(tick.get("volume", 0.0) or 0.0),
            "volume_real": float(tick.get("volume_real", 0.0) or 0.0),
            "flags": int(tick.get("flags", 0) or 0),
        })

    def snapshot(self) -> Dict:
        if len(self._ticks) < 10:
            return {"ready": False}
        ticks = list(self._ticks)
        n = len(ticks)
        now = ticks[-1]["ts"]
        window = now - ticks[0]["ts"]
        velocity = n / window if window > 0 else 0

        # Recent ticks for momentum/direction
        recent = ticks[-30:] if n >= 30 else ticks
        rn = len(recent)
        price_delta = recent[-1]["bid"] - recent[0]["bid"]
        dt = recent[-1]["ts"] - recent[0]["ts"]
        momentum = price_delta / dt if dt > 0 else 0

        # Directional ticks
        dir_up = sum(1 for i in range(1, rn) if recent[i]["bid"] > recent[i-1]["bid"])
        dir_dn = sum(1 for i in range(1, rn) if recent[i]["bid"] < recent[i-1]["bid"])
        total_directional = max(1, dir_up + dir_dn)
        dir_ticks = dir_up - dir_dn
        buy_ratio = dir_up / total_directional
        sell_ratio = dir_dn / total_directional
        dir_pct = buy_ratio

        # Direction stability: count sign changes in last 20 ticks
        sign_changes = 0
        for i in range(2, min(21, rn)):
            d1 = recent[-i+1]["bid"] - recent[-i]["bid"]
            d2 = recent[-i]["bid"] - recent[-i-1]["bid"] if i+1 <= rn else 0
            if d1 * d2 < 0:
                sign_changes += 1
        dir_stable = sign_changes < 8  # < 8 reversals in 20 ticks = stable

        # Velocity trend (is it increasing?)
        vel_increasing = False
        if n >= 40:
            mid = n - 20
            t1 = ticks[mid]["ts"] - ticks[mid-20]["ts"]
            t2 = ticks[-1]["ts"] - ticks[mid]["ts"]
            v1 = 20 / t1 if t1 > 0 else 0
            v2 = 20 / t2 if t2 > 0 else 0
            vel_increasing = v2 > v1 * 1.1

        # === SPREAD MODEL ===
        # Short window: last 50-150 ticks (1-3 seconds worth)
        short_spreads = [t["spread"] for t in ticks[-100:]]
        spread_mean = sum(short_spreads) / len(short_spreads)
        spread_std = math.sqrt(sum((s - spread_mean)**2 for s in short_spreads) / len(short_spreads))

        # Long window: last 5-10 minutes for percentile
        cutoff = now - 600  # 10 minutes
        long_spreads = [t["spread"] for t in ticks if t["ts"] >= cutoff]
        if len(long_spreads) < 20:
            long_spreads = short_spreads  # fallback

        # Percentile: what % of recent spreads are >= current spread
        current_spread = ticks[-1]["spread"]
        below = sum(1 for s in long_spreads if s <= current_spread)
        spread_pctl = below / len(long_spreads) if long_spreads else 0.5

        spread_stable = spread_std < 0.05 and current_spread < spread_mean * 1.5

        return {
            "ready": True,
            "velocity": round(velocity, 2),
            "vel_increasing": vel_increasing,
            "momentum": round(momentum, 4),
            "dir_ticks": dir_ticks,
            "dir_pct": round(dir_pct, 3),
            "buy_ratio": round(buy_ratio, 3),
            "sell_ratio": round(sell_ratio, 3),
            "dir_stable": dir_stable,
            "spread": round(current_spread, 3),
            "spread_mean": round(spread_mean, 3),
            "spread_std": round(spread_std, 4),
            "spread_pctl": round(spread_pctl, 3),
            "spread_stable": spread_stable,
            "tick_count": n,
        }

    def check_spread_ok(self, max_spread: float, max_pctl: float = 0.5) -> tuple:
        """Pre-execution spread check. Returns (ok, reason)."""
        s = self.snapshot()
        if not s.get("ready"):
            return False, "Tick buffer not ready"
        if s["spread_mean"] > max_spread:
            return False, f"Spread mean {s['spread_mean']:.3f} > {max_spread}"
        if s["spread_std"] > 0.08:
            return False, f"Spread spiking (std {s['spread_std']:.4f})"
        if s["spread_pctl"] > max_pctl:
            return False, f"Spread percentile {s['spread_pctl']:.0%} > {max_pctl:.0%}"
        return True, "OK"

    def check_execution_quality(self) -> tuple:
        """Pre-execution quality gate. Returns (ok, reason)."""
        s = self.snapshot()
        if not s.get("ready"):
            return False, "Not ready"
        if not s["vel_increasing"] and s["velocity"] < 3:
            return False, f"Low/falling velocity ({s['velocity']})"
        if not s["dir_stable"]:
            return False, "Direction oscillating"
        if not s["spread_stable"]:
            return False, "Spread unstable"
        return True, "OK"

    def spread_changed(self, signal_spread: float, max_delta: float = 0.03) -> bool:
        """Check if spread widened since signal was generated."""
        if not self._ticks:
            return False
        current = self._ticks[-1]["spread"]
        return (current - signal_spread) > max_delta

    def is_chaotic(self) -> bool:
        s = self.snapshot()
        if not s.get("ready"):
            return False
        return (s["velocity"] > 50 and not s["spread_stable"]) or s["spread"] > 1.0
