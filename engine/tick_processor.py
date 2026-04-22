"""
Tick Processor — rolling tick buffer for micro-execution intelligence.
Tracks velocity, momentum shift, spread stability.
"""
import time
from collections import deque
from typing import Dict, Optional


class TickProcessor:
    def __init__(self, buffer_size: int = 200):
        self._ticks: deque = deque(maxlen=buffer_size)
        self._last_ts: float = 0

    def feed(self, tick: Dict):
        now = time.time()
        self._ticks.append({
            "bid": tick["bid"], "ask": tick["ask"],
            "spread": tick.get("spread", tick["ask"] - tick["bid"]),
            "ts": now,
        })
        self._last_ts = now

    def snapshot(self) -> Dict:
        if len(self._ticks) < 5:
            return {"ready": False}
        ticks = list(self._ticks)
        n = len(ticks)
        now = ticks[-1]["ts"]
        window = now - ticks[0]["ts"]
        velocity = n / window if window > 0 else 0

        # Price momentum over last N ticks
        recent = ticks[-20:] if n >= 20 else ticks
        price_delta = recent[-1]["bid"] - recent[0]["bid"]
        dt = recent[-1]["ts"] - recent[0]["ts"]
        momentum = price_delta / dt if dt > 0 else 0

        # Directional ticks
        dir_up = sum(1 for i in range(1, len(recent)) if recent[i]["bid"] > recent[i-1]["bid"])
        dir_dn = sum(1 for i in range(1, len(recent)) if recent[i]["bid"] < recent[i-1]["bid"])
        dir_ticks = dir_up - dir_dn

        # Spread stats
        spreads = [t["spread"] for t in ticks[-50:]]
        avg_spread = sum(spreads) / len(spreads)
        max_spread = max(spreads)
        spread_stable = max_spread < avg_spread * 2.5

        # Tick acceleration (compare velocity of last 10 vs prev 10)
        accel = 0.0
        if n >= 20:
            mid = n // 2
            t1 = ticks[mid]["ts"] - ticks[0]["ts"]
            t2 = ticks[-1]["ts"] - ticks[mid]["ts"]
            v1 = mid / t1 if t1 > 0 else 0
            v2 = (n - mid) / t2 if t2 > 0 else 0
            accel = v2 - v1

        return {
            "ready": True,
            "velocity": round(velocity, 2),
            "momentum": round(momentum, 4),
            "dir_ticks": dir_ticks,
            "spread": round(ticks[-1]["spread"], 3),
            "avg_spread": round(avg_spread, 3),
            "max_spread": round(max_spread, 3),
            "spread_stable": spread_stable,
            "acceleration": round(accel, 2),
            "tick_count": n,
        }

    def is_chaotic(self) -> bool:
        """True if tick behavior looks like news spike — avoid entry."""
        s = self.snapshot()
        if not s.get("ready"):
            return False
        return (s["velocity"] > 50 and not s["spread_stable"]) or s["max_spread"] > 1.0
