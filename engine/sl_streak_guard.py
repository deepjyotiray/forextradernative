"""
SL Streak Guard — tightens re-entry conditions after back-to-back SL hits.

When 2+ consecutive SL hits are detected for a strategy:
  1. A mandatory cooldown is enforced (default 15 min) — no new entries allowed.
  2. After cooldown, a STRICT gate activates until a winning trade resets it:
       - D1 + H4 + H1 must all trend the same direction (vs normal D1+H4 only)
       - Macro must actively support the direction (not just neutral)
       - Minimum confidence score raised to STRICT_MIN_CONFIDENCE (80)
       - Pullback depth must be in tighter 0.35–0.65 range (vs 0.30–0.70)

State is in-memory only — resets on restart (intentional: stale streak state
from a previous session should not carry over).
"""
from __future__ import annotations

import time
from typing import Dict

# Cooldown after 2+ consecutive SL hits before any re-entry is considered
COOLDOWN_SECONDS = 15 * 60  # 15 minutes

# Minimum consecutive SL hits to trigger the guard
TRIGGER_COUNT = 2

# Confidence threshold required while strict gate is active (0–100 scale)
STRICT_MIN_CONFIDENCE = 80

# Tighter pullback window while strict gate is active
STRICT_PULLBACK_MIN = 0.35
STRICT_PULLBACK_MAX = 0.65


class SLStreakGuard:
    """Per-strategy SL streak tracker."""

    def __init__(self):
        # {strategy_name: {"count": int, "last_sl_ts": float, "strict_until_win": bool}}
        self._state: Dict[str, Dict] = {}

    def _get(self, strategy: str) -> Dict:
        if strategy not in self._state:
            self._state[strategy] = {"count": 0, "last_sl_ts": 0.0, "strict_until_win": False}
        return self._state[strategy]

    def record_sl(self, strategy: str):
        """Call this every time a trade closes as an SL hit."""
        s = self._get(strategy)
        s["count"] += 1
        s["last_sl_ts"] = time.time()
        if s["count"] >= TRIGGER_COUNT:
            s["strict_until_win"] = True

    def record_win(self, strategy: str):
        """Call this every time a trade closes as a win — resets the guard."""
        s = self._get(strategy)
        s["count"] = 0
        s["strict_until_win"] = False
        # last_sl_ts intentionally kept so cooldown can still expire naturally

    def cooldown_remaining(self, strategy: str) -> float:
        """Seconds remaining in mandatory cooldown. 0 means cooldown is over."""
        s = self._get(strategy)
        if s["count"] < TRIGGER_COUNT or s["last_sl_ts"] <= 0:
            return 0.0
        elapsed = time.time() - s["last_sl_ts"]
        return max(0.0, COOLDOWN_SECONDS - elapsed)

    def is_strict_active(self, strategy: str) -> bool:
        """True if the strict re-entry gate is active (cooldown over but no win yet)."""
        s = self._get(strategy)
        return s["strict_until_win"] and self.cooldown_remaining(strategy) == 0.0

    def status(self, strategy: str) -> Dict:
        s = self._get(strategy)
        remaining = self.cooldown_remaining(strategy)
        return {
            "consecutive_sl_hits": s["count"],
            "cooldown_remaining_seconds": round(remaining, 1),
            "in_cooldown": remaining > 0,
            "strict_gate_active": self.is_strict_active(strategy),
        }


# Singleton
sl_streak_guard = SLStreakGuard()
