"""
Session Detector — thin proxy over session_filter.get_session_state_at().

All callers should use session_filter directly. This module exists only for
backward compatibility and delegates every call to the single authoritative
source so session names are always consistent.
"""
from datetime import datetime, timezone, timedelta
from typing import Tuple

from .session_filter import get_session_state_at, get_session_at, is_market_open_at

_IST = timezone(timedelta(hours=5, minutes=30))


class SessionDetector:
    """Thin wrapper — delegates to session_filter for consistency."""

    @staticmethod
    def get_current_session() -> Tuple[str, str]:
        """Return (session, phase) using the authoritative session_filter logic."""
        now_utc = datetime.now(timezone.utc)
        state = get_session_state_at(now_utc)
        session = state.get("session", "CLOSED")
        phase = state.get("phase", session)
        return session, phase

    @staticmethod
    def get_session_info() -> dict:
        """Return detailed session info consistent with master_trade_gate."""
        now_utc = datetime.now(timezone.utc)
        state = get_session_state_at(now_utc)
        session = state.get("session", "CLOSED")
        phase = state.get("phase", session)
        return {
            "session": session,
            "phase": phase,
            "utc_time": now_utc.strftime("%H:%M:%S"),
            "ist_time": now_utc.astimezone(_IST).strftime("%H:%M:%S"),
            "is_major_session": session in ("LONDON", "NEW_YORK"),
            "is_overlap": phase == "OVERLAP",
            "is_quiet_time": session in ("CLOSED", "ROLLOVER", "ASIAN") and not state.get("premium_only"),
            "allowed": state.get("allowed", False),
            "premium_only": state.get("premium_only", False),
            "reason": state.get("reason", ""),
        }

    @staticmethod
    def is_trading_hours() -> bool:
        """True when session_filter allows trading."""
        now_utc = datetime.now(timezone.utc)
        return get_session_state_at(now_utc).get("allowed", False)

    @staticmethod
    def get_next_session_change() -> dict:
        """Approximate time until the next session boundary (UTC hours)."""
        now_utc = datetime.now(timezone.utc)
        hour = now_utc.hour
        boundaries = [(0, "ASIAN_START"), (7, "LONDON_START"), (13, "NEW_YORK_START"), (22, "CLOSED_START")]
        next_boundary = next(((h, n) for h, n in boundaries if hour < h), (24, "ASIAN_START"))
        next_hour, next_name = next_boundary
        if next_hour == 24:
            next_time = now_utc.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        else:
            next_time = now_utc.replace(hour=next_hour, minute=0, second=0, microsecond=0)
        minutes_until = int((next_time - now_utc).total_seconds() / 60)
        return {
            "next_session": next_name,
            "next_time_utc": next_time.strftime("%H:%M:%S"),
            "minutes_until": minutes_until,
            "time_until": f"{minutes_until // 60}h {minutes_until % 60}m",
        }
