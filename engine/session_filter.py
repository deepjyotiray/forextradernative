"""
Session filter — session-aware trading control.
"""
from datetime import datetime, timezone
import config as cfg
from .backtest_context import get_backtest_now

_SESSION_STARTS = (cfg.ASIAN_START, cfg.LONDON_START, cfg.NY_START)


def get_session() -> str:
    return get_session_at(_now_utc())


def get_session_at(now: datetime) -> str:
    h = now.hour
    if h < cfg.LONDON_START:
        return "ASIAN"
    if h < cfg.NY_START:
        return "LONDON"
    if h < 22:
        return "NEW_YORK"
    return "ASIAN"


def get_session_state(now: datetime | None = None) -> dict:
    return get_session_state_at(now or _now_utc())


def get_session_state_at(now: datetime) -> dict:
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)

    rollover = _is_rollover_hour(now)
    market_open = is_market_open_at(now)
    session = get_session_at(now) if market_open else "CLOSED"
    phase = "OVERLAP" if cfg.TRADE_WINDOW_OVERLAP_START <= now.hour < cfg.TRADE_WINDOW_OVERLAP_END else session

    allowed = market_open
    premium_only = False
    reason = f"SESSION_PASS: {phase}"

    if rollover:
        allowed = bool(getattr(cfg, "TRADE_ROLLOVER", False))
        reason = "SESSION_PASS: ROLLOVER" if allowed else "SESSION_BLOCK: ROLLOVER"
        return {
            "session": "ROLLOVER",
            "phase": "ROLLOVER",
            "market_open": False,
            "rollover": True,
            "allowed": allowed,
            "premium_only": False,
            "reason": reason,
            "label": "ROLLOVER",
        }

    if not market_open:
        allowed = bool(getattr(cfg, "TRADE_CLOSED_SESSION", False))
        reason = "SESSION_PASS: CLOSED" if allowed else "SESSION_BLOCK: CLOSED"
        return {
            "session": "CLOSED",
            "phase": "CLOSED",
            "market_open": False,
            "rollover": False,
            "allowed": allowed,
            "premium_only": False,
            "reason": reason,
            "label": "CLOSED",
        }

    if session == "ASIAN":
        if bool(getattr(cfg, "PREMIUM_ONLY_ASIA", True)):
            allowed = True
            premium_only = True
            reason = "SESSION_PREMIUM_ONLY: ASIA"
        else:
            allowed = bool(getattr(cfg, "TRADE_ASIA_SESSION", False))
            reason = "SESSION_PASS: ASIA" if allowed else "SESSION_BLOCK: ASIA"
    else:
        allowed = True
        reason = f"SESSION_PASS: {phase}"

    return {
        "session": session,
        "phase": phase,
        "market_open": market_open,
        "rollover": False,
        "allowed": allowed,
        "premium_only": premium_only,
        "reason": reason,
        "label": phase,
    }


def is_session_open_blocked() -> tuple:
    return is_session_open_blocked_at(_now_utc())


def is_session_open_blocked_at(now: datetime) -> tuple:
    h, m = now.hour, now.minute
    if h in _SESSION_STARTS and m < cfg.SESSION_BLOCK_MINUTES:
        names = {cfg.ASIAN_START: "ASIAN", cfg.LONDON_START: "LONDON", cfg.NY_START: "NEW_YORK"}
        return True, f"{names.get(h, 'UNKNOWN')} session open protection ({cfg.SESSION_BLOCK_MINUTES - m}m remaining)"
    return False, ""


def is_market_open() -> bool:
    return is_market_open_at(_now_utc())


def is_market_open_at(now: datetime) -> bool:
    wd, h = now.weekday(), now.hour
    if wd == 5:
        return False
    if wd == 6:
        return h >= 22
    if wd == 4 and h >= 22:
        return False
    if h == 21:
        return False
    return True


def _is_rollover_hour(now: datetime) -> bool:
    wd, h = now.weekday(), now.hour
    return wd not in (5,) and h == 21


def _now_utc() -> datetime:
    override = get_backtest_now()
    if isinstance(override, datetime):
        return override.astimezone(timezone.utc)
    return datetime.now(timezone.utc)
