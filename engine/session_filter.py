"""
Session filter — session-aware trading control.
"""
from datetime import datetime, timezone
import config as cfg

_SESSION_STARTS = (cfg.ASIAN_START, cfg.LONDON_START, cfg.NY_START)


def get_session() -> str:
    h = datetime.now(timezone.utc).hour
    if h < cfg.LONDON_START:
        return "ASIAN"
    if h < cfg.NY_START:
        return "LONDON"
    if h < 22:
        return "NEW_YORK"
    return "ASIAN"


def is_session_open_blocked() -> tuple:
    now = datetime.now(timezone.utc)
    h, m = now.hour, now.minute
    if h in _SESSION_STARTS and m < cfg.SESSION_BLOCK_MINUTES:
        names = {cfg.ASIAN_START: "ASIAN", cfg.LONDON_START: "LONDON", cfg.NY_START: "NEW_YORK"}
        return True, f"{names.get(h, 'UNKNOWN')} session open protection ({cfg.SESSION_BLOCK_MINUTES - m}m remaining)"
    return False, ""


def is_market_open() -> bool:
    now = datetime.now(timezone.utc)
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
