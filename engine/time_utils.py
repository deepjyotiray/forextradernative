"""
Shared time helpers for consistent IST-facing tracker timelines.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from .backtest_context import get_backtest_now


UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))


def coerce_utc(value) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except Exception:
            return None
    return None


def now_utc() -> datetime:
    override = get_backtest_now()
    if isinstance(override, datetime):
        return override.astimezone(UTC) if override.tzinfo else override.replace(tzinfo=UTC)
    return datetime.now(UTC)


def now_ist() -> datetime:
    return now_utc().astimezone(IST)


def now_ts() -> float:
    return now_utc().timestamp()


def isoformat_ist(value=None) -> str:
    dt = coerce_utc(value) if value is not None else now_utc()
    if dt is None:
        dt = now_utc()
    return dt.astimezone(IST).isoformat()


def date_str_ist(value=None) -> str:
    dt = coerce_utc(value) if value is not None else now_utc()
    if dt is None:
        dt = now_utc()
    return dt.astimezone(IST).strftime("%Y-%m-%d")

