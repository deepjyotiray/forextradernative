"""
Backtest execution context helpers.

Lets existing strategy logging functions detect backtest mode and avoid
writing to live log files.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Callable, Dict, Optional


_IS_BACKTEST: ContextVar[bool] = ContextVar("_IS_BACKTEST", default=False)
_DECISION_SINK: ContextVar[Optional[Callable[[Dict], None]]] = ContextVar(
    "_DECISION_SINK",
    default=None,
)
_BACKTEST_NOW: ContextVar[Optional[datetime]] = ContextVar("_BACKTEST_NOW", default=None)


def is_backtest_mode() -> bool:
    return bool(_IS_BACKTEST.get())


def emit_backtest_decision(decision: Dict):
    sink = _DECISION_SINK.get()
    if sink is not None:
        sink(decision)


def set_backtest_now(dt: Optional[datetime]):
    if dt is None:
        _BACKTEST_NOW.set(None)
        return
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    _BACKTEST_NOW.set(dt)


def get_backtest_now() -> Optional[datetime]:
    return _BACKTEST_NOW.get()


@contextmanager
def backtest_context(decision_sink: Optional[Callable[[Dict], None]] = None):
    token_mode = _IS_BACKTEST.set(True)
    token_sink = _DECISION_SINK.set(decision_sink)
    try:
        yield
    finally:
        _DECISION_SINK.reset(token_sink)
        _IS_BACKTEST.reset(token_mode)
