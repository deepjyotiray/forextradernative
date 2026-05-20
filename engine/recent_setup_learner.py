"""
Recent setup learner - lightweight recency-biased setup ranking from closed trades.

This does not place trades on its own. It scores a candidate setup using recent
non-breakeven outcomes so AUTO arbitration can prefer what is actually working
right now by strategy, direction, session, and regime.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import config as cfg
from .order_database import OrderDatabase

_CACHE_TTL_SECONDS = 60.0
_CACHE_ROWS: List[Dict[str, Any]] = []
_CACHE_AT = 0.0
_ORDER_DB = OrderDatabase()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_direction(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"BUY", "LONG"}:
        return "BUY"
    if text in {"SELL", "SHORT"}:
        return "SELL"
    return text


def _normalize_session(value: Any) -> str:
    return str(value or "").strip().upper() or "UNKNOWN"


def _normalize_regime(value: Any) -> str:
    return str(value or "").strip().upper() or "UNKNOWN"


def _risk_unit(order: Dict[str, Any]) -> float:
    sl_distance = abs(_safe_float(order.get("sl_distance"), 0.0))
    volume = _safe_float(order.get("initial_volume"), _safe_float(order.get("volume"), 0.0))
    risk = sl_distance * volume * _safe_float(getattr(cfg, "PIP_VALUE_PER_LOT", 0.0), 0.0)
    return risk if risk > 0 else 0.0


def _final_r(order: Dict[str, Any]) -> float:
    stored = order.get("final_r")
    if stored is not None:
        return _safe_float(stored, 0.0)
    pnl = _safe_float(order.get("final_pnl"), 0.0)
    risk = _risk_unit(order)
    return round(pnl / risk, 3) if risk > 0 else 0.0


def _feature_value(order: Dict[str, Any], key: str, fallback: str = "") -> str:
    features = order.get("features") or {}
    return str(features.get(key) or order.get(key) or fallback)


def _load_recent_rows() -> List[Dict[str, Any]]:
    global _CACHE_ROWS, _CACHE_AT
    now = time.time()
    if now - _CACHE_AT <= _CACHE_TTL_SECONDS and _CACHE_ROWS:
        return _CACHE_ROWS

    rows = _ORDER_DB.get_closed_orders(days=14)
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        pnl = _safe_float(row.get("final_pnl"), 0.0)
        if pnl == 0.0:
            continue
        filtered.append(
            {
                "strategy": str(row.get("strategy") or "").upper(),
                "direction": _normalize_direction(row.get("direction")),
                "session": _normalize_session(_feature_value(row, "session", row.get("session_type") or "UNKNOWN")),
                "regime": _normalize_regime(_feature_value(row, "regime", row.get("market_phase") or "UNKNOWN")),
                "final_r": _final_r(row),
            }
        )

    _CACHE_ROWS = filtered
    _CACHE_AT = now
    return filtered


def _match_rows(rows: List[Dict[str, Any]], strategy: str, direction: str, session: str, regime: str) -> Dict[str, List[Dict[str, Any]]]:
    strategy = str(strategy or "").upper()
    direction = _normalize_direction(direction)
    session = _normalize_session(session)
    regime = _normalize_regime(regime)

    exact = [
        r for r in rows
        if r["strategy"] == strategy and r["direction"] == direction and r["session"] == session and r["regime"] == regime
    ]
    session_only = [
        r for r in rows
        if r["strategy"] == strategy and r["direction"] == direction and r["session"] == session
    ]
    strategy_only = [
        r for r in rows
        if r["strategy"] == strategy and r["direction"] == direction
    ]
    return {
        "exact": exact,
        "session_only": session_only,
        "strategy_only": strategy_only,
    }


def _score_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "sample_size": 0,
            "win_rate": 0.0,
            "avg_final_r": 0.0,
            "expectancy_r": 0.0,
            "score_bonus": 0.0,
        }

    sample_size = len(rows)
    final_rs = [_safe_float(r.get("final_r"), 0.0) for r in rows]
    winners = [v for v in final_rs if v > 0]
    losers = [v for v in final_rs if v < 0]
    win_rate = len(winners) / sample_size if sample_size else 0.0
    avg_final_r = sum(final_rs) / sample_size if sample_size else 0.0
    avg_win_r = (sum(winners) / len(winners)) if winners else 0.0
    avg_loss_r = (abs(sum(losers)) / len(losers)) if losers else 0.0
    expectancy_r = (win_rate * avg_win_r) - ((1.0 - win_rate) * avg_loss_r)

    # Reliability-scaled bounded score for arbitration only.
    reliability = min(1.0, sample_size / 12.0)
    raw_score = (expectancy_r * 10.0) + ((win_rate - 0.5) * 8.0)
    score_bonus = max(-8.0, min(8.0, raw_score * reliability))

    return {
        "sample_size": sample_size,
        "win_rate": round(win_rate, 3),
        "avg_final_r": round(avg_final_r, 3),
        "expectancy_r": round(expectancy_r, 3),
        "score_bonus": round(score_bonus, 3),
    }


def assess_signal(signal: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
    rows = _load_recent_rows()
    strategy = str(signal.get("_strategy_name") or signal.get("strategy") or "").upper()
    direction = signal.get("signal")
    session = signal.get("_session") or (market_data.get("tick_snapshot") or {}).get("session") or "UNKNOWN"
    regime = (market_data.get("regime") or {}).get("state") or "UNKNOWN"

    matched = _match_rows(rows, strategy, direction, session, regime)
    scope = "strategy_only"
    scope_rows = matched["strategy_only"]
    if len(matched["exact"]) >= 4:
        scope = "exact"
        scope_rows = matched["exact"]
    elif len(matched["session_only"]) >= 6:
        scope = "session_only"
        scope_rows = matched["session_only"]

    stats = _score_rows(scope_rows)
    return {
        "enabled": True,
        "scope": scope,
        "strategy": strategy,
        "direction": _normalize_direction(direction),
        "session": _normalize_session(session),
        "regime": _normalize_regime(regime),
        **stats,
    }
