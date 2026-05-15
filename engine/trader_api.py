"""
Auto Trader API routes for FastAPI integration.
Consolidates all trading system endpoints onto a single port.
"""
from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.responses import FileResponse
from starlette.responses import Response as _RawResponse
from typing import Dict, Any, Set
from bisect import bisect_left
import hmac
import json
import os
import time
import asyncio
import uuid
import re
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
import config as cfg
from engine import strategy_configs as _scfg_store
from engine.deployment_metadata import capture_code_snapshot, compare_snapshots
from engine.debug_bundle import build_debug_bundle, build_debug_manifest
from engine.market_context_qa import market_context_qa_service

_IST = timezone(timedelta(hours=5, minutes=30))

try:
    import orjson
    def _fast_json(obj):
        return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY).decode()
except ImportError:
    import json as _json
    class _NumpyEncoder(_json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.bool_, np.integer, np.floating)):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)
    def _fast_json(obj):
        return _json.dumps(obj, cls=_NumpyEncoder, separators=(',', ':'))

# --- WebSocket broadcast infrastructure ---
_ws_clients: Set[WebSocket] = set()
_ws_latest_tick: str = '{}'
_ws_latest_cycle: int = -1

# This will be set by the main application when the auto_trader instance is available
_auto_trader_instance = None
_status_cache = None
_status_cache_time = 0.0
_STATUS_CACHE_TTL = 1.0  # seconds — keep open trades panel fresh

# Separate caches for slow-changing data
_blockers_cache = None
_blockers_cache_time = 0.0
_BLOCKERS_CACHE_TTL = 10.0

_xgb_cache = None
_xgb_cache_time = 0.0
_XGB_CACHE_TTL = 15.0

_perf_cache = None
_perf_cache_time = 0.0
_PERF_CACHE_TTL = 10.0

_closed_history_cache = None
_closed_history_cache_time = 0.0
_CLOSED_HISTORY_CACHE_TTL = 15.0

_heavy_status_cache = None
_heavy_status_cache_time = 0.0
_HEAVY_STATUS_CACHE_TTL = 15.0
_deployment_cache = None
_deployment_cache_time = 0.0
_DEPLOYMENT_CACHE_TTL = 5.0
_ai_reviews_cache = None
_ai_reviews_cache_time = 0.0
_AI_REVIEWS_CACHE_TTL = 10.0
_manual_ai_ideas: Dict[str, Dict[str, Any]] = {}
_MANUAL_AI_IDEA_TTL_SECONDS = 15 * 60
_config_version = 1

router = APIRouter(tags=["trading"])


def _is_local_request(request: Request) -> bool:
    host = str(getattr(getattr(request, "client", None), "host", "") or "").strip().lower()
    return host in {"127.0.0.1", "::1", "localhost"}


def _require_debug_bundle_access(request: Request) -> None:
    required_token = os.getenv("DEBUG_BUNDLE_TOKEN", "").strip()
    if required_token:
        provided = (
            str(request.headers.get("x-debug-bundle-token", "") or "").strip()
            or str(request.query_params.get("token", "") or "").strip()
        )
        if not provided or not hmac.compare_digest(provided, required_token):
            raise HTTPException(status_code=403, detail="Invalid debug bundle token")
        return
    if not _is_local_request(request):
        raise HTTPException(
            status_code=403,
            detail="Debug bundle export is local-only unless DEBUG_BUNDLE_TOKEN is configured",
        )

def _invalidate_status_cache(include_slow: bool = False):
    """Force the next dashboard/status request and WS tick to reflect control changes."""
    global _status_cache, _status_cache_time, _ws_latest_cycle
    global _blockers_cache, _blockers_cache_time, _xgb_cache, _xgb_cache_time
    global _perf_cache, _perf_cache_time, _closed_history_cache, _closed_history_cache_time
    global _heavy_status_cache, _heavy_status_cache_time, _deployment_cache, _deployment_cache_time
    global _ai_reviews_cache, _ai_reviews_cache_time
    _status_cache = None
    _status_cache_time = 0.0
    _ws_latest_cycle = -1
    if include_slow:
        _blockers_cache = None
        _blockers_cache_time = 0.0
        _xgb_cache = None
        _xgb_cache_time = 0.0
        _perf_cache = None
        _perf_cache_time = 0.0
        _closed_history_cache = None
        _closed_history_cache_time = 0.0
        _heavy_status_cache = None
        _heavy_status_cache_time = 0.0
        _deployment_cache = None
        _deployment_cache_time = 0.0
        _ai_reviews_cache = None
        _ai_reviews_cache_time = 0.0

def _bump_config_version():
    global _config_version
    _config_version += 1
    return _config_version


def _prune_manual_ai_ideas():
    now = time.time()
    expired_ids = [
        idea_id
        for idea_id, idea in list(_manual_ai_ideas.items())
        if float((idea or {}).get("expires_at_ts", 0.0) or 0.0) <= now
    ]
    for idea_id in expired_ids:
        _manual_ai_ideas.pop(idea_id, None)


def _store_manual_ai_idea(idea: Dict[str, Any]) -> Dict[str, Any]:
    _prune_manual_ai_ideas()
    idea_id = uuid.uuid4().hex[:12]
    expires_at_ts = time.time() + _MANUAL_AI_IDEA_TTL_SECONDS
    stored = dict(idea or {})
    stored["idea_id"] = idea_id
    stored["expires_at_ts"] = expires_at_ts
    stored["expires_in_seconds"] = _MANUAL_AI_IDEA_TTL_SECONDS
    _manual_ai_ideas[idea_id] = stored
    if len(_manual_ai_ideas) > 40:
        oldest = sorted(
            _manual_ai_ideas.items(),
            key=lambda item: float((item[1] or {}).get("generated_at_ts", 0.0) or 0.0),
        )[:10]
        for old_id, _ in oldest:
            _manual_ai_ideas.pop(old_id, None)
    return stored


def _get_manual_ai_idea_or_404(idea_id: str) -> Dict[str, Any]:
    _prune_manual_ai_ideas()
    idea = _manual_ai_ideas.get(str(idea_id or "").strip())
    if not idea:
        raise HTTPException(status_code=404, detail="AI trade idea not found or expired")
    return idea

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def set_auto_trader_instance(instance):
    """Set the auto trader instance for the API to use."""
    global _auto_trader_instance
    _auto_trader_instance = instance

def get_auto_trader():
    """Get the auto trader instance, raise error if not available."""
    if _auto_trader_instance is None:
        raise HTTPException(status_code=503, detail="Auto trader not initialized")
    return _auto_trader_instance

def _get_cached_blockers() -> dict:
    """Get live blockers with caching (reads JSONL file — expensive)."""
    global _blockers_cache, _blockers_cache_time
    now = time.monotonic()
    if _blockers_cache is not None and (now - _blockers_cache_time) < _BLOCKERS_CACHE_TTL:
        return _blockers_cache
    from engine.decision_logger import get_live_blockers
    try:
        _blockers_cache = get_live_blockers(50)
    except Exception:
        _blockers_cache = {"latest_by_strategy": {}, "top_reasons": [], "recent_skipped_count": 0}
    _blockers_cache_time = now
    return _blockers_cache


def _get_deployment_status(trader) -> dict:
    global _deployment_cache, _deployment_cache_time
    now = time.monotonic()
    if _deployment_cache is not None and (now - _deployment_cache_time) < _DEPLOYMENT_CACHE_TTL:
        return _deployment_cache
    base_dir = Path(__file__).resolve().parent.parent
    runtime_snapshot = getattr(trader, "_deployment_snapshot", None) or capture_code_snapshot(
        base_dir,
        as_runtime=True,
        record_reason="status_recovered_runtime_snapshot",
    )
    current_snapshot = capture_code_snapshot(base_dir)
    _deployment_cache = compare_snapshots(runtime_snapshot, current_snapshot)
    _deployment_cache_time = now
    return _deployment_cache


def _get_live_strategy_blockers(trader) -> dict:
    """Build current per-strategy blockers from the latest in-memory strategy evaluation."""
    latest_by_strategy = {}
    reason_counts = {}

    try:
        results = dict(getattr(trader, "_last_strategy_results", {}) or {})
    except Exception:
        results = {}

    try:
        enabled = list((trader.strat_mgr.enabled_strategies() if trader and trader.strat_mgr else []) or [])
    except Exception:
        enabled = []

    if not results and trader is not None and enabled:
        try:
            positions = list(getattr(trader, "_cached_positions", None) or [])
            results = dict(
                (
                    trader.strat_mgr.evaluate_all(
                        {
                            "m1_df": getattr(trader, "_candles", {}).get("M1"),
                            "m5_df": getattr(trader, "_candles", {}).get("M5"),
                            "m15_df": getattr(trader, "_candles", {}).get("M15"),
                            "m30_df": getattr(trader, "_candles", {}).get("M30"),
                            "h1_df": getattr(trader, "_candles", {}).get("H1"),
                            "h4_df": getattr(trader, "_candles", {}).get("H4"),
                            "d1_df": getattr(trader, "_candles", {}).get("D1"),
                            "symbol": getattr(getattr(trader, "bridge", None), "current_symbol", cfg.SYMBOL),
                            "tick": getattr(trader, "_last_tick", {}) or {},
                            "zones": getattr(trader, "_zones", {}) or {},
                            "indicators": getattr(trader, "_indicators", {}) or {},
                            "tf_context": getattr(trader, "_tf_context", {}) or {},
                            "account": getattr(trader, "_last_account", {}) or {},
                            "positions": positions,
                            "correlation": getattr(trader, "_correlation", {}) or {},
                            "calendar": getattr(trader, "_calendar", {}) or {},
                            "bias": getattr(trader, "_bias", {}) or {},
                            "regime": getattr(trader, "_regime", {}) or {},
                            "liquidity": getattr(trader, "_liquidity", {}) or {},
                            "tick_snapshot": (
                                trader.tick_proc.snapshot() if getattr(trader, "tick_proc", None) else {}
                            ) or {},
                            "tick_pressure": getattr(trader, "_tick_pressure", {}) or {},
                            "_risk_manager": getattr(trader, "risk", None),
                            "now_utc": datetime.now(timezone.utc),
                            "strategy_trade_counts": _strategy_trade_counts_for_trader(trader),
                        },
                        strategy_names=enabled,
                    ).get("all_results")
                    or {}
                )
            )
        except Exception:
            results = {}

    if not results:
        return {"latest_by_strategy": {}, "top_reasons": [], "recent_skipped_count": 0}

    for name in enabled:
        row = results.get(name) or {}
        signal = str(row.get("signal") or "").upper()
        gate_allowed = row.get("gate_allowed")
        reason = str(row.get("reason") or "").strip()
        is_blocked = signal not in ("BUY", "SELL") or gate_allowed is False
        if not is_blocked or not reason:
            continue
        latest_by_strategy[name] = {
            "reason": reason,
            "timestamp": None,
            "spread_mean": None,
            "compression_ok": None,
        }
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    top_reasons = [
        {"reason": reason, "count": count}
        for reason, count in sorted(reason_counts.items(), key=lambda item: item[1], reverse=True)[:5]
    ]
    return {
        "latest_by_strategy": latest_by_strategy,
        "top_reasons": top_reasons,
        "recent_skipped_count": len(latest_by_strategy),
    }


def _merge_blocker_views(history_blockers: dict, live_blockers: dict) -> dict:
    """Prefer live per-strategy blockers while keeping historical frequency counts."""
    history_blockers = history_blockers or {}
    live_blockers = live_blockers or {}

    latest_by_strategy = dict(history_blockers.get("latest_by_strategy") or {})
    latest_by_strategy.update(live_blockers.get("latest_by_strategy") or {})

    return {
        "latest_by_strategy": latest_by_strategy,
        "top_reasons": list(history_blockers.get("top_reasons") or live_blockers.get("top_reasons") or []),
        "recent_skipped_count": max(
            int(history_blockers.get("recent_skipped_count") or 0),
            int(live_blockers.get("recent_skipped_count") or 0),
        ),
    }


def _get_cached_ai_reviews(limit: int = 12) -> list:
    """Get recent AI review transcripts with caching."""
    global _ai_reviews_cache, _ai_reviews_cache_time
    now = time.monotonic()
    if _ai_reviews_cache is not None and (now - _ai_reviews_cache_time) < _AI_REVIEWS_CACHE_TTL:
        return _ai_reviews_cache[:limit]
    from engine.decision_logger import get_recent_ai_reviews
    try:
        _ai_reviews_cache = get_recent_ai_reviews(limit=max(limit, 12))
    except Exception:
        _ai_reviews_cache = []
    _ai_reviews_cache_time = now
    return _ai_reviews_cache[:limit]


_STRATEGY_LABELS = {
    "SMC_CONFLUENCE": "SMC",
    "M15_SUPPORT_RESISTANCE_REJECTION_V1": "M15 SR",
    "M15_SCALP_DEEP": "M15 Scalp Deep",
    "SWEEP_SCALPER": "Sweep Scalper",
    "TREND_CHANNEL": "Trend Channel",
    "SWING_ENGINE": "Swing Engine",
    "INTRADAY_ENGINE": "Intraday Engine",
    "HTF_LONG": "HTF Long",
    "HTF_SHORT": "HTF Short",
}

_STRATEGY_TRIGGER_HINTS = {
    "SMC_CONFLUENCE": "Needs a clean zone interaction, sweep or rejection, supportive tick pressure, and score above the SMC threshold.",
    "M15_SUPPORT_RESISTANCE_REJECTION_V1": "Needs price to reject a strong M15 support or resistance zone with candle confirmation and clean spread.",
    "M15_SCALP_DEEP": "Needs aligned D1 and H4 structure, a strong M15 rejection at a quality zone, supportive micro pressure, and a minimum RR before entry.",
    "SWEEP_SCALPER": "Needs a fresh liquidity sweep or compression release, clear short-term momentum, and session-quality execution.",
    "TREND_CHANNEL": "Needs price to touch a valid trend channel boundary in supertrend direction with enough room to the opposite wall.",
    "SWING_ENGINE": "Needs D1 and H4 to align, price to be near a key swing level, and an H4 rejection or strong directional candle.",
    "INTRADAY_ENGINE": "Needs H1 and M15 alignment, a valid session, a sweep at a known intraday level, and strong M5 confirmation.",
    "HTF_LONG": "Needs DXY and US10Y both trending down (macro bullish), weekly bias STRONG_BULLISH or EARLY_BULLISH, pullback 20-50% into the weekly range, and H1 structure UP.",
    "HTF_SHORT": "Needs DXY and US10Y both trending up (macro bearish), weekly bias STRONG_BEARISH or EARLY_BEARISH, pullback 20-50% into the weekly range, and H1 structure DOWN.",
}

_STRATEGY_RECHECK_SECONDS = {
    "SMC_CONFLUENCE": 180,
    "M15_SUPPORT_RESISTANCE_REJECTION_V1": 900,
    "M15_SCALP_DEEP": 240,
    "SWEEP_SCALPER": 120,
    "TREND_CHANNEL": 900,
    "SWING_ENGINE": 3600,
    "INTRADAY_ENGINE": 300,
    "HTF_LONG": 14400,
    "HTF_SHORT": 14400,
}


def _strategy_label(name: str) -> str:
    key = str(name or "").upper()
    return _STRATEGY_LABELS.get(key, key.replace("_", " ").title())


def _parse_datetime(value: Any):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def _format_wait(seconds: float) -> str:
    total = max(0, int(round(float(seconds or 0.0))))
    if total <= 0:
        return "now"
    if total < 60:
        return f"about {total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"about {minutes}m" if secs < 30 else f"about {minutes + 1}m"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"about {hours}h {minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"about {days}d {hours}h"


def _seconds_until_next_trade_window(now_utc: datetime) -> float:
    hour = now_utc.hour
    windows = sorted(
        [
            (int(getattr(cfg, "TRADE_WINDOW_LONDON_START", 7) or 7), int(getattr(cfg, "TRADE_WINDOW_LONDON_END", 13) or 13)),
            (int(getattr(cfg, "TRADE_WINDOW_OVERLAP_START", 13) or 13), int(getattr(cfg, "TRADE_WINDOW_OVERLAP_END", 17) or 17)),
        ]
    )
    for start, end in windows:
        if start <= hour < end:
            return 0.0
    candidates = []
    for add_days in range(0, 3):
        base = (now_utc + timedelta(days=add_days)).replace(minute=0, second=0, microsecond=0)
        for start, _end in windows:
            candidate = base.replace(hour=start)
            if add_days > 0 or candidate > now_utc:
                candidates.append((candidate - now_utc).total_seconds())
    return min(candidates) if candidates else 0.0


def _calendar_release_seconds(calendar_state: Dict[str, Any], now_utc: datetime):
    if not bool((calendar_state or {}).get("blocked")):
        return None
    event = (calendar_state or {}).get("next_event") or {}
    event_time = _parse_datetime(event.get("time_utc"))
    if event_time is None:
        return None
    after_seconds = max(0, int(getattr(cfg, "CALENDAR_BLOCK_AFTER_MINUTES", 15) or 0)) * 60
    release_time = event_time + timedelta(seconds=after_seconds)
    return max(0.0, (release_time - now_utc).total_seconds())


def _clean_reason(reason: Any) -> str:
    text = str(reason or "").strip()
    if not text:
        return "No active trigger yet."
    return text.replace("AUTO_GATE_ERR:", "Gate error:").strip()


def _clean_close_reason(primary: Any, *fallbacks: Any) -> str:
    for value in (primary, *fallbacks):
        text = str(value or "").strip()
        if text:
            return text
    return "Reason unavailable"


def _strategy_names_for_trader(trader) -> list[str]:
    names = []
    try:
        names = [name for name in getattr(trader.strat_mgr, "available", []) if name and name != "AUTO"]
    except Exception:
        names = []
    if names:
        return names
    try:
        return list(_scfg_store.list_strategies())
    except Exception:
        return []


def _strategy_trade_counts_for_trader(trader) -> Dict[str, int]:
    counts = {str(name).upper(): 0 for name in _strategy_names_for_trader(trader)}
    try:
        open_trades = (
            trader.trades.open_trades.values()
            if getattr(trader, "trades", None) and getattr(trader.trades, "open_trades", None)
            else []
        )
        for trade in open_trades:
            key = str(getattr(trade, "strategy", "") or "").upper()
            if key:
                counts[key] = counts.get(key, 0) + 1
    except Exception:
        pass
    return counts


def _reason_to_trigger_text(strategy_name: str, reason: Any) -> str:
    text = str(reason or "").strip()
    lower = text.lower()
    if not lower:
        return _STRATEGY_TRIGGER_HINTS.get(strategy_name, "Needs its core setup and execution filters to line up.")
    if "outside trade window" in lower or "session" in lower and "not allowed" in lower:
        return "Needs the configured London or overlap session window to be active."
    if "high-impact news" in lower or "event in" in lower or "post-event" in lower:
        return "Needs the news blackout window to clear before a new trade can trigger."
    if "spread" in lower:
        return "Needs spread and execution quality to tighten back inside the strategy limit."
    if "compression" in lower:
        return "Needs a cleaner compression and release structure before entry quality is acceptable."
    if "volume ratio" in lower or "volume" in lower:
        return "Needs stronger participation and follow-through from current order flow."
    if "not aligned" in lower or "trend unclear" in lower or "trend" in lower and "aligned" in lower:
        return "Needs higher-timeframe trend alignment to come back into line."
    if "m15 structure not yet turned" in lower:
        return "Needs M15 structure to turn in the setup direction first."
    if "no setup detected" in lower or "no clean" in lower or "no strategy produced a signal" in lower:
        return _STRATEGY_TRIGGER_HINTS.get(strategy_name, "Needs its core setup and execution filters to line up.")
    if "price" in lower and "channel boundary" in lower:
        return "Needs price to return to the active channel boundary with room for a clean RR."
    if "rr " in lower:
        return "Needs a better reward-to-risk profile before it is worth taking."
    if "daily trade cap" in lower or "max" in lower and "trade" in lower:
        return "Needs current strategy exposure to clear before another entry is allowed."
    if "no tick" in lower or "insufficient" in lower or "missing" in lower:
        return "Needs fresh market data on the required timeframes."
    return _STRATEGY_TRIGGER_HINTS.get(strategy_name, "Needs its core setup and execution filters to line up.")


def _build_strategy_eval_data(trader) -> Dict[str, Any]:
    positions = list(getattr(trader, "_cached_positions", None) or [])
    return {
        "m1_df": getattr(trader, "_candles", {}).get("M1"),
        "m5_df": getattr(trader, "_candles", {}).get("M5"),
        "m15_df": getattr(trader, "_candles", {}).get("M15"),
        "m30_df": getattr(trader, "_candles", {}).get("M30"),
        "h1_df": getattr(trader, "_candles", {}).get("H1"),
        "h4_df": getattr(trader, "_candles", {}).get("H4"),
        "d1_df": getattr(trader, "_candles", {}).get("D1"),
        "symbol": getattr(getattr(trader, "bridge", None), "current_symbol", cfg.SYMBOL),
        "tick": getattr(trader, "_last_tick", {}) or {},
        "zones": getattr(trader, "_zones", {}) or {},
        "indicators": getattr(trader, "_indicators", {}) or {},
        "tf_context": getattr(trader, "_tf_context", {}) or {},
        "account": getattr(trader, "_last_account", {}) or {},
        "positions": positions,
        "correlation": getattr(trader, "_correlation", {}) or {},
        "calendar": getattr(trader, "_calendar", {}) or {},
        "bias": getattr(trader, "_bias", {}) or {},
        "regime": getattr(trader, "_regime", {}) or {},
        "liquidity": getattr(trader, "_liquidity", {}) or {},
        "tick_snapshot": (trader.tick_proc.snapshot() if getattr(trader, "tick_proc", None) else {}) or {},
        "tick_pressure": getattr(trader, "_tick_pressure", {}) or {},
        "_risk_manager": getattr(trader, "risk", None),
        "now_utc": datetime.now(timezone.utc),
        "strategy_trade_counts": _strategy_trade_counts_for_trader(trader),
    }


def _build_strategy_guidance(trader) -> Dict[str, Any]:
    now_utc = datetime.now(timezone.utc)
    enabled = list((trader.strat_mgr.enabled_strategies() if trader and trader.strat_mgr else []) or [])
    evaluation = {"signal": {"signal": "NO_TRADE", "reason": "No strategy produced a signal"}, "strategy": "AUTO", "all_results": {}}
    if trader and trader.strat_mgr and enabled:
        try:
            evaluation = trader.strat_mgr.evaluate_all(_build_strategy_eval_data(trader), strategy_names=enabled)
        except Exception as exc:
            evaluation = {
                "signal": {"signal": "NO_TRADE", "reason": f"Strategy evaluation error: {exc}"},
                "strategy": "AUTO",
                "all_results": {},
            }

    results = dict(evaluation.get("all_results") or {})
    market_open = bool(getattr(trader, "enabled", False))
    engine_enabled = bool(getattr(trader, "enabled", False))
    tick = getattr(trader, "_last_tick", {}) or {}
    regime = getattr(trader, "_regime", {}) or {}
    bias = getattr(trader, "_bias", {}) or {}
    tick_pressure = getattr(trader, "_tick_pressure", {}) or {}
    calendar_state = getattr(trader, "_calendar", {}) or {}
    positions = list(getattr(trader, "_cached_positions", None) or [])

    strategy_rows = []
    for name in enabled:
        row = dict(results.get(name) or {})
        signal = str(row.get("signal") or "NO_TRADE").upper()
        gate_allowed = bool(row.get("gate_allowed")) if signal in {"BUY", "SELL"} else False
        status = "ready" if signal in {"BUY", "SELL"} and gate_allowed else "blocked" if signal in {"BUY", "SELL"} else "waiting"
        strategy_rows.append(
            {
                "name": name,
                "label": _strategy_label(name),
                "status": status,
                "signal": signal if signal in {"BUY", "SELL"} else "WAIT",
                "confidence": round(float(row.get("confidence") or 0.0), 3),
                "gate_allowed": gate_allowed,
                "gate_score": int(row.get("gate_score") or 0),
                "confirmation_count": int(row.get("confirmation_count") or 0),
                "rr": round(float(row.get("rr") or 0.0), 2),
                "pressure_score": round(float(row.get("pressure_score") or 0.0), 3),
                "arb_score": round(float(row.get("arb_score") or 0.0), 3),
                "reason": _clean_reason(row.get("reason") or row.get("gate_reason")),
                "trigger_hint": _reason_to_trigger_text(name, row.get("reason") or row.get("gate_reason")),
            }
        )

    strategy_rows.sort(
        key=lambda item: (
            1 if item["status"] == "ready" else 0,
            1 if item["status"] == "blocked" else 0,
            float(item.get("arb_score") or 0.0),
            float(item.get("confidence") or 0.0),
            float(item.get("rr") or 0.0),
        ),
        reverse=True,
    )

    ready_candidates = [row for row in strategy_rows if row["status"] == "ready"]
    blocked_candidates = [row for row in strategy_rows if row["status"] == "blocked"]
    lead = ready_candidates[0] if ready_candidates else blocked_candidates[0] if blocked_candidates else (strategy_rows[0] if strategy_rows else None)

    spread = float(tick.get("spread") or 0.0)
    bias_dir = str(bias.get("direction") or "NEUTRAL").upper()
    bias_conf = round(float(bias.get("confidence") or 0.0) * 100.0)
    regime_state = str(regime.get("state") or "UNKNOWN").upper()
    pressure_score = round(float(tick_pressure.get("pressure_score") or 0.0), 2)
    pressure_bias = str(tick_pressure.get("directional_bias") or "NEUTRAL").upper()

    if positions:
        action_now = f"Manage the {len(positions)} open trade{'s' if len(positions) != 1 else ''} first and avoid adding new exposure unless a clearly better setup appears."
    elif ready_candidates:
        action_now = f"Best live setup right now is {lead['label']} {lead['signal']}. Stay patient and only act if that setup is still valid at entry."
    elif blocked_candidates:
        action_now = f"Stay flat for now. The closest setup is {lead['label']} {lead['signal']}, but it is blocked by: {lead['reason']}"
    else:
        action_now = "Stay flat for now. No enabled strategy has a tradable setup yet."

    wait_seconds = None
    wait_reason = ""
    calendar_wait = _calendar_release_seconds(calendar_state, now_utc)
    if positions:
        wait_seconds = 300
        wait_reason = "Recheck soon while the current position is being managed."
    elif bool(calendar_state.get("blocked")) and calendar_wait is not None:
        wait_seconds = calendar_wait
        wait_reason = "Wait for the calendar blackout to clear."
    elif not engine_enabled:
        wait_reason = "The trading engine is off, so there is nothing to wait for until you start it."
    elif lead and "outside trade window" in str(lead.get("reason") or "").lower():
        wait_seconds = _seconds_until_next_trade_window(now_utc)
        wait_reason = "Wait for the next configured trading window."
    elif lead:
        wait_seconds = _STRATEGY_RECHECK_SECONDS.get(lead["name"], 300)
        wait_reason = f"Give the {lead['label']} setup time to refresh on its normal structure."
    elif enabled:
        wait_seconds = min(_STRATEGY_RECHECK_SECONDS.get(name, 300) for name in enabled)
        wait_reason = "Let the fastest enabled strategy build a fresh setup."

    if wait_seconds is not None:
        wait_advice = f"{wait_reason} Recheck in {_format_wait(wait_seconds)}."
    else:
        wait_advice = wait_reason or "Recheck when market conditions change."

    market_summary = (
        f"{regime_state.title()} regime, {bias_dir.title()} bias at {bias_conf}%, "
        f"spread {spread:.2f}, tick pressure {pressure_bias.title()} {pressure_score:+.2f}."
    )
    if bool(calendar_state.get("blocked")):
        market_summary += f" News block active: {str(calendar_state.get('reason') or '').strip()}."

    trigger_summary = (
        "A trade can trigger only when at least one enabled strategy prints BUY or SELL and its gate passes. "
        "The per-strategy watch list below shows the nearest blocker or the exact setup each strategy still needs."
    )

    lead_summary = None
    if lead:
        lead_summary = {
            "strategy": lead["name"],
            "label": lead["label"],
            "status": lead["status"],
            "signal": lead["signal"],
            "reason": lead["reason"],
            "trigger_hint": lead["trigger_hint"],
        }

    return {
        "generated_at": now_utc.isoformat(),
        "symbol": getattr(getattr(trader, "bridge", None), "current_symbol", cfg.SYMBOL),
        "market_summary": market_summary,
        "action_now": action_now,
        "wait_advice": wait_advice,
        "trigger_summary": trigger_summary,
        "lead_strategy": lead_summary,
        "strategies": strategy_rows,
    }


def _compact_candles(df, keep: int = 6) -> list:
    if df is None:
        return []
    rows = []
    try:
        tail = df.tail(keep)
    except Exception:
        return []
    for row in getattr(tail, "itertuples", lambda: [])():
        rows.append(
            {
                "time": str(getattr(row, "datetime", "")),
                "open": round(float(getattr(row, "open", 0.0) or 0.0), 3),
                "high": round(float(getattr(row, "high", 0.0) or 0.0), 3),
                "low": round(float(getattr(row, "low", 0.0) or 0.0), 3),
                "close": round(float(getattr(row, "close", 0.0) or 0.0), 3),
                "volume": round(float(getattr(row, "volume", 0.0) or 0.0), 0),
            }
        )
    return rows


def _build_market_question_snapshot(trader) -> Dict[str, Any]:
    guidance = _build_strategy_guidance(trader)
    tick = getattr(trader, "_last_tick", {}) or {}
    account = getattr(trader, "_last_account", {}) or {}
    indicators = getattr(trader, "_indicators", {}) or {}
    zones = getattr(trader, "_zones", {}) or {}
    regime = getattr(trader, "_regime", {}) or {}
    bias = getattr(trader, "_bias", {}) or {}
    liquidity = getattr(trader, "_liquidity", {}) or {}
    calendar_state = getattr(trader, "_calendar", {}) or {}
    correlation = getattr(trader, "_correlation", {}) or {}
    tick_pressure = getattr(trader, "_tick_pressure", {}) or {}
    positions = list(getattr(trader, "_cached_positions", None) or [])
    blockers = _merge_blocker_views(_get_cached_blockers(), _get_live_strategy_blockers(trader))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": getattr(getattr(trader, "bridge", None), "current_symbol", cfg.SYMBOL),
        "session": str(getattr(trader, "get_full_status", lambda **_: {})(
            include_closed_history=False,
            include_xgb=False,
            include_performance=False,
        ).get("session", "")),
        "engine_enabled": bool(getattr(trader, "enabled", False)),
        "guidance": guidance,
        "market": {
            "tick": {
                "bid": round(float(tick.get("bid", 0.0) or 0.0), 3),
                "ask": round(float(tick.get("ask", 0.0) or 0.0), 3),
                "spread": round(float(tick.get("spread", 0.0) or 0.0), 3),
                "time": str(tick.get("time", "")),
            },
            "account": {
                "balance": round(float(account.get("balance", 0.0) or 0.0), 2),
                "equity": round(float(account.get("equity", 0.0) or 0.0), 2),
                "free_margin": round(float(account.get("free_margin", 0.0) or 0.0), 2),
            },
            "regime": {
                "state": str(regime.get("state", "")),
                "direction": str(regime.get("direction", "")),
                "atr_ratio": round(float(regime.get("atr_ratio", 0.0) or 0.0), 4),
                "trade_allowed": bool(regime.get("trade_allowed", True)),
                "reasons": list(regime.get("reasons", []) or [])[:4],
            },
            "bias": {
                "direction": str(bias.get("direction", "")),
                "confidence": round(float(bias.get("confidence", 0.0) or 0.0), 4),
                "reasons": list(bias.get("reasons", []) or [])[:4],
            },
            "tick_pressure": {
                "ready": bool(tick_pressure.get("ready")),
                "directional_bias": str(tick_pressure.get("directional_bias", "")),
                "pressure_score": round(float(tick_pressure.get("pressure_score", 0.0) or 0.0), 4),
                "burst_rate": round(float(tick_pressure.get("burst_rate", 0.0) or 0.0), 2),
                "spread_mean": round(float(tick_pressure.get("spread_mean", 0.0) or 0.0), 3),
                "spread_std": round(float(tick_pressure.get("spread_std", 0.0) or 0.0), 4),
                "spread_shock": round(float(tick_pressure.get("spread_shock", 0.0) or 0.0), 3),
            },
            "calendar": calendar_state,
            "correlation": correlation,
            "indicators": {
                key: indicators.get(key)
                for key in ("ema9", "ema15", "ema20", "ema9_slope", "ema20_slope", "atr", "atr14", "atr_ratio", "rsi", "body_ratio", "range_10")
                if key in indicators
            },
            "zones": {
                "support": list((zones.get("support") or [])[:3]),
                "resistance": list((zones.get("resistance") or [])[:3]),
            },
            "liquidity": {
                "key_levels": liquidity.get("key_levels", {}),
                "recent_buy_sweep": liquidity.get("recent_buy_sweep"),
                "recent_sell_sweep": liquidity.get("recent_sell_sweep"),
            },
            "positions": positions[:5],
            "blockers": blockers,
            "recent_candles": {
                "M1": _compact_candles(getattr(trader, "_candles", {}).get("M1")),
                "M5": _compact_candles(getattr(trader, "_candles", {}).get("M5")),
                "M15": _compact_candles(getattr(trader, "_candles", {}).get("M15")),
                "H1": _compact_candles(getattr(trader, "_candles", {}).get("H1")),
            },
        },
    }


def _get_cached_xgb(indicators: dict, regime: dict, bias: dict, tick: dict) -> tuple:
    """Get XGB predictions with caching (model inference — expensive)."""
    global _xgb_cache, _xgb_cache_time
    now = time.monotonic()
    if _xgb_cache is not None and (now - _xgb_cache_time) < _XGB_CACHE_TTL:
        return _xgb_cache
    from engine.xgb_model import xgb_model
    if getattr(cfg, "XGB_BYPASS_ENABLED", False):
        xgb_info = xgb_model.get_feature_importance()
        xgb_pred = {"available": False, "reason": "XGBoost bypass enabled"}
        _xgb_cache = (xgb_info, xgb_pred)
        _xgb_cache_time = now
        return _xgb_cache
    if not getattr(cfg, "XGB_TRAINING_ENABLED", True):
        xgb_info = xgb_model.get_feature_importance()
        xgb_pred = {"available": False, "reason": "XGBoost training disabled"}
        _xgb_cache = (xgb_info, xgb_pred)
        _xgb_cache_time = now
        return _xgb_cache
    xgb_info = xgb_model.get_feature_importance()
    xgb_pred = {"available": False, "reason": "Model not trained yet (need 15+ trades with features)"}
    if xgb_model.is_trained:
        try:
            dummy = {"volume": 0.01, "sl_distance": 0.5}
            buy_prob = xgb_model.predict_win_prob({**dummy, "signal": "BUY"}, indicators, regime, bias, tick)
            sell_prob = xgb_model.predict_win_prob({**dummy, "signal": "SELL"}, indicators, regime, bias, tick)
            if buy_prob > 0.6 and buy_prob > sell_prob:
                sentiment, sentiment_color = "BULLISH", "green"
            elif sell_prob > 0.6 and sell_prob > buy_prob:
                sentiment, sentiment_color = "BEARISH", "red"
            else:
                sentiment, sentiment_color = "NEUTRAL", "yellow"
            xgb_pred = {"available": True, "buy_probability": buy_prob, "sell_probability": sell_prob,
                        "sentiment": sentiment, "sentiment_color": sentiment_color}
        except Exception as e:
            xgb_pred = {"available": False, "reason": f"Prediction error: {str(e)}"}
    _xgb_cache = (xgb_info, xgb_pred)
    _xgb_cache_time = now
    return _xgb_cache


def _get_cached_perf(trader) -> dict:
    """Get performance stats with caching."""
    global _perf_cache, _perf_cache_time
    now = time.monotonic()
    if _perf_cache is not None and (now - _perf_cache_time) < _PERF_CACHE_TTL:
        return _perf_cache
    if trader.perf:
        try:
            _perf_cache = trader.perf.get_stats()
        except Exception:
            _perf_cache = {}
    else:
        _perf_cache = {}
    _perf_cache_time = now
    return _perf_cache


def _get_cached_closed_history(trader, days: int = 30) -> list:
    """Get dashboard closed-history payload with caching."""
    global _closed_history_cache, _closed_history_cache_time
    now = time.monotonic()
    if _closed_history_cache is not None and (now - _closed_history_cache_time) < _CLOSED_HISTORY_CACHE_TTL:
        return _closed_history_cache

    db_closed = trader.trades.order_db.get_closed_orders_compact(days) if trader.trades and trader.trades.order_db else []
    mt5_map = {t.get("ticket"): t for t in (getattr(trader, "_mt5_closed_history", None) or []) if t.get("ticket")}
    closed_for_dash = []
    db_tickets = set()

    for order in db_closed:
        ticket = order.get("ticket")
        mt5_trade = mt5_map.get(ticket, {})
        exit_price = order.get("exit_price") or mt5_trade.get("exit_price") or 0
        close_time = order.get("close_time") or mt5_trade.get("close_time") or ""
        pnl = order.get("final_pnl") or 0
        features = order.get("features") or {}
        close_reason = _clean_close_reason(
            order.get("close_reason"),
            order.get("mt5_close_reason"),
            mt5_trade.get("comment"),
            order.get("strategy"),
        )
        closed_for_dash.append({
            "ticket": ticket,
            "direction": order.get("direction"),
            "volume": order.get("volume"),
            "entry_price": order.get("entry_price"),
            "exit_price": exit_price,
            "pnl": pnl,
            "won": pnl > 0,
            "close_time": close_time,
            "strategy": order.get("strategy", ""),
            "setup_type": features.get("_scalper_setup_type") or features.get("ai_manual_setup_type") or "",
            "close_reason": close_reason,
            "close_reason_category": order.get("close_reason_category") or "",
        })
        db_tickets.add(ticket)

    for trade in (getattr(trader, "_mt5_closed_history", None) or []):
        ticket = trade.get("ticket")
        if not ticket or ticket in db_tickets:
            continue
        pnl = trade.get("pnl", 0)
        closed_for_dash.append({
            "ticket": ticket,
            "direction": trade.get("direction", ""),
            "volume": trade.get("volume", 0),
            "entry_price": trade.get("entry_price", 0),
            "exit_price": trade.get("exit_price", 0),
            "pnl": pnl,
            "won": pnl > 0,
            "close_time": trade.get("close_time", ""),
            "close_reason": _clean_close_reason(trade.get("comment"), "MT5 history"),
            "close_reason_category": "",
        })

    closed_for_dash.sort(key=lambda item: item.get("close_time") or "", reverse=True)
    _closed_history_cache = closed_for_dash
    _closed_history_cache_time = now
    return _closed_history_cache


def _build_risk_dict(trader) -> dict:
    if trader.risk:
        return {
            "pnl": getattr(trader.risk, 'daily_pnl', 0),
            "trades": getattr(trader.risk, 'daily_trades', 0),
            "consecutive_losses": getattr(trader.risk, 'consecutive_losses', 0),
            "risk_multiplier": getattr(trader.risk, 'risk_multiplier', 1.0),
            "target_hit": getattr(trader.risk, 'target_hit', False),
            "loss_limit_hit": getattr(trader.risk, 'loss_limit_hit', False),
        }
    return {"pnl": 0, "trades": 0, "consecutive_losses": 0,
            "risk_multiplier": 1.0, "target_hit": False, "loss_limit_hit": False}


def _build_config_dict() -> dict:
    return {
        "MAX_POSITIONS": cfg.MAX_POSITIONS,
        "MAX_OPEN_TRADES": cfg.MAX_OPEN_TRADES,
        "MAX_TRADES_PER_DAY": cfg.MAX_TRADES_PER_DAY,
        "STOP_AFTER_CONSECUTIVE_LOSSES": cfg.STOP_AFTER_CONSECUTIVE_LOSSES,
        "MAX_RISK_PCT": cfg.MAX_RISK_PCT,
        "INTRADAY_RISK_PCT": cfg.INTRADAY_RISK_PCT,
        "SWING_RISK_PCT": cfg.SWING_RISK_PCT,
        "MAX_DRAWDOWN_PCT": cfg.MAX_DRAWDOWN_PCT,
        "MAX_LOT": cfg.MAX_LOT,
        "MIN_LOT": cfg.MIN_LOT,
        "DAILY_TARGET_DOLLARS": cfg.DAILY_TARGET_DOLLARS,
        "DAILY_LOSS_LIMIT_PCT": cfg.DAILY_LOSS_LIMIT_PCT,
        "DAILY_PROFIT_LOCK": cfg.DAILY_PROFIT_LOCK,
        "DAILY_PROFIT_LOCK_AFTER": cfg.DAILY_PROFIT_LOCK_AFTER,
        "DAILY_MAX_LOSS": cfg.DAILY_MAX_LOSS,
        "DAILY_MAX_LOSS_PCT_OF_ACCOUNT": cfg.DAILY_MAX_LOSS_PCT_OF_ACCOUNT,
        "MAX_CONSECUTIVE_LOSSES": cfg.MAX_CONSECUTIVE_LOSSES,
        "MIN_TRADE_COOLDOWN": cfg.MIN_TRADE_COOLDOWN,
        "LOSS_STREAK_PAUSE": cfg.LOSS_STREAK_PAUSE,
        "SESSION_MAX_TRADES": cfg.SESSION_MAX_TRADES,
        "SESSION_CENTRALIZED": cfg.SESSION_CENTRALIZED,
    }


def _build_sl_streak_guard_status() -> dict:
    from engine.sl_streak_guard import sl_streak_guard
    strategies = ["SWING_ENGINE", "INTRADAY_ENGINE", "SMC_CONFLUENCE", "SWEEP_SCALPER", "M15_SCALP_DEEP"]
    return {s: sl_streak_guard.status(s) for s in strategies
            if sl_streak_guard.status(s)["consecutive_sl_hits"] > 0}


def _build_ai_trade_advisor_status() -> dict:
    from engine.ai_trade_advisor import ai_trade_advisor_service
    service = ai_trade_advisor_service
    now = time.time()
    last_call_at = float(getattr(service, "_last_call_at", 0.0) or 0.0)
    cooldown_seconds = max(0.0, float(getattr(service, "min_seconds_between_calls", 0.0) or 0.0))
    seconds_since_last_call = max(0.0, now - last_call_at) if last_call_at > 0 else None
    cooldown_remaining = max(0.0, cooldown_seconds - (seconds_since_last_call or 0.0)) if last_call_at > 0 else 0.0
    has_api_key = bool(getattr(service, "api_key", ""))
    enabled = bool(getattr(service, "enabled", False))

    if not enabled:
        status = "disabled"
        reason = "AI advisor disabled"
    elif not has_api_key:
        status = "disabled"
        reason = "Missing NVIDIA API key"
    elif cooldown_remaining > 0.0:
        status = "cooldown"
        reason = "Cooling down between reviews"
    elif last_call_at <= 0.0:
        status = "ready"
        reason = "Waiting for a borderline live setup"
    else:
        status = "ready"
        reason = "Ready for the next borderline setup"

    return {
        "enabled": enabled,
        "has_api_key": has_api_key,
        "status": status,
        "reason": reason,
        "ready": enabled and has_api_key,
        "model": str(getattr(service, "model", "") or ""),
        "timeout_seconds": float(getattr(service, "timeout_seconds", 0.0) or 0.0),
        "min_seconds_between_calls": cooldown_seconds,
        "seconds_since_last_call": round(seconds_since_last_call, 1) if seconds_since_last_call is not None else None,
        "cooldown_remaining": round(cooldown_remaining, 1),
        "last_called_at": last_call_at if last_call_at > 0 else None,
    }


def _build_gate_config_dict() -> dict:
    return {
        "CALENDAR_BLOCKING_ENABLED": cfg.CALENDAR_BLOCKING_ENABLED,
        "CALENDAR_BLOCK_BEFORE_MINUTES": cfg.CALENDAR_BLOCK_BEFORE_MINUTES,
        "CALENDAR_BLOCK_AFTER_MINUTES": cfg.CALENDAR_BLOCK_AFTER_MINUTES,
        "ALL_GATES_OVERRIDE_ENABLED": cfg.ALL_GATES_OVERRIDE_ENABLED,
        "SPREAD_GATE_OVERRIDE_ENABLED": cfg.SPREAD_GATE_OVERRIDE_ENABLED,
        "COMPRESSION_GATE_OVERRIDE_ENABLED": cfg.COMPRESSION_GATE_OVERRIDE_ENABLED,
        "EXECUTION_GATE_OVERRIDE_ENABLED": cfg.EXECUTION_GATE_OVERRIDE_ENABLED,
        "TIME_GATE_OVERRIDE_ENABLED": cfg.TIME_GATE_OVERRIDE_ENABLED,
        "SPREAD_MEAN_GATE_OVERRIDE_ENABLED": cfg.SPREAD_MEAN_GATE_OVERRIDE_ENABLED,
        "SPREAD_VOLATILITY_GATE_OVERRIDE_ENABLED": cfg.SPREAD_VOLATILITY_GATE_OVERRIDE_ENABLED,
        "SPREAD_PERCENTILE_GATE_OVERRIDE_ENABLED": cfg.SPREAD_PERCENTILE_GATE_OVERRIDE_ENABLED,
        "SPREAD_DELTA_GATE_OVERRIDE_ENABLED": cfg.SPREAD_DELTA_GATE_OVERRIDE_ENABLED,
        "COMPRESSION_RANGE_GATE_OVERRIDE_ENABLED": cfg.COMPRESSION_RANGE_GATE_OVERRIDE_ENABLED,
        "ATR_RISING_GATE_OVERRIDE_ENABLED": cfg.ATR_RISING_GATE_OVERRIDE_ENABLED,
        "EXECUTION_QUALITY_GATE_OVERRIDE_ENABLED": cfg.EXECUTION_QUALITY_GATE_OVERRIDE_ENABLED,
        "TICK_DIRECTION_GATE_OVERRIDE_ENABLED": cfg.TICK_DIRECTION_GATE_OVERRIDE_ENABLED,
        "POST_SIGNAL_SPREAD_GATE_OVERRIDE_ENABLED": cfg.POST_SIGNAL_SPREAD_GATE_OVERRIDE_ENABLED,
        "TIER1_SPREAD_RECHECK_MAX_DELTA": cfg.TIER1_SPREAD_RECHECK_MAX_DELTA,
        "TIER1_EARLY_FAIL_EXIT_ENABLED": cfg.TIER1_EARLY_FAIL_EXIT_ENABLED,
        "TIER1_EARLY_FAIL_TICKS": cfg.TIER1_EARLY_FAIL_TICKS,
        "TIER1_BLOCK_SPIKE_ENTRY": cfg.TIER1_BLOCK_SPIKE_ENTRY,
        "TIER1_MAX_ENTRY_DELAY_MS": cfg.TIER1_MAX_ENTRY_DELAY_MS,
        "SWING_PEAK_CAPTURE_ENABLED": cfg.SWING_PEAK_CAPTURE_ENABLED,
        "SMC_SPREAD_MEAN_MAX": cfg.SMC_SPREAD_MEAN_MAX,
        "SMC_SPREAD_STD_MAX": cfg.SMC_SPREAD_STD_MAX,
        "SMC_SPREAD_PERCENTILE_MAX": cfg.SMC_SPREAD_PERCENTILE_MAX,
        "SMC_CURRENT_SPREAD_DELTA_MAX": cfg.SMC_CURRENT_SPREAD_DELTA_MAX,
        "SCALPER_SPREAD_MEAN_MAX": cfg.SCALPER_SPREAD_MEAN_MAX,
        "SCALPER_SPREAD_STD_MAX": cfg.SCALPER_SPREAD_STD_MAX,
        "SCALPER_SPREAD_PERCENTILE_MAX": cfg.SCALPER_SPREAD_PERCENTILE_MAX,
        "SCALPER_CURRENT_SPREAD_DELTA_MAX": cfg.SCALPER_CURRENT_SPREAD_DELTA_MAX,
        "COMPRESSION_RANGE_LOOKBACK": cfg.COMPRESSION_RANGE_LOOKBACK,
        "COMPRESSION_ATR_MULTIPLIER": cfg.COMPRESSION_ATR_MULTIPLIER,
        "SMC_THRESHOLD_WITH_TREND": cfg.SMC_THRESHOLD_WITH_TREND,
        "SMC_THRESHOLD_COUNTER": cfg.SMC_THRESHOLD_COUNTER,
        "SMC_THRESHOLD_COUNTER_MAX": cfg.SMC_THRESHOLD_COUNTER_MAX,
        "SMC_MIN_RR": cfg.SMC_MIN_RR,
        "SWING_PEAK_CAPTURE_TRIGGER_R": cfg.SWING_PEAK_CAPTURE_TRIGGER_R,
        "SWING_PEAK_CAPTURE_GIVEBACK_PCT": cfg.SWING_PEAK_CAPTURE_GIVEBACK_PCT,
        "SWING_PEAK_CAPTURE_MIN_LOCK_R": cfg.SWING_PEAK_CAPTURE_MIN_LOCK_R,
        "SMC_TIMEFRAME_EMA_SLOPE_MIN": cfg.SMC_TIMEFRAME_EMA_SLOPE_MIN,
        "SMC_LTF_TICK_CONFLICT_LONG_MAX": cfg.SMC_LTF_TICK_CONFLICT_LONG_MAX,
        "SMC_LTF_TICK_CONFLICT_SHORT_MIN": cfg.SMC_LTF_TICK_CONFLICT_SHORT_MIN,
        "SCALPER_QUALITY_THRESHOLD": cfg.SCALPER_QUALITY_THRESHOLD,
        "SCALPER_ATR_MIN": cfg.SCALPER_ATR_MIN,
        "SCALPER_ATR_MAX": cfg.SCALPER_ATR_MAX,
        "SCALPER_EMA20_SLOPE_MIN": cfg.SCALPER_EMA20_SLOPE_MIN,
        "SCALPER_BODY_RATIO_MIN": cfg.SCALPER_BODY_RATIO_MIN,
        "SCALPER_TICK_DIR_THRESHOLD": cfg.SCALPER_TICK_DIR_THRESHOLD,
        "SCALPER_SWEEP_LOOKBACK": cfg.SCALPER_SWEEP_LOOKBACK,
        "SCALPER_SWEEP_TOLERANCE": cfg.SCALPER_SWEEP_TOLERANCE,
        "SCALPER_MAX_TRADES_SESSION": cfg.SCALPER_MAX_TRADES_SESSION,
        "SCALPER_LEVEL_COOLDOWN": cfg.SCALPER_LEVEL_COOLDOWN,
        "M15_SR_ENABLED": cfg.M15_SR_ENABLED,
        "M15_SR_AS_CONTEXT_ONLY": cfg.M15_SR_AS_CONTEXT_ONLY,
        "M15_SR_LOOKBACK_HOURS": cfg.M15_SR_LOOKBACK_HOURS,
        "M15_SR_FALLBACK_LOOKBACK_HOURS": cfg.M15_SR_FALLBACK_LOOKBACK_HOURS,
        "M15_SR_EXTENDED_LOOKBACK_HOURS": cfg.M15_SR_EXTENDED_LOOKBACK_HOURS,
        "M15_SR_MIN_TOUCHES": cfg.M15_SR_MIN_TOUCHES,
        "M15_SR_ZONE_ATR_MULT": cfg.M15_SR_ZONE_ATR_MULT,
        "M15_SR_ENTRY_BUFFER_ATR": cfg.M15_SR_ENTRY_BUFFER_ATR,
        "M15_SR_SL_BUFFER_ATR": cfg.M15_SR_SL_BUFFER_ATR,
        "M15_SR_MIN_RR": cfg.M15_SR_MIN_RR,
        "M15_SR_DEFAULT_RR": cfg.M15_SR_DEFAULT_RR,
        "M15_SR_MAX_SPREAD": cfg.M15_SR_MAX_SPREAD,
        "M15_SR_MAX_TRADES_PER_DAY": cfg.M15_SR_MAX_TRADES_PER_DAY,
        "M15_SR_COOLDOWN_CANDLES": cfg.M15_SR_COOLDOWN_CANDLES,
        "M15_SR_BLOCK_AFTER_SPIKE_CANDLES": cfg.M15_SR_BLOCK_AFTER_SPIKE_CANDLES,
        "M15_SR_SPIKE_RANGE_ATR_MULT": cfg.M15_SR_SPIKE_RANGE_ATR_MULT,
        "M15_SR_SPIKE_VOLUME_MULT": cfg.M15_SR_SPIKE_VOLUME_MULT,
        "M15_SR_MAX_DISTANCE_FROM_ZONE_ATR": cfg.M15_SR_MAX_DISTANCE_FROM_ZONE_ATR,
        "M15_SR_REQUIRE_CANDLE_CONFIRMATION": cfg.M15_SR_REQUIRE_CANDLE_CONFIRMATION,
        "TRADE_SCORE_ENABLED": cfg.TRADE_SCORE_ENABLED,
        "TRADE_SCORE_MIN": cfg.TRADE_SCORE_MIN,
        "TRADE_SCORE_PREMIUM": cfg.TRADE_SCORE_PREMIUM,
        "TRADE_SCORE_RELAXED_MIN": cfg.TRADE_SCORE_RELAXED_MIN,
        "TRADE_SCORE_STRICT_AFTER_LOSS": cfg.TRADE_SCORE_STRICT_AFTER_LOSS,
        "CONFIRMATION_3_OF_4_ENABLED": cfg.CONFIRMATION_3_OF_4_ENABLED,
        "BLOCK_WEAK_COUNTER_TREND": cfg.BLOCK_WEAK_COUNTER_TREND,
        "COUNTER_TREND_MIN_SCORE": cfg.COUNTER_TREND_MIN_SCORE,
        "COUNTER_TREND_REQUIRE_M15_ZONE": cfg.COUNTER_TREND_REQUIRE_M15_ZONE,
        "COUNTER_TREND_REQUIRE_SWEEP": cfg.COUNTER_TREND_REQUIRE_SWEEP,
        "COUNTER_TREND_REQUIRE_CANDLE_CONFIRMATION": cfg.COUNTER_TREND_REQUIRE_CANDLE_CONFIRMATION,
        "AUTO_RELAX_ENABLED": cfg.AUTO_RELAX_ENABLED,
        "AUTO_RELAX_AFTER_MINUTES": cfg.AUTO_RELAX_AFTER_MINUTES,
        "AUTO_RELAX_MIN_SCORE": cfg.AUTO_RELAX_MIN_SCORE,
        "AUTO_RELAX_RR": cfg.AUTO_RELAX_RR,
        "AUTO_RELAX_ONLY_GOOD_SESSION": cfg.AUTO_RELAX_ONLY_GOOD_SESSION,
        "AUTO_RELAX_BLOCK_AFTER_LOSS": cfg.AUTO_RELAX_BLOCK_AFTER_LOSS,
        "AUTO_RELAX_BLOCK_DRAWDOWN": cfg.AUTO_RELAX_BLOCK_DRAWDOWN,
        "AUTO_RELAX_REQUIRE_STABLE_SPREAD": cfg.AUTO_RELAX_REQUIRE_STABLE_SPREAD,
        "XGB_BYPASS_ENABLED": cfg.XGB_BYPASS_ENABLED,
        "XGB_TRAINING_ENABLED": cfg.XGB_TRAINING_ENABLED,
        "XGB_BLOCKING_ENABLED": cfg.XGB_BLOCKING_ENABLED,
        "XGB_BLOCK_THRESHOLD": cfg.XGB_BLOCK_THRESHOLD,
        "XGB_BLEND_CONFIDENCE_ENABLED": cfg.XGB_BLEND_CONFIDENCE_ENABLED,
        "XGB_LOG_CONFIDENCE_ENABLED": cfg.XGB_LOG_CONFIDENCE_ENABLED,
        "AI_TRADE_ADVISOR_ENABLED": cfg.AI_TRADE_ADVISOR_ENABLED,
        "AI_TRADE_ADVISOR_TIMEOUT_SECONDS": cfg.AI_TRADE_ADVISOR_TIMEOUT_SECONDS,
        "AI_TRADE_ADVISOR_MIN_SECONDS_BETWEEN_CALLS": cfg.AI_TRADE_ADVISOR_MIN_SECONDS_BETWEEN_CALLS,
        "AI_TRADE_ADVISOR_MAX_CONFIDENCE": cfg.AI_TRADE_ADVISOR_MAX_CONFIDENCE,
        "AI_TRADE_ADVISOR_SCORE_BUFFER": cfg.AI_TRADE_ADVISOR_SCORE_BUFFER,
        "AI_TRADE_ADVISOR_REJECT_CONFIDENCE_MIN": cfg.AI_TRADE_ADVISOR_REJECT_CONFIDENCE_MIN,
        "AI_TRADE_ADVISOR_FAIL_OPEN": cfg.AI_TRADE_ADVISOR_FAIL_OPEN,
        "TRADE_WINDOW_LONDON_START": cfg.TRADE_WINDOW_LONDON_START,
        "TRADE_WINDOW_LONDON_END": cfg.TRADE_WINDOW_LONDON_END,
        "TRADE_WINDOW_OVERLAP_START": cfg.TRADE_WINDOW_OVERLAP_START,
        "TRADE_WINDOW_OVERLAP_END": cfg.TRADE_WINDOW_OVERLAP_END,
        "MARKET_TICK_POLL_INTERVAL": cfg.MARKET_TICK_POLL_INTERVAL,
        "DASHBOARD_WS_PUSH_INTERVAL": cfg.DASHBOARD_WS_PUSH_INTERVAL,
    }


def _build_gate_defaults_dict() -> dict:
    import config as cfg
    defaults = cfg.get_runtime_defaults()
    return {key: defaults[key] for key in _build_gate_config_dict() if key in defaults}


def _build_profile_dict() -> dict:
    import config as cfg
    return cfg.list_runtime_profiles()


def _get_live_config_baseline() -> dict:
    """Return editable live-profile values without adaptive runtime overrides."""
    import config as cfg
    baseline = cfg.get_runtime_defaults()
    try:
        profile_values = cfg.get_profile_values(cfg.get_active_profile("live"), role="live")
    except Exception:
        profile_values = {}
    baseline.update(profile_values)
    return baseline


def _build_calendar_detail() -> dict:
    """Return full calendar state including all loaded events and blackout config."""
    from engine.calendar import calendar as _cal
    import config as _cfg
    state = _cal.check()
    return {
        **state,
        "events": _cal.get_events(),
        "block_before_minutes": int(getattr(_cfg, "CALENDAR_BLOCK_BEFORE_MINUTES", 30) or 30),
        "block_after_minutes": int(getattr(_cfg, "CALENDAR_BLOCK_AFTER_MINUTES", 15) or 15),
    }


def _build_tick_data(trader) -> dict:
    """Build the tick payload dict from trader state. Pure memory reads."""
    import config as cfg
    from engine.session_filter import get_session, is_market_open
    started_at_utc = getattr(trader, "_started_at_utc", None)
    last_restart_utc = started_at_utc.isoformat() if started_at_utc else None
    last_restart_ist = started_at_utc.astimezone(_IST).strftime("%Y-%m-%d %H:%M:%S IST") if started_at_utc else None
    return {
        "app_version": cfg.APP_VERSION,
        "enabled": trader.enabled,
        "mt5_connected": trader._mt5_connected,
        "tick": trader._last_tick,
        "account": trader._last_account,
        "session": get_session(),
        "market_open": is_market_open(),
        "symbol": trader.bridge.current_symbol if trader.bridge else cfg.SYMBOL,
        "strategy": trader.strat_mgr.active_name if trader.strat_mgr else "AUTO",
        "mt5_positions": getattr(trader, '_cached_positions', None) or [],
        "mt5_floating_pnl": getattr(trader, '_cached_floating_pnl', None) or {"total": 0, "count": 0},
        "mt5_today_pnl": getattr(trader, '_mt5_today_pnl', None) or {},
        "risk": _build_risk_dict(trader),
        "indicators": trader._indicators or {},
        "stats": trader._stats,
        "config_version": _config_version,
    }


def _refresh_tick_cache(trader) -> str:
    """Serialize tick data once per market tick or engine cycle. Returns cached JSON string."""
    global _ws_latest_tick, _ws_latest_cycle
    tick_seq = getattr(trader, "_tick_seq", 0)
    pos_ver = getattr(trader, "_positions_version", 0)
    pnl_cents = round((getattr(trader, "_cached_floating_pnl", None) or {}).get("total", 0) * 100)
    marker = (tick_seq, pos_ver, pnl_cents)
    if marker == _ws_latest_cycle:
        return _ws_latest_tick
    _ws_latest_tick = _fast_json(_build_tick_data(trader))
    _ws_latest_cycle = marker
    return _ws_latest_tick


# --- WebSocket: server pushes tick data every engine cycle ---
@router.websocket("/ws")
async def ws_tick(ws: WebSocket):
    import config as cfg
    await ws.accept()
    _ws_clients.add(ws)

    async def _reader():
        """Drain incoming frames so we detect disconnect."""
        try:
            while True:
                await ws.receive_text()
        except (WebSocketDisconnect, Exception):
            pass

    reader_task = asyncio.create_task(_reader())
    last_tick_seq = -1
    last_positions_version = -1
    last_pnl_cents = None
    try:
        while not reader_task.done():
            trader = _auto_trader_instance
            if trader is not None:
                tick_seq = getattr(trader, "_tick_seq", 0)
                pos_ver = getattr(trader, "_positions_version", 0)
                pnl_cents = round((getattr(trader, "_cached_floating_pnl", None) or {}).get("total", 0) * 100)
                if tick_seq != last_tick_seq or pos_ver != last_positions_version or pnl_cents != last_pnl_cents:
                    msg = _refresh_tick_cache(trader)
                    await ws.send_text(msg)
                    last_tick_seq = tick_seq
                    last_positions_version = pos_ver
                    last_pnl_cents = pnl_cents
            await asyncio.sleep(max(0.02, float(getattr(cfg, 'DASHBOARD_WS_PUSH_INTERVAL', 0.03) or 0.03)))
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        reader_task.cancel()
        _ws_clients.discard(ws)


# Keep /tick as HTTP fallback (e.g. curl, other clients)
@router.get("/tick")
async def get_tick():
    """HTTP fallback for tick data. Prefer /ws WebSocket for real-time."""
    trader = get_auto_trader()
    return _RawResponse(content=_refresh_tick_cache(trader), media_type="application/json")


def _build_status_sync() -> dict:
    """Build lightweight status dict — runs in thread pool to avoid blocking event loop."""
    trader = get_auto_trader()
    import config as cfg

    status = trader.get_full_status(
        include_closed_history=False,
        include_xgb=False,
        include_performance=False,
    )
    status.update({
        "app_version": cfg.APP_VERSION,
        "symbol": trader.bridge.current_symbol if trader.bridge else cfg.SYMBOL,
        "strategy": trader.strat_mgr.active_name if trader.strat_mgr else "AUTO",
        "config_version": _config_version,
    })
    # Strip legacy group aliases from available list
    if "strategies" in status and "available" in status["strategies"]:
        pass  # no groups to strip anymore
    for persistent_key in (
        "tier1_enabled", "session_override_enabled", "daily_target_enabled",
        "daily_target", "available_symbols", "strategies", "risk_config", "gate_config"
    ):
        status.pop(persistent_key, None)
    for heavy_key in (
        "closed_history",
        "performance",
        "xgb",
        "xgb_live_prediction",
        "live_blockers",
    ):
        status.pop(heavy_key, None)
    status["mt5_positions"] = getattr(trader, '_cached_positions', []) or []
    status["mt5_floating_pnl"] = getattr(trader, '_cached_floating_pnl', {"total": 0, "count": 0})
    status["mt5_today_pnl"] = getattr(trader, '_mt5_today_pnl', {})
    status["risk"] = _build_risk_dict(trader)
    status["tick_pressure"] = getattr(trader, "_tick_pressure", None) or {"ready": False}
    status["ai_trade_advisor"] = _build_ai_trade_advisor_status()
    status["deployment"] = _get_deployment_status(trader)
    return convert_numpy_types(status)


@router.get("/status")
async def get_status(include_heavy: bool = False):
    """Default lightweight status. Use include_heavy=true for legacy all-in-one payload."""
    global _status_cache, _status_cache_time
    now = time.monotonic()
    if _status_cache is not None and (now - _status_cache_time) < _STATUS_CACHE_TTL:
        if not include_heavy:
            return _status_cache
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _build_status_sync)
    _status_cache = result
    _status_cache_time = time.monotonic()
    if include_heavy:
        heavy = await get_status_heavy()
        merged = dict(result)
        merged.update(heavy)
        return merged
    return result


@router.get("/status/closed-history")
async def get_closed_history():
    """Closed trade history panel."""
    global _closed_history_cache, _closed_history_cache_time
    now = time.monotonic()
    if _closed_history_cache is not None and (now - _closed_history_cache_time) < _CLOSED_HISTORY_CACHE_TTL:
        return {"closed_history": _closed_history_cache}
    trader = get_auto_trader()
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: _get_cached_closed_history(trader)),
            timeout=6.0,
        )
    except asyncio.TimeoutError:
        result = _closed_history_cache or []
    return {"closed_history": result}


@router.get("/status/performance")
async def get_performance_panel():
    """Performance stats panel."""
    global _perf_cache, _perf_cache_time
    now = time.monotonic()
    if _perf_cache is not None and (now - _perf_cache_time) < _PERF_CACHE_TTL:
        return {"performance": _perf_cache}
    trader = get_auto_trader()
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: _get_cached_perf(trader)),
            timeout=6.0,
        )
    except asyncio.TimeoutError:
        result = _perf_cache or {}
    return {"performance": result}


@router.get("/status/xgb")
async def get_xgb_panel():
    """XGBoost model info and live prediction panel."""
    global _xgb_cache, _xgb_cache_time
    now = time.monotonic()
    if _xgb_cache is not None and (now - _xgb_cache_time) < _XGB_CACHE_TTL:
        xgb_info, xgb_pred = _xgb_cache
        return {"xgb": xgb_info, "xgb_live_prediction": xgb_pred}
    trader = get_auto_trader()
    loop = asyncio.get_event_loop()
    try:
        xgb_info, xgb_pred = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: _get_cached_xgb(
                trader._indicators if isinstance(trader._indicators, dict) else {},
                trader._regime if isinstance(trader._regime, dict) else {},
                trader._bias if isinstance(trader._bias, dict) else {},
                trader._last_tick or {},
            )),
            timeout=6.0,
        )
    except asyncio.TimeoutError:
        if _xgb_cache is not None:
            xgb_info, xgb_pred = _xgb_cache
        else:
            xgb_info, xgb_pred = {}, {"available": False, "reason": "timeout"}
    return convert_numpy_types({"xgb": xgb_info, "xgb_live_prediction": xgb_pred})


@router.get("/status/blockers")
async def get_blockers_panel():
    """Live trade blockers panel."""
    global _blockers_cache, _blockers_cache_time
    now = time.monotonic()
    if _blockers_cache is not None and (now - _blockers_cache_time) < _BLOCKERS_CACHE_TTL:
        history = _blockers_cache
    else:
        loop = asyncio.get_event_loop()
        try:
            history = await asyncio.wait_for(
                loop.run_in_executor(None, _get_cached_blockers),
                timeout=6.0,
            )
        except asyncio.TimeoutError:
            history = _blockers_cache or {"latest_by_strategy": {}, "top_reasons": [], "recent_skipped_count": 0}
    trader = _auto_trader_instance
    live = _get_live_strategy_blockers(trader) if trader is not None else {"latest_by_strategy": {}, "top_reasons": [], "recent_skipped_count": 0}
    return {"live_blockers": _merge_blocker_views(history, live)}


@router.get("/strategy-guidance")
async def get_strategy_guidance():
    """Deterministic plain-language market guidance based on enabled strategies."""
    trader = get_auto_trader()
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: _build_strategy_guidance(trader)),
            timeout=6.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Strategy guidance request timed out")
    return {"_guidance": convert_numpy_types(result)}


@router.post("/market-question")
async def ask_market_question(request: Request):
    """Answer a custom user question using live market and strategy context."""
    trader = get_auto_trader()
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    question = str((body or {}).get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    loop = asyncio.get_event_loop()

    def _run():
        snapshot = _build_market_question_snapshot(trader)
        result = market_context_qa_service.answer_question(question, snapshot)
        if "guidance" not in result and isinstance(snapshot, dict):
            result["guidance"] = snapshot.get("guidance") or {}
        return result

    try:
        result = await asyncio.wait_for(loop.run_in_executor(None, _run), timeout=30.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Market question request timed out")
    return convert_numpy_types(result)


@router.get("/status/ai-reviews")
async def get_ai_reviews_panel(limit: int = 12):
    """Recent AI trade advisor requests and responses for the dashboard."""
    safe_limit = max(1, min(int(limit or 12), 30))
    global _ai_reviews_cache, _ai_reviews_cache_time
    now = time.monotonic()
    if _ai_reviews_cache is not None and (now - _ai_reviews_cache_time) < _AI_REVIEWS_CACHE_TTL:
        return {"ai_trade_reviews": _ai_reviews_cache[:safe_limit]}
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: _get_cached_ai_reviews(safe_limit)),
            timeout=6.0,
        )
    except asyncio.TimeoutError:
        result = (_ai_reviews_cache or [])[:safe_limit]
    return {"ai_trade_reviews": result}


@router.get("/status/heavy")
async def get_status_heavy():
    """Aggregate all heavy panels concurrently. Kept for backward compatibility."""
    global _heavy_status_cache, _heavy_status_cache_time
    now = time.monotonic()
    if _heavy_status_cache is not None and (now - _heavy_status_cache_time) < _HEAVY_STATUS_CACHE_TTL:
        return _heavy_status_cache
    ch, perf, xgb_resp, bl, ai_reviews = await asyncio.gather(
        get_closed_history(),
        get_performance_panel(),
        get_xgb_panel(),
        get_blockers_panel(),
        get_ai_reviews_panel(),
    )
    result = {**ch, **perf, **xgb_resp, **bl, **ai_reviews}
    _heavy_status_cache = result
    _heavy_status_cache_time = time.monotonic()
    return result

@router.get("/calendar")
async def get_calendar():
    """Full calendar state: blocked status, all events, blackout window config."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _build_calendar_detail)
    return convert_numpy_types(result)


@router.get("/logs")
async def get_logs():
    """Get recent trading logs."""
    trader = get_auto_trader()
    return {"logs": list(trader._log)[-200:]}


_LOG_LINE_PATTERN = re.compile(
    r'^\[(\d{2}:\d{2}:\d{2}\.\d{3}) UTC \| (\d{2}:\d{2}:\d{2}\.\d{3}) IST(?: \| (\d{4}-\d{2}-\d{2}))?\]'
    r'\[([A-Z_]+)\](?:\[([\d.]+)\])? (.*)$'
)

# Shared cache for parsed trader.log entries.
_parsed_logs_cache: list = []
_parsed_logs_times: list = []
_parsed_logs_file_size: int = 0
_parsed_logs_file_mtime: float = 0.0


def _trader_log_path() -> Path:
    return Path(__file__).resolve().parent.parent / "trader.log"


def _coerce_log_datetime(ist_date: str, ist_time: str):
    date_part = str(ist_date or "").strip()
    time_part = str(ist_time or "").strip()
    if not date_part or not time_part:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(f"{date_part} {time_part}", fmt).replace(tzinfo=_IST)
        except Exception:
            continue
    return None


def _read_parsed_logs():
    global _parsed_logs_cache, _parsed_logs_times
    global _parsed_logs_file_size, _parsed_logs_file_mtime
    log_path = _trader_log_path()
    if not log_path.exists():
        _parsed_logs_cache = []
        _parsed_logs_times = []
        _parsed_logs_file_size = 0
        _parsed_logs_file_mtime = 0.0
        return [], []

    stat = log_path.stat()
    if (
        _parsed_logs_cache
        and stat.st_size == _parsed_logs_file_size
        and stat.st_mtime == _parsed_logs_file_mtime
    ):
        return _parsed_logs_cache, _parsed_logs_times

    entries = []
    times = []
    raw = log_path.read_bytes().decode("utf-8", errors="ignore")
    for line in raw.splitlines():
        match = _LOG_LINE_PATTERN.match(line.strip())
        if not match:
            continue
        utc_t, ist_t, ist_date, tag, pr, msg = match.groups()
        dt_ist = _coerce_log_datetime(ist_date, ist_t)
        if dt_ist is None:
            continue
        entry = {
            "time": ist_t,
            "time_ist": ist_t,
            "ist_date": ist_date,
            "tag": tag,
            "msg": msg,
            "price": float(pr) if pr else None,
            "ts_ist": dt_ist.isoformat(timespec="milliseconds"),
        }
        entries.append(entry)
        times.append(dt_ist)

    _parsed_logs_cache = entries
    _parsed_logs_times = times
    _parsed_logs_file_size = stat.st_size
    _parsed_logs_file_mtime = stat.st_mtime
    return _parsed_logs_cache, _parsed_logs_times


def _parse_log_cursor(value: str | None):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_IST)
        return dt.astimezone(_IST)
    except Exception:
        return None


@router.get("/logs/recent")
async def get_recent_logs(
    window_minutes: int = Query(60, ge=1, le=24 * 60),
    before: str | None = Query(None),
):
    """Return the latest available hour of logs, or older one-hour windows when paginating backward."""
    try:
        entries, times = _read_parsed_logs()
    except Exception as exc:
        return {"logs": [], "count": 0, "total_count": 0, "has_more_before": False, "error": str(exc)}

    total_count = len(entries)
    if total_count == 0:
        return {"logs": [], "count": 0, "total_count": 0, "has_more_before": False}

    end_idx = total_count
    cursor_dt = _parse_log_cursor(before)
    if before:
        if cursor_dt is None:
            raise HTTPException(status_code=400, detail="Invalid 'before' cursor")
        end_idx = bisect_left(times, cursor_dt)

    if end_idx <= 0:
        latest_ts = entries[-1]["ts_ist"]
        return {
            "logs": [],
            "count": 0,
            "total_count": total_count,
            "has_more_before": False,
            "latest_ts_ist": latest_ts,
            "oldest_ts_ist": None,
            "newest_ts_ist": None,
        }

    anchor_dt = times[end_idx - 1]
    start_dt = anchor_dt - timedelta(minutes=int(window_minutes))
    start_idx = bisect_left(times, start_dt)
    payload = entries[start_idx:end_idx]

    return {
        "logs": payload,
        "count": len(payload),
        "total_count": total_count,
        "has_more_before": start_idx > 0,
        "latest_ts_ist": entries[-1]["ts_ist"],
        "oldest_ts_ist": payload[0]["ts_ist"] if payload else None,
        "newest_ts_ist": payload[-1]["ts_ist"] if payload else None,
        "window_minutes": int(window_minutes),
    }


@router.get("/logs/today")
async def get_logs_today(limit: int = Query(0, ge=0, le=1000)):
    """Return today's IST log entries with optional tail limit."""
    try:
        entries, _times = _read_parsed_logs()
    except Exception as exc:
        return {"logs": [], "count": 0, "error": str(exc)}

    today_ist_date = datetime.now(_IST).strftime("%Y-%m-%d")
    today_entries = [entry for entry in entries if entry.get("ist_date") == today_ist_date]
    payload = today_entries[-limit:] if limit else today_entries
    return {"logs": payload, "count": len(today_entries)}


@router.get("/debug/manifest")
async def get_debug_manifest(
    request: Request,
    recent_days: int = Query(7, ge=1, le=30),
):
    """Inspect which runtime artifacts are available for a portable debug export."""
    _require_debug_bundle_access(request)
    base_dir = Path(__file__).resolve().parent.parent
    return build_debug_manifest(base_dir, recent_days=recent_days)


@router.get("/debug/export")
async def download_debug_bundle(
    request: Request,
    recent_days: int = Query(7, ge=1, le=30),
):
    """Build and download a portable runtime debug bundle."""
    _require_debug_bundle_access(request)
    base_dir = Path(__file__).resolve().parent.parent
    bundle = build_debug_bundle(base_dir, recent_days=recent_days)
    return FileResponse(
        bundle["bundle_path"],
        media_type="application/zip",
        filename=bundle["bundle_name"],
    )

@router.get("/trades")
async def get_trades():
    """Get current trades status."""
    trader = get_auto_trader()
    trades_status = trader.trades.status if trader.trades else {}
    return convert_numpy_types(trades_status)


@router.post("/trades/{ticket}/extend-timeout")
async def extend_trade_timeout(ticket: int, request: Request):
    """Manually extend a trade's time-based exit window."""
    trader = get_auto_trader()
    if not trader.trades:
        raise HTTPException(status_code=503, detail="Trade manager unavailable")

    body = await request.json()
    minutes = body.get("minutes")
    seconds = body.get("seconds")
    extend_seconds = 0
    try:
        if seconds is not None:
            extend_seconds = int(seconds)
        elif minutes is not None:
            extend_seconds = int(round(float(minutes) * 60))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid extension value")

    if extend_seconds <= 0:
        raise HTTPException(status_code=400, detail="Extension must be greater than zero")

    updated = trader.trades.extend_trade_timeout(ticket, extend_seconds)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Open trade #{ticket} not found")

    trader.log(
        "API",
        f"Trade #{ticket} timeout extended by {extend_seconds}s to {updated['timeout_seconds']}s",
    )
    _invalidate_status_cache()
    return convert_numpy_types({
        "success": True,
        "ticket": ticket,
        "extended_by_seconds": extend_seconds,
        **updated,
    })


@router.post("/trades/{ticket}/manual-exit")
async def manual_exit_trade(ticket: int, request: Request):
    """Manually close an open trade from the dashboard."""
    trader = get_auto_trader()
    try:
        body = await request.json()
    except Exception:
        body = {}
    reason = str(body.get("reason") or "Manual dashboard exit").strip() or "Manual dashboard exit"

    if trader.trades:
        updated = trader.trades.manually_exit_trade(ticket, reason)
        if updated is not None:
            if not updated.get("success"):
                trader.log("API", f"Manual close failed for #{ticket}: {updated.get('error') or 'unknown error'}")
                raise HTTPException(status_code=502, detail=updated.get("error") or f"Failed to close trade #{ticket}")
            trader.log("API", f"Trade #{ticket} manually closed: {reason}")
            _invalidate_status_cache()
            return convert_numpy_types(updated)

    bridge = getattr(trader, "bridge", None)
    if not bridge:
        raise HTTPException(status_code=503, detail="Trade execution bridge unavailable")

    result = bridge.close_trade(ticket)
    if not result.get("success"):
        trader.log("API", f"Manual close failed for untracked #{ticket}: {result.get('error') or 'unknown error'}")
        raise HTTPException(status_code=404, detail=result.get("error") or f"Open trade #{ticket} not found")

    trader.log("API", f"Untracked trade #{ticket} manually closed: {reason}")
    _invalidate_status_cache()
    return convert_numpy_types({
        "success": True,
        "ticket": ticket,
        "reason": reason,
        "close_reason_category": "manual_exit",
        "tracked": False,
        **result,
    })


@router.post("/trades/{ticket}/modify-sltp")
async def modify_trade_sltp(ticket: int, request: Request):
    """Modify SL and/or TP of an open trade from the dashboard."""
    trader = get_auto_trader()
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    bridge = getattr(trader, "bridge", None)
    if not bridge:
        raise HTTPException(status_code=503, detail="Trade execution bridge unavailable")

    sl_raw = body.get("sl")
    tp_raw = body.get("tp")
    if sl_raw is None or tp_raw is None:
        raise HTTPException(status_code=400, detail="Both sl and tp are required")

    try:
        new_sl = round(float(sl_raw), 2)
        new_tp = round(float(tp_raw), 2)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="sl and tp must be numeric")

    if new_sl <= 0 or new_tp <= 0:
        raise HTTPException(status_code=400, detail="sl and tp must be positive")

    result = bridge.modify_trade(ticket, new_sl, new_tp)
    if not result.get("success"):
        raise HTTPException(status_code=502, detail=result.get("error") or f"Failed to modify trade #{ticket}")

    trader.log("API", f"Trade #{ticket} modified: SL={new_sl:.2f} TP={new_tp:.2f}")
    _invalidate_status_cache()
    return convert_numpy_types({"success": True, "ticket": ticket, "sl": new_sl, "tp": new_tp})


@router.post("/ai/manual-idea")
async def generate_manual_ai_trade_idea(request: Request):
    """Generate an on-demand AI trade idea for manual review on the dashboard."""
    from engine.ai_manual_trade_ideas import ai_manual_trade_idea_service, build_manual_trade_context
    from engine.session_filter import get_session, is_market_open

    trader = get_auto_trader()
    try:
        body = await request.json()
    except Exception:
        body = {}

    timeframe = str(body.get("timeframe") or "M5").strip().upper()
    limit = max(60, min(int(body.get("limit") or 120), 240))

    def _run():
        df = trader.bridge.fetch_candles(timeframe)
        if df is None or df.empty:
            return {
                "status": "error",
                "provider": "nvidia",
                "model": getattr(ai_manual_trade_idea_service, "model", ""),
                "reasoning": f"No candle data available for {timeframe}",
                "actionable": False,
            }

        tick = trader._last_tick or trader.bridge.get_tick() or {}
        account = trader._last_account or trader.bridge.get_account() or {}
        indicators = dict(getattr(trader, "_indicators", {}) or {})
        regime = dict(getattr(trader, "_regime", {}) or {})
        bias = dict(getattr(trader, "_bias", {}) or {})
        liquidity = dict(getattr(trader, "_liquidity", {}) or {})
        zones = dict(getattr(trader, "_zones", {}) or {})
        tick_pressure = dict(getattr(trader, "_tick_pressure", {}) or {})
        xgb_info, xgb_pred = _get_cached_xgb(indicators, regime, bias, tick)
        _ = xgb_info  # feature importance not needed here

        candles = []
        for row in df.tail(limit).itertuples():
            candles.append(
                {
                    "open": float(row.open),
                    "high": float(row.high),
                    "low": float(row.low),
                    "close": float(row.close),
                    "volume": float(row.volume),
                }
            )

        context = build_manual_trade_context(
            symbol=trader.bridge.current_symbol if trader.bridge else cfg.SYMBOL,
            timeframe=timeframe,
            tick=tick,
            session=get_session(),
            market_open=is_market_open(),
            indicators=indicators,
            regime=regime,
            bias=bias,
            liquidity=liquidity,
            tick_pressure=tick_pressure,
            xgb_live_prediction=xgb_pred,
            account=account,
            candles=candles,
            support_zones=(zones.get("support") or []),
            resistance_zones=(zones.get("resistance") or []),
        )
        result = ai_manual_trade_idea_service.generate_trade_idea(context, tick)
        result["symbol"] = trader.bridge.current_symbol if trader.bridge else cfg.SYMBOL
        result["timeframe"] = timeframe
        result["session"] = get_session()
        result["generated_at_ts"] = time.time()
        result["market_open"] = bool(is_market_open())
        return result

    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(loop.run_in_executor(None, _run), timeout=15.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="AI trade idea request timed out")

    if result.get("status") == "ready":
        result = _store_manual_ai_idea(result)
    return convert_numpy_types(result)


@router.post("/ai/manual-idea/{idea_id}/market")
async def execute_manual_ai_trade_market(idea_id: str):
    """Execute a previously generated AI trade idea as a market order."""
    trader = get_auto_trader()
    idea = _get_manual_ai_idea_or_404(idea_id)
    if not idea.get("actionable"):
        raise HTTPException(status_code=400, detail="AI trade idea is not actionable")

    bridge = getattr(trader, "bridge", None)
    if not bridge:
        raise HTTPException(status_code=503, detail="Trade execution bridge unavailable")
    if not trader.trades:
        raise HTTPException(status_code=503, detail="Trade manager unavailable")

    action = str(idea.get("action") or "").upper()
    sl = float(idea.get("sl") or 0.0)
    tp = float(idea.get("tp") or 0.0)
    tick = bridge.get_tick() or {}
    account = bridge.get_account() or {}
    entry_now = float(tick.get("ask") if action == "BUY" else tick.get("bid") or 0.0)
    sl_distance = abs(entry_now - sl)
    if action not in {"BUY", "SELL"} or sl_distance <= 0 or sl <= 0 or tp <= 0:
        raise HTTPException(status_code=400, detail="AI trade idea has invalid execution levels")

    open_count = trader.trades.open_count if trader.trades else len(bridge.get_my_positions())
    allowed, reason = trader.risk.can_trade(account, open_count)
    if not allowed:
        raise HTTPException(status_code=400, detail=reason)

    high_conf = float(idea.get("confidence") or 0.0) >= 0.80
    lot = trader.risk.calculate_lot(account, sl_distance, high_conf=high_conf)
    lot = max(cfg.MIN_LOT, min(cfg.MAX_LOT, lot))

    comment = f"FT_AI_MKT_{str(idea.get('timeframe') or 'M5')[:4]}"
    result = bridge.open_trade(action, lot, sl, tp, comment=comment)
    if not result.get("success"):
        raise HTTPException(status_code=502, detail=result.get("error") or "Failed to place market order")

    live_tick_metrics = trader.tick_proc.snapshot() or {}
    trader.trades.register_trade(
        result["ticket"],
        action,
        lot,
        result.get("price") or entry_now,
        sl,
        tp,
        sl_distance,
        strategy="AI_MANUAL",
        confidence=float(idea.get("confidence") or 0.0),
        reason=str(idea.get("reasoning") or "AI manual trade"),
        features={
            "ai_manual": True,
            "ai_manual_idea_id": idea["idea_id"],
            "ai_manual_mode": "market",
            "ai_manual_symbol": idea.get("symbol"),
            "ai_manual_timeframe": idea.get("timeframe"),
            "ai_manual_setup_type": idea.get("setup_type"),
            "ai_manual_pending_order_type": idea.get("pending_order_type"),
            "ai_manual_reasoning": idea.get("reasoning"),
            "ai_manual_invalidation": idea.get("invalidation"),
            "ai_manual_checklist": idea.get("checklist"),
            "ai_manual_suggested_entry": idea.get("entry"),
            "ai_manual_provider": idea.get("provider"),
            "ai_manual_model": idea.get("model"),
            "entry_tick_velocity": live_tick_metrics.get("velocity", 0),
            "entry_tick_count": live_tick_metrics.get("tick_count", 0),
            "entry_tick_pressure_score": getattr(trader, "_tick_pressure", {}).get("pressure_score"),
            "entry_tick_pressure_bias": getattr(trader, "_tick_pressure", {}).get("directional_bias"),
            "entry_tick_burst_rate": getattr(trader, "_tick_pressure", {}).get("burst_rate"),
            "regime": getattr(trader, "_regime", {}).get("state", ""),
            "bias_conf": getattr(trader, "_bias", {}).get("confidence", 0),
            "session": idea.get("session"),
            "profile_name": "manual",
        },
        session_type=str(idea.get("session") or ""),
        market_phase=str((getattr(trader, "_regime", {}) or {}).get("state") or ""),
    )
    trader.risk.record_trade_opened()
    trader.log(
        "API",
        f"AI manual market trade {action} {lot:.2f} lot @ {float(result.get('price') or entry_now):.2f} | SL:{sl:.2f} TP:{tp:.2f}",
    )
    _invalidate_status_cache(include_slow=True)
    return convert_numpy_types(
        {
            "success": True,
            "mode": "market",
            "idea_id": idea["idea_id"],
            "ticket": result["ticket"],
            "price": result.get("price") or entry_now,
            "volume": lot,
            "direction": action,
            "sl": sl,
            "tp": tp,
            "reasoning": idea.get("reasoning"),
        }
    )


@router.post("/ai/manual-idea/{idea_id}/pending")
async def execute_manual_ai_trade_pending(idea_id: str):
    """Place a previously generated AI trade idea as a pending order."""
    trader = get_auto_trader()
    idea = _get_manual_ai_idea_or_404(idea_id)
    if not idea.get("actionable"):
        raise HTTPException(status_code=400, detail="AI trade idea is not actionable")

    bridge = getattr(trader, "bridge", None)
    if not bridge:
        raise HTTPException(status_code=503, detail="Trade execution bridge unavailable")

    action = str(idea.get("action") or "").upper()
    entry = float(idea.get("entry") or 0.0)
    sl = float(idea.get("sl") or 0.0)
    tp = float(idea.get("tp") or 0.0)
    sl_distance = abs(entry - sl)
    if action not in {"BUY", "SELL"} or entry <= 0 or sl_distance <= 0 or sl <= 0 or tp <= 0:
        raise HTTPException(status_code=400, detail="AI trade idea has invalid pending-order levels")

    account = bridge.get_account() or {}
    open_count = trader.trades.open_count if trader.trades else len(bridge.get_my_positions())
    allowed, reason = trader.risk.can_trade(account, open_count)
    if not allowed:
        raise HTTPException(status_code=400, detail=reason)

    high_conf = float(idea.get("confidence") or 0.0) >= 0.80
    lot = trader.risk.calculate_lot(account, sl_distance, high_conf=high_conf)
    lot = max(cfg.MIN_LOT, min(cfg.MAX_LOT, lot))

    comment = f"FT_AI_PND_{str(idea.get('timeframe') or 'M5')[:4]}"
    result = bridge.open_pending_trade(action, lot, entry, sl, tp, comment=comment)
    if not result.get("success"):
        raise HTTPException(status_code=502, detail=result.get("error") or "Failed to place pending order")

    trader.log(
        "API",
        f"AI manual pending order {result.get('order_type')} {lot:.2f} lot @ {entry:.2f} | SL:{sl:.2f} TP:{tp:.2f}",
    )
    _invalidate_status_cache(include_slow=True)
    return convert_numpy_types(
        {
            "success": True,
            "mode": "pending",
            "idea_id": idea["idea_id"],
            "ticket": result["ticket"],
            "price": entry,
            "volume": lot,
            "direction": action,
            "order_type": result.get("order_type") or idea.get("pending_order_type"),
            "sl": sl,
            "tp": tp,
            "reasoning": idea.get("reasoning"),
        }
    )

@router.get("/performance")
async def get_performance():
    """Get performance statistics."""
    trader = get_auto_trader()
    perf_stats = trader.perf.get_stats()
    return convert_numpy_types(perf_stats)


@router.get("/orders/review")
async def get_orders_review(days: int = 30):
    """Get segmented trade outcome review from orders.db."""
    trader = get_auto_trader()
    if not trader.trades or not trader.trades.order_db:
        return {"error": "Order database unavailable"}
    return convert_numpy_types(trader.trades.order_db.get_trade_outcome_review(days))



@router.get("/strategy-config")
async def get_strategy_configs():
    """Return all per-strategy configs plus their locked/disabled metadata."""
    configs = _scfg_store.get_all()
    meta = _scfg_store.get_all_meta()
    return {"strategies": configs, "meta": meta}


@router.post("/strategy-config/{name}")
async def save_strategy_config(name: str, request: Request):
    """Save per-strategy config and hot-apply CFG_MAP keys to live cfg."""
    trader = get_auto_trader()
    name = name.upper()
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    try:
        updated = _scfg_store.save(name, body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    trader.log("API", f"Strategy config saved: {name}")
    _bump_config_version()
    _invalidate_status_cache(include_slow=True)
    return {"name": name, "config": updated}


@router.post("/strategy-config/{name}/reset")
async def reset_strategy_config(name: str):
    """Reset a strategy config to defaults."""
    trader = get_auto_trader()
    name = name.upper()
    try:
        defaults = _scfg_store.reset(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    trader.log("API", f"Strategy config reset to defaults: {name}")
    _bump_config_version()
    _invalidate_status_cache(include_slow=True)
    return {"name": name, "config": defaults}


@router.get("/strategies", response_class=HTMLResponse)
async def get_strategies_page():
    """Serve the strategy management page."""
    base_dir = Path(__file__).resolve().parent.parent
    page_path = base_dir / "strategies.html"
    if page_path.exists():
        return HTMLResponse(page_path.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="strategies.html not found")

@router.get("/config")
async def get_config():
    """Get current configuration."""
    trader = get_auto_trader()
    baseline = _get_live_config_baseline()

    def editable_value(key: str):
        if key in getattr(cfg, "_ADAPTIVE_MANAGED_KEYS", set()):
            return baseline.get(key, getattr(cfg, key))
        return getattr(cfg, key)

    strats = trader.strat_mgr.status()
    return {
        "app_version": cfg.APP_VERSION,
        "symbol": trader.bridge.current_symbol if trader.bridge else cfg.SYMBOL,
        "strategy": trader.strat_mgr.active_name,
        "strategies": strats,
        "config_version": _config_version,
        "profiles": _build_profile_dict(),
        "risk_pct": editable_value("MAX_RISK_PCT"),
        "daily_target": editable_value("DAILY_TARGET_DOLLARS"),
        "tier1_enabled": cfg.TIER1_ENABLED,
        "session_override_enabled": cfg.SESSION_OVERRIDE_ENABLED,
        "daily_target_enabled": cfg.DAILY_TARGET_ENABLED,
        "gate_config": {
            key: (baseline.get(key, value) if key in getattr(cfg, "_ADAPTIVE_MANAGED_KEYS", set()) else value)
            for key, value in _build_gate_config_dict().items()
        },
        "gate_defaults": _build_gate_defaults_dict(),
        "xgb_config": {
            "XGB_BYPASS_ENABLED": editable_value("XGB_BYPASS_ENABLED"),
            "XGB_TRAINING_ENABLED": editable_value("XGB_TRAINING_ENABLED"),
            "XGB_BLOCKING_ENABLED": editable_value("XGB_BLOCKING_ENABLED"),
            "XGB_BLOCK_THRESHOLD": editable_value("XGB_BLOCK_THRESHOLD"),
            "XGB_BLEND_CONFIDENCE_ENABLED": editable_value("XGB_BLEND_CONFIDENCE_ENABLED"),
            "XGB_LOG_CONFIDENCE_ENABLED": editable_value("XGB_LOG_CONFIDENCE_ENABLED"),
        },
        # Risk configuration parameters - nested under risk_config for dashboard
        "risk_config": {
            "MAX_OPEN_TRADES": editable_value("MAX_OPEN_TRADES"),
            "MAX_TRADES_PER_DAY": editable_value("MAX_TRADES_PER_DAY"),
            "STOP_AFTER_CONSECUTIVE_LOSSES": editable_value("STOP_AFTER_CONSECUTIVE_LOSSES"),
            "MAX_POSITIONS": editable_value("MAX_POSITIONS"),
            "MAX_RISK_PCT": editable_value("MAX_RISK_PCT"),
            "INTRADAY_RISK_PCT": editable_value("INTRADAY_RISK_PCT"),
            "SWING_RISK_PCT": editable_value("SWING_RISK_PCT"),
            "MAX_DRAWDOWN_PCT": editable_value("MAX_DRAWDOWN_PCT"),
            "MAX_LOT": editable_value("MAX_LOT"),
            "MIN_LOT": editable_value("MIN_LOT"),
            "DAILY_TARGET_DOLLARS": editable_value("DAILY_TARGET_DOLLARS"),
            "DAILY_LOSS_LIMIT_PCT": editable_value("DAILY_LOSS_LIMIT_PCT"),
            "DAILY_PROFIT_LOCK": editable_value("DAILY_PROFIT_LOCK"),
            "DAILY_PROFIT_LOCK_AFTER": editable_value("DAILY_PROFIT_LOCK_AFTER"),
            "DAILY_MAX_LOSS": editable_value("DAILY_MAX_LOSS"),
            "DAILY_MAX_LOSS_PCT_OF_ACCOUNT": editable_value("DAILY_MAX_LOSS_PCT_OF_ACCOUNT"),
            "MAX_CONSECUTIVE_LOSSES": editable_value("MAX_CONSECUTIVE_LOSSES"),
            "MIN_TRADE_COOLDOWN": editable_value("MIN_TRADE_COOLDOWN"),
            "LOSS_STREAK_PAUSE": editable_value("LOSS_STREAK_PAUSE"),
            "SESSION_MAX_TRADES": editable_value("SESSION_MAX_TRADES"),
        }
    }

@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """Get trading dashboard HTML."""
    base_dir = Path(__file__).resolve().parent.parent
    dash_path = base_dir / "dashboard.html"
    if dash_path.exists():
        return HTMLResponse(dash_path.read_text(encoding="utf-8"))
    else:
        raise HTTPException(status_code=404, detail="Dashboard not found")

@router.post("/risk/reset")
async def reset_risk_gates(request: Request):
    """
    Reset blocking risk gates so trading can resume immediately.
    Body: {"hard": false}  — soft reset clears pause/streak/cooldown.
           {"hard": true}  — also clears daily loss flag, trade counter, daily P&L.
    """
    trader = get_auto_trader()
    try:
        body = await request.json()
    except Exception:
        body = {}
    hard = bool(body.get("hard", False))
    result = trader.risk.reset_gates(hard=hard)
    trader.log("API", f"Risk gates reset (hard={hard}): {result['cleared']}")
    _invalidate_status_cache()
    return convert_numpy_types({"success": True, **result})


@router.post("/start")
async def start_trading():
    """Enable trading."""
    trader = get_auto_trader()
    trader.enabled = True
    cfg.TRADING_ENABLED = True
    try:
        cfg.save_runtime_config()
    except Exception:
        pass
    trader.log("API", "Trading ENABLED")
    _invalidate_status_cache()
    return {"enabled": True, "strategy": trader.strat_mgr.active_name if trader.strat_mgr else "AUTO"}

@router.post("/stop")
async def stop_trading():
    """Disable trading."""
    trader = get_auto_trader()
    trader.enabled = False
    cfg.TRADING_ENABLED = False
    try:
        cfg.save_runtime_config()
    except Exception:
        pass
    trader.log("API", "Trading DISABLED")
    _invalidate_status_cache()
    return {"enabled": False}

@router.post("/emergency")
async def emergency_stop():
    """Emergency stop - disable trading and close all positions."""
    trader = get_auto_trader()
    trader.enabled = False
    cfg.TRADING_ENABLED = False
    try:
        cfg.save_runtime_config()
    except Exception:
        pass
    if trader.trades:
        trader.trades.close_all()
    trader.log("API", "EMERGENCY STOP")
    _invalidate_status_cache(include_slow=True)
    return {"enabled": False}

@router.post("/strategy/{strategy_name}")
async def set_strategy(strategy_name: str):
    """Set active trading strategy."""
    trader = get_auto_trader()
    name = strategy_name.upper()
    if trader.strat_mgr.set_active(name):
        cfg.DEFAULT_STRATEGY = name
        try:
            cfg.save_runtime_config()
        except Exception:
            pass
        trader.log("API", f"Strategy -> {name}")
        _bump_config_version()
        _invalidate_status_cache()
        return trader.strat_mgr.status()
    else:
        raise HTTPException(
            status_code=400, 
            detail={"error": f"Unknown: {name}", "available": trader.strat_mgr.available}
        )


@router.post("/strategies/select")
async def set_strategies_selection(request: Request):
    """Set AUTO or an explicit strategy selection."""
    trader = get_auto_trader()
    body = await request.json()
    raw_selected = body.get("selected") or []
    if isinstance(raw_selected, str):
        raw_selected = [raw_selected]
    if not isinstance(raw_selected, list):
        raise HTTPException(status_code=400, detail="selected must be a list of strategy names")
    selected = [str(item or "").strip().upper() for item in raw_selected if str(item or "").strip()]
    if not trader.strat_mgr.set_selection(selected):
        raise HTTPException(
            status_code=400,
            detail={"error": f"Unknown strategy in selection: {selected}", "available": trader.strat_mgr.available},
        )
    cfg.DEFAULT_STRATEGY = "AUTO" if trader.strat_mgr.is_auto else trader.strat_mgr.enabled_strategies()
    try:
        cfg.save_runtime_config()
    except Exception:
        pass
    trader.log("API", f"Strategy selection -> {trader.strat_mgr.active_name}")
    _bump_config_version()
    _invalidate_status_cache()
    result = trader.strat_mgr.status()
    return result

@router.post("/symbol/{symbol}")
async def set_symbol(symbol: str):
    """Set trading symbol."""
    trader = get_auto_trader()
    import config as cfg
    
    symbol = symbol.upper()
    if symbol not in cfg.AVAILABLE_SYMBOLS:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Unknown symbol: {symbol}", "available": cfg.AVAILABLE_SYMBOLS}
        )
    
    if trader.bridge.set_symbol(symbol):
        # Clear all cached data when symbol changes
        trader._candles.clear()
        trader._candle_counts.clear()
        trader._indicators.clear()
        trader._zones.clear()
        trader._regime.clear()
        trader._bias.clear()
        trader._liquidity.clear()
        trader._last_tick = {}
        
        # Force immediate data refresh for new symbol
        try:
            for tf in ["M1", "M5", "M15", "H1", "H4", "D1"]:
                df = trader.bridge.fetch_candles(tf)
                if not df.empty:
                    trader._candles[tf] = df
                    trader._candle_counts[tf] = len(df)
            trader._recompute_all()
        except Exception as e:
            trader.log("API", f"Symbol data refresh failed: {e}")
        
        trader.log("API", f"Symbol -> {symbol} (cache cleared, data refreshed)")
        _bump_config_version()
        _invalidate_status_cache(include_slow=True)
        return {"symbol": symbol, "available": cfg.AVAILABLE_SYMBOLS}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to set {symbol}")

@router.post("/config")
async def update_config(request: Request):
    """Update trading configuration."""
    trader = get_auto_trader()
    body = await request.json()
    
    _RISK_KEYS = {
        'MAX_POSITIONS': int, 'MAX_RISK_PCT': float, 'MAX_DRAWDOWN_PCT': float,
        'INTRADAY_RISK_PCT': float, 'SWING_RISK_PCT': float,
        'MAX_LOT': float, 'MIN_LOT': float, 'DAILY_TARGET_DOLLARS': float,
        'DAILY_LOSS_LIMIT_PCT': float, 'MAX_CONSECUTIVE_LOSSES': int,
        'MIN_TRADE_COOLDOWN': float, 'LOSS_STREAK_PAUSE': int,
        'SESSION_MAX_TRADES': int, 'MAX_OPEN_TRADES': int, 'MAX_TRADES_PER_DAY': int,
        'STOP_AFTER_CONSECUTIVE_LOSSES': int, 'DAILY_PROFIT_LOCK_AFTER': float,
        'DAILY_MAX_LOSS': float, 'DAILY_MAX_LOSS_PCT_OF_ACCOUNT': float
    }
    _GATE_FLOAT_KEYS = {
        'SMC_SPREAD_MEAN_MAX', 'SMC_SPREAD_STD_MAX', 'SMC_SPREAD_PERCENTILE_MAX',
        'SMC_CURRENT_SPREAD_DELTA_MAX', 'SCALPER_SPREAD_MEAN_MAX',
        'SCALPER_SPREAD_STD_MAX', 'SCALPER_SPREAD_PERCENTILE_MAX',
        'SCALPER_CURRENT_SPREAD_DELTA_MAX', 'COMPRESSION_ATR_MULTIPLIER',
        'SMC_THRESHOLD_WITH_TREND', 'SMC_THRESHOLD_COUNTER', 'SMC_THRESHOLD_COUNTER_MAX',
        'SMC_MIN_RR', 'SWING_PEAK_CAPTURE_TRIGGER_R', 'SWING_PEAK_CAPTURE_GIVEBACK_PCT',
        'SWING_PEAK_CAPTURE_MIN_LOCK_R', 'SMC_TIMEFRAME_EMA_SLOPE_MIN', 'SMC_LTF_TICK_CONFLICT_LONG_MAX',
        'SMC_LTF_TICK_CONFLICT_SHORT_MIN', 'SCALPER_QUALITY_THRESHOLD',
        'SCALPER_ATR_MIN', 'SCALPER_ATR_MAX', 'SCALPER_EMA20_SLOPE_MIN',
        'SCALPER_BODY_RATIO_MIN', 'SCALPER_TICK_DIR_THRESHOLD',
        'M15_SR_ZONE_ATR_MULT', 'M15_SR_ENTRY_BUFFER_ATR', 'M15_SR_SL_BUFFER_ATR',
        'M15_SR_MIN_RR', 'M15_SR_DEFAULT_RR', 'M15_SR_MAX_SPREAD',
        'M15_SR_SPIKE_RANGE_ATR_MULT', 'M15_SR_SPIKE_VOLUME_MULT',
        'M15_SR_MAX_DISTANCE_FROM_ZONE_ATR',
        'TIER1_SPREAD_RECHECK_MAX_DELTA', 'XGB_BLOCK_THRESHOLD',
        'TRADE_SCORE_MIN', 'TRADE_SCORE_PREMIUM', 'TRADE_SCORE_RELAXED_MIN',
        'TRADE_SCORE_STRICT_AFTER_LOSS', 'COUNTER_TREND_MIN_SCORE',
        'AUTO_RELAX_AFTER_MINUTES', 'AUTO_RELAX_MIN_SCORE', 'AUTO_RELAX_RR',
        'AI_TRADE_ADVISOR_TIMEOUT_SECONDS', 'AI_TRADE_ADVISOR_MIN_SECONDS_BETWEEN_CALLS',
        'AI_TRADE_ADVISOR_MAX_CONFIDENCE', 'AI_TRADE_ADVISOR_REJECT_CONFIDENCE_MIN',
        'SCALPER_SWEEP_TOLERANCE', 'MARKET_TICK_POLL_INTERVAL',
        'DASHBOARD_WS_PUSH_INTERVAL'
    }
    _GATE_INT_KEYS = {
        'COMPRESSION_RANGE_LOOKBACK', 'SCALPER_SWEEP_LOOKBACK',
        'SCALPER_MAX_TRADES_SESSION', 'SCALPER_LEVEL_COOLDOWN',
        'CALENDAR_BLOCK_BEFORE_MINUTES', 'CALENDAR_BLOCK_AFTER_MINUTES',
        'M15_SR_LOOKBACK_HOURS', 'M15_SR_FALLBACK_LOOKBACK_HOURS',
        'M15_SR_EXTENDED_LOOKBACK_HOURS',
        'M15_SR_MIN_TOUCHES', 'M15_SR_MAX_TRADES_PER_DAY',
        'M15_SR_COOLDOWN_CANDLES', 'M15_SR_BLOCK_AFTER_SPIKE_CANDLES',
        'AI_TRADE_ADVISOR_SCORE_BUFFER',
        'TIER1_EARLY_FAIL_TICKS', 'TIER1_MAX_ENTRY_DELAY_MS',
        'TRADE_WINDOW_LONDON_START', 'TRADE_WINDOW_LONDON_END',
        'TRADE_WINDOW_OVERLAP_START', 'TRADE_WINDOW_OVERLAP_END'
    }
    _GATE_BOOL_KEYS = {
        'CALENDAR_BLOCKING_ENABLED',
        'ALL_GATES_OVERRIDE_ENABLED', 'SPREAD_GATE_OVERRIDE_ENABLED',
        'COMPRESSION_GATE_OVERRIDE_ENABLED', 'EXECUTION_GATE_OVERRIDE_ENABLED',
        'TIME_GATE_OVERRIDE_ENABLED', 'SPREAD_MEAN_GATE_OVERRIDE_ENABLED',
        'SPREAD_VOLATILITY_GATE_OVERRIDE_ENABLED', 'SPREAD_PERCENTILE_GATE_OVERRIDE_ENABLED',
        'SPREAD_DELTA_GATE_OVERRIDE_ENABLED', 'COMPRESSION_RANGE_GATE_OVERRIDE_ENABLED',
        'ATR_RISING_GATE_OVERRIDE_ENABLED', 'EXECUTION_QUALITY_GATE_OVERRIDE_ENABLED',
        'M15_SR_ENABLED', 'M15_SR_AS_CONTEXT_ONLY', 'M15_SR_REQUIRE_CANDLE_CONFIRMATION',
        'SWING_PEAK_CAPTURE_ENABLED', 'TICK_DIRECTION_GATE_OVERRIDE_ENABLED',
        'POST_SIGNAL_SPREAD_GATE_OVERRIDE_ENABLED', 'DAILY_PROFIT_LOCK',
        'SESSION_CENTRALIZED', 'TRADE_ASIA_SESSION', 'TRADE_CLOSED_SESSION',
        'TRADE_ROLLOVER', 'PREMIUM_ONLY_ASIA', 'TIER1_EARLY_FAIL_EXIT_ENABLED',
        'TIER1_BLOCK_SPIKE_ENTRY', 'TRADE_SCORE_ENABLED', 'CONFIRMATION_3_OF_4_ENABLED',
        'BLOCK_WEAK_COUNTER_TREND', 'COUNTER_TREND_REQUIRE_M15_ZONE',
        'COUNTER_TREND_REQUIRE_SWEEP', 'COUNTER_TREND_REQUIRE_CANDLE_CONFIRMATION',
        'AI_TRADE_ADVISOR_ENABLED', 'AI_TRADE_ADVISOR_FAIL_OPEN',
        'AUTO_RELAX_ENABLED', 'AUTO_RELAX_ONLY_GOOD_SESSION',
        'AUTO_RELAX_BLOCK_AFTER_LOSS', 'AUTO_RELAX_BLOCK_DRAWDOWN',
        'AUTO_RELAX_REQUIRE_STABLE_SPREAD', 'XGB_BYPASS_ENABLED',
        'XGB_TRAINING_ENABLED', 'XGB_BLOCKING_ENABLED',
        'XGB_BLEND_CONFIDENCE_ENABLED', 'XGB_LOG_CONFIDENCE_ENABLED'
    }
    
    updated = {}
    for k, v in body.items():
        if k in _RISK_KEYS:
            val = _RISK_KEYS[k](v)
            setattr(cfg, k, val)
            updated[k] = val
        elif k in _GATE_FLOAT_KEYS:
            val = max(0.0, float(v))
            if k.endswith('_PERCENTILE_MAX'):
                val = min(1.0, val)
            setattr(cfg, k, val)
            updated[k] = val
        elif k in _GATE_INT_KEYS:
            val = max(0, int(v))
            setattr(cfg, k, val)
            updated[k] = val
        elif k in _GATE_BOOL_KEYS:
            val = bool(v)
            setattr(cfg, k, val)
            updated[k] = val
    
    if 'TIER1_ENABLED' in body:
        cfg.TIER1_ENABLED = bool(body['TIER1_ENABLED'])
        updated['TIER1_ENABLED'] = cfg.TIER1_ENABLED
    
    if 'SESSION_OVERRIDE_ENABLED' in body:
        cfg.SESSION_OVERRIDE_ENABLED = bool(body['SESSION_OVERRIDE_ENABLED'])
        updated['SESSION_OVERRIDE_ENABLED'] = cfg.SESSION_OVERRIDE_ENABLED
    
    if 'DAILY_TARGET_ENABLED' in body:
        cfg.DAILY_TARGET_ENABLED = bool(body['DAILY_TARGET_ENABLED'])
        updated['DAILY_TARGET_ENABLED'] = cfg.DAILY_TARGET_ENABLED
    
    if updated:
        try:
            cfg.save_runtime_config()
            adaptive_updates = {
                key: value for key, value in updated.items()
                if key in getattr(cfg, "_ADAPTIVE_MANAGED_KEYS", set())
            }
            if adaptive_updates:
                cfg.update_runtime_profile(
                    cfg.get_active_profile("live"),
                    values=adaptive_updates,
                )
        except Exception as e:
            trader.log("API", f"Config persistence failed: {e}")
        try:
            from engine import adaptive_params
            adaptive_params._capture_base()
        except Exception as e:
            trader.log("API", f"Adaptive base refresh failed: {e}")
        _bump_config_version()
        if any(k in updated for k in ('LOSS_STREAK_PAUSE', 'MAX_CONSECUTIVE_LOSSES')):
            trader.risk._loss_streak_pause_until = 0.0
            trader.risk._consecutive_losses = 0
        trader.log("API", f"Config updated: {updated}")
        _invalidate_status_cache(include_slow=True)
    
    return {"updated": updated, "config_version": _config_version}


@router.get("/config/profiles")
async def get_config_profiles():
    """Get saved config profiles and active assignments."""
    return _build_profile_dict()


@router.post("/config/profiles/{profile_name}")
async def save_config_profile(profile_name: str, request: Request):
    """Save current in-memory config into a named profile."""
    trader = get_auto_trader()
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    import config as cfg
    try:
        profile = cfg.save_runtime_profile(
            profile_name,
            description=body.get("description"),
            role_scope=body.get("role_scope"),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    trader.log("API", f"Saved config profile: {profile_name}")
    _bump_config_version()
    _invalidate_status_cache(include_slow=True)
    return {"saved": profile_name, "profile": profile, "profiles": _build_profile_dict(), "config_version": _config_version}


@router.post("/config/profiles/{profile_name}/activate/{role}")
async def activate_config_profile(profile_name: str, role: str, request: Request):
    """Assign a profile to live or backtest; applying immediately for live."""
    trader = get_auto_trader()
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    apply_now = bool(body.get("apply_now", role.strip().lower() == "live"))
    import config as cfg
    try:
        cfg.set_active_profile(role, profile_name, apply_now=apply_now)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    if role.strip().lower() == "live" and apply_now:
        trader.log("API", f"Applied live config profile: {profile_name}")
    else:
        trader.log("API", f"Set {role} config profile: {profile_name}")
    _bump_config_version()
    _invalidate_status_cache(include_slow=True)
    return {"active_profiles": _build_profile_dict()["active_profiles"], "config_version": _config_version}


@router.post("/config/profiles/{profile_name}/update")
async def update_config_profile(profile_name: str, request: Request):
    """Update a saved profile with explicit values."""
    trader = get_auto_trader()
    body = await request.json()
    import config as cfg
    try:
        profile = cfg.update_runtime_profile(
            profile_name,
            values=body.get("values", {}),
            description=body.get("description"),
            role_scope=body.get("role_scope"),
        )
        if cfg.get_active_profile("live") == profile["name"]:
            cfg.apply_runtime_profile(profile["name"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    if cfg.get_active_profile("live") == profile["name"]:
        trader.log("API", f"Updated config profile and applied live changes: {profile_name}")
    else:
        trader.log("API", f"Updated config profile: {profile_name}")
    _bump_config_version()
    _invalidate_status_cache(include_slow=True)
    return {"profile": profile, "profiles": _build_profile_dict(), "config_version": _config_version}


@router.post("/config/profiles/{profile_name}/apply-to-live")
async def apply_profile_to_live(profile_name: str, request: Request):
    """Copy all knobs from a source profile into a live-scoped target profile and apply immediately."""
    trader = get_auto_trader()
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    import config as cfg
    target_profile = str(body.get("target_profile") or cfg.get_active_profile("live")).strip()
    try:
        source_values = cfg.get_profile_values(profile_name)
        if not source_values:
            raise ValueError(f"Source profile has no values: {profile_name}")
        profile = cfg.copy_profile_to_profile(
            source_name=profile_name,
            target_name=target_profile,
            target_scope="live",
            apply_if_active_live=True,
        )
        if cfg.get_active_profile("live") != profile["name"]:
            cfg.set_active_profile("live", profile["name"], apply_now=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    trader.log("API", f"Applied profile {profile_name} to live profile {target_profile}")
    _bump_config_version()
    _invalidate_status_cache(include_slow=True)
    return {
        "source_profile": profile_name,
        "target_profile": target_profile,
        "profile": profile,
        "profiles": _build_profile_dict(),
        "config_version": _config_version,
    }

@router.get("/chart-data")
async def get_chart_data(
    symbol: str = Query(default=None),
    timeframe: str = Query(default="M5"),
    limit: int = Query(default=200, ge=20, le=1000),
):
    """Single payload for the chart panel: candles + all bot analysis layers."""
    from engine.trendline import detect_trendlines
    trader = get_auto_trader()
    sym = (symbol or "").upper() or (trader.bridge.current_symbol if trader.bridge else cfg.SYMBOL)
    tf  = timeframe.upper()

    def _run():
        df = trader.bridge.fetch_candles(tf)
        if df is None or df.empty:
            return {"error": "no_data"}
        df = df.tail(limit).reset_index(drop=True)

        candles = [
            {
                "time":  int(row.datetime.timestamp()),
                "open":  round(float(row.open),  5),
                "high":  round(float(row.high),  5),
                "low":   round(float(row.low),   5),
                "close": round(float(row.close), 5),
                "volume": round(float(row.volume), 0),
            }
            for row in df.itertuples()
        ]

        tl_result = detect_trendlines(df, symbol=sym, timeframe=tf)

        # ATR for the selected chart timeframe (not M1)
        # Use Wilder's RMA smoothing to match TradingView/MT5 display
        from engine.indicators import wilder_atr as _compute_atr
        import numpy as np
        try:
            h_arr = df["high"].values.astype(float)
            l_arr = df["low"].values.astype(float)
            c_arr = df["close"].values.astype(float)
            chart_atr14 = round(float(_compute_atr(h_arr, l_arr, c_arr, 14)[-1]), 4)
            chart_atr7  = round(float(_compute_atr(h_arr, l_arr, c_arr, 7)[-1]),  4)
        except Exception:
            chart_atr14 = 0.0
            chart_atr7  = 0.0

        # Zones from trader cache
        zones = {}
        try:
            zones = dict(getattr(trader, "_zones", {}) or {})
        except Exception:
            pass

        # Regime + bias from trader cache
        regime = dict(getattr(trader, "_regime", {}) or {})
        bias   = dict(getattr(trader, "_bias",   {}) or {})

        # Open trades
        open_trades = getattr(trader, "_cached_positions", None) or []

        # Last signal from strategy (if available)
        last_signal = {}
        try:
            last_signal = dict(getattr(trader, "_last_signal", {}) or {})
        except Exception:
            pass

        # Tick snapshot for pressure/quality data
        tick_snap = {}
        try:
            tick_snap = dict(getattr(trader, "_tick_snap", {}) or {})
        except Exception:
            pass

        # Liquidity key levels
        liquidity = {}
        try:
            liquidity = dict(getattr(trader, "_liquidity", {}) or {})
        except Exception:
            pass

        # Indicators (already computed each cycle)
        indicators = dict(getattr(trader, "_indicators", {}) or {})

        return {
            "symbol":      sym,
            "timeframe":   tf,
            "candles":     candles,
            "trendlines":  tl_result.get("trendlines", []),
            "swing_highs": tl_result.get("swing_highs", []),
            "swing_lows":  tl_result.get("swing_lows",  []),
            "structure":   tl_result.get("structure",   []),
            "trend":       tl_result.get("trend",       "range"),
            "zones":       zones,
            "regime":      regime,
            "bias":        bias,
            "open_trades": open_trades,
            "last_signal": last_signal,
            "tick_snap":   tick_snap,
            "liquidity":   liquidity,
            "indicators":  indicators,
            "chart_atr14": chart_atr14,
            "chart_atr7":  chart_atr7,
        }

    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(loop.run_in_executor(None, _run), timeout=10.0)
    except asyncio.TimeoutError:
        return JSONResponse({"error": "timeout"}, status_code=504)
    return convert_numpy_types(result)


@router.get("/htf-status")
async def get_htf_status():
    """HTF Bias Engine — live macro, weekly sentiment, and signal state."""
    from engine.macro_fetcher import macro_fetcher
    from engine.calendar import calendar
    from engine.htf_bias_engine import evaluate as htf_evaluate, _macro_bias, _weekly_bias
    trader = _auto_trader_instance

    macro_state = macro_fetcher.get()
    calendar.poll()
    cal = calendar.check()

    def _run():
        # Read directly from the engine's cached market_state — same object strategies use
        ms = dict(getattr(trader, "_market_state", {}) or {})
        tick = getattr(trader, "_last_tick", {}) or {}
        price = float(tick.get("bid", 0.0))

        structure = dict(ms.get("structure") or {})
        weekly = dict(ms.get("weekly_range") or {})
        pullback_pct = float(ms.get("pullback_depth") or 0.0)
        body_ratio = dict(ms.get("body_ratio") or {})

        htf_data = {
            "price": price,
            "ohlc": dict(ms.get("ohlc") or {}),
            "weekly_range": weekly,
            "structure": structure,
            "momentum": {
                "D1_body_ratio": body_ratio.get("D1", 0.0),
                "H4_body_ratio": body_ratio.get("H4", 0.0),
            },
            "macro": {
                "dxy_trend":   macro_state["dxy_trend"],
                "us10y_trend": macro_state["us10y_trend"],
            },
            "news": {
                "recent_events_bias": "NEUTRAL",
                "high_impact_soon": bool(cal.get("blocked")),
            },
            "pullback": {"depth_pct": pullback_pct},
            "account": {},
        }

        result = htf_evaluate(htf_data)
        w_bias = _weekly_bias(htf_data)
        macro  = _macro_bias(htf_data)

        return {
            "macro": {
                "dxy_trend":   macro_state["dxy_trend"],
                "us10y_trend": macro_state["us10y_trend"],
                "macro_bias":  macro,
                "source":      macro_state.get("source", "unknown"),
                "fetched_at":  macro_state.get("fetched_at_iso", ""),
            },
            "weekly": {
                "bias":        w_bias,
                "high_7d":     weekly.get("high_7d", 0.0),
                "low_7d":      weekly.get("low_7d", 0.0),
                "mid_7d":      weekly.get("mid_7d", 0.0),
                "pullback_pct": round(pullback_pct * 100, 1),
            },
            "structure": structure,
            "momentum": htf_data["momentum"],
            "news_blocked": bool(cal.get("blocked")),
            "news_reason":  str(cal.get("reason") or ""),
            "signal": {
                "decision":   result["decision"],
                "strategy":   result["strategy"],
                "confidence": result["confidence"],
                "entry_zone": result["entry_zone"],
                "sl":         result["sl"],
                "tp":         result["tp"],
                "reason":     result["reason"],
            },
        }

    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(loop.run_in_executor(None, _run), timeout=6.0)
    except asyncio.TimeoutError:
        return JSONResponse({"error": "timeout"}, status_code=504)
    return convert_numpy_types(result)


@router.get("/trendlines")
async def get_trendlines(
    symbol: str = Query(default=None),
    timeframe: str = Query(default="M5"),
):
    """Detect trendlines for a symbol/timeframe using live MT5 candle data."""
    from engine.trendline import detect_trendlines
    trader = get_auto_trader()
    sym = (symbol or "").upper() or (trader.bridge.current_symbol if trader.bridge else cfg.SYMBOL)
    loop = asyncio.get_event_loop()
    def _run():
        df = trader.bridge.fetch_candles(timeframe.upper())
        if df is None or df.empty:
            return {"error": "no_data"}
        return detect_trendlines(df, symbol=sym, timeframe=timeframe.upper())
    try:
        result = await asyncio.wait_for(loop.run_in_executor(None, _run), timeout=8.0)
    except asyncio.TimeoutError:
        return JSONResponse({"error": "timeout"}, status_code=504)
    return convert_numpy_types(result)


@router.post("/shutdown")
async def shutdown():
    """Shutdown the trading system."""
    trader = get_auto_trader()
    trader.enabled = False
    trader._running = False
    if trader.trades:
        trader.trades.close_all()
    trader.log("API", "SHUTDOWN")
    _invalidate_status_cache(include_slow=True)
    
    # Schedule shutdown
    import threading
    import time
    import os
    threading.Thread(target=lambda: (time.sleep(1), os._exit(0)), daemon=True).start()
    
    return {"message": "Shutting down"}

@router.post("/reload")
async def hot_reload():
    """Hot reload the trading system - restart background service."""
    trader = get_auto_trader()
    trader.log("API", "HOT RELOAD initiated")
    previous_enabled = bool(trader.enabled)
    
    # Stop current trading
    trader.enabled = False
    _invalidate_status_cache()
    
    # Close any open positions (optional - comment out if you want to keep positions)
    # if trader.trades:
    #     trader.trades.close_all()
    
    # Restart the engine in a separate thread
    import threading
    import time
    
    def restart_engine():
        time.sleep(0.5)  # Brief pause
        trader.log("API", "Restarting engine...")
        
        # Reinitialize components
        try:
            # Reconnect MT5
            if trader.bridge:
                trader.bridge.connect()
            
            # Reload strategies
            if trader.strat_mgr:
                trader._register_strategies()
                
                # Restore active strategy
                if hasattr(trader, "_apply_startup_strategy"):
                    trader._apply_startup_strategy()
            
            # Clear caches to force fresh data
            trader._candles.clear()
            trader._candle_counts.clear()
            trader._indicators.clear()
            trader._zones.clear()
            trader._regime.clear()
            trader._bias.clear()
            trader._liquidity.clear()
            trader._last_tick = {}
            
            # Reload candle data
            for tf in ["M1", "M5", "M15", "H1", "H4", "D1"]:
                try:
                    df = trader.bridge.fetch_candles(tf)
                    if not df.empty:
                        trader._candles[tf] = df
                        trader._candle_counts[tf] = len(df)
                except Exception as e:
                    trader.log("RELOAD", f"Failed to load {tf} data: {e}")
            
            # Recompute all indicators
            trader._recompute_all()
            if hasattr(trader, "mark_restart_time"):
                trader.mark_restart_time()
            trader.enabled = previous_enabled
            
            trader.log("ENGINE", "Trading engine started")
            trader.log("API", "HOT RELOAD completed - engine restarted")
            _invalidate_status_cache(include_slow=True)
            
        except Exception as e:
            trader.log("ERROR", f"Hot reload failed: {e}")
    
    threading.Thread(target=restart_engine, daemon=True).start()
    
    return {"message": "Hot reload initiated - engine restarting"}
