from __future__ import annotations

import copy
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from .config_adapter import safe_float
from .market_context_builder import build_market_context


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _setup_type(signal: Dict[str, Any]) -> str:
    return str(
        signal.get("_zone_type")
        or signal.get("setup_type")
        or signal.get("_signal_family")
        or signal.get("_route_type")
        or "UNKNOWN"
    ).upper()


def _direction(signal: Dict[str, Any]) -> str:
    action = str(signal.get("signal") or signal.get("direction") or "").upper()
    if action == "BUY":
        return "BUY"
    if action == "SELL":
        return "SELL"
    return action or "UNKNOWN"


def _signal_id(signal: Dict[str, Any], setup_signature: str) -> str:
    return str(signal.get("_signal_id") or signal.get("_trade_id") or setup_signature or f"signal-{int(time.time() * 1000)}")


def build_signal_snapshot(strategy_name: str, signal: Dict[str, Any], market_data: Dict[str, Any], setup_signature: str) -> Dict[str, Any]:
    context = build_market_context(signal, market_data)
    gate_preview = signal.get("_master_gate_preview") or {}
    validation = signal.get("_validation_result") or {}
    tick = market_data.get("tick") or {}
    snapshot = {
        "signal_id": _signal_id(signal, setup_signature),
        "trade_id": str(signal.get("_trade_id") or ""),
        "timestamp": _utc_now_iso(),
        "symbol": context.get("symbol") or "",
        "strategy": strategy_name,
        "direction": _direction(signal),
        "setup_type": _setup_type(signal),
        "entry_reason": str(signal.get("reason") or ""),
        "entry_price": round(safe_float(signal.get("entry"), safe_float(tick.get("bid") or tick.get("ask"), 0.0)), 2),
        "proposed_sl": round(safe_float(signal.get("sl"), 0.0), 2),
        "proposed_tp": round(safe_float(signal.get("tp"), 0.0), 2),
        "rr": round(safe_float(signal.get("rr"), 0.0), 3),
        "spread_at_signal": context.get("spread_at_signal"),
        "spread_before_entry": context.get("spread_before_entry"),
        "session": context.get("session"),
        "market_state": context.get("market_state"),
        "trend": context.get("trend"),
        "price_vs_vwap": context.get("price_vs_vwap"),
        "price_vs_ema9": context.get("price_vs_ema9"),
        "price_vs_ema20": context.get("price_vs_ema20"),
        "ema20_slope": context.get("ema20_slope"),
        "atr": context.get("atr"),
        "recent_high": context.get("recent_high"),
        "recent_low": context.get("recent_low"),
        "nearest_support_zone": context.get("nearest_support_zone"),
        "nearest_resistance_zone": context.get("nearest_resistance_zone"),
        "liquidity_sweep": context.get("liquidity_sweep"),
        "rejection_candle": context.get("rejection_candle"),
        "candle_confirmation": context.get("candle_confirmation"),
        "tick_pressure": context.get("tick_pressure"),
        "volume_state": context.get("volume_state"),
        "compression_state": context.get("compression_state"),
        "abnormal_candle_state": context.get("abnormal_candle_state"),
        "news_state": context.get("news_state"),
        "all_gates_passed": list(gate_preview.get("passed_gates", [])),
        "all_gates_failed": list(gate_preview.get("failed_gates", [])),
        "validation_blocks": list((validation or {}).get("blocks", [])),
        "status": "SIGNAL_CAPTURED",
        "reason_taken_or_skipped": "",
    }
    return snapshot


def build_open_trade_snapshot(signal_snapshot: Dict[str, Any], signal: Dict[str, Any], trade_ticket: int, fill_price: float) -> Dict[str, Any]:
    live_snapshot = {
        "signal_id": signal_snapshot.get("signal_id"),
        "trade_ticket": int(trade_ticket),
        "entry_time": _utc_now_iso(),
        "fill_price": round(safe_float(fill_price), 2),
        "direction": signal_snapshot.get("direction"),
        "initial_sl": round(safe_float(signal.get("sl"), 0.0), 2),
        "initial_tp": round(safe_float(signal.get("tp"), 0.0), 2),
        "mfe_points": 0.0,
        "mae_points": 0.0,
        "mfe_r": 0.0,
        "mae_r": 0.0,
        "time_to_mfe_seconds": None,
        "time_to_mae_seconds": None,
        "price_immediately_moved_in_favour": False,
        "price_immediately_moved_against": False,
        "spread_widened": False,
        "market_state_changed": False,
        "micro_trend_flipped": False,
        "vwap_ema_relationship_changed": False,
        "invalidation_before_sl": False,
        "open_context": copy.deepcopy(signal_snapshot),
    }
    return live_snapshot


def update_open_trade_snapshot(existing: Dict[str, Any], trade, market_data: Dict[str, Any]) -> Dict[str, Any]:
    updated = copy.deepcopy(existing)
    entry_price = safe_float(updated.get("fill_price"), 0.0)
    current_price = safe_float(getattr(trade, "current_price", 0.0), entry_price)
    sl_distance = max(0.01, safe_float(getattr(trade, "sl_distance", 0.0), 0.01))
    if str(getattr(trade, "direction", "")).upper() == "BUY":
        favourable = current_price - entry_price
        adverse = min(0.0, current_price - entry_price)
    else:
        favourable = entry_price - current_price
        adverse = min(0.0, entry_price - current_price)

    prior_mfe = safe_float(updated.get("mfe_points"), 0.0)
    prior_mae = safe_float(updated.get("mae_points"), 0.0)
    now_ts = datetime.now(timezone.utc)
    entry_time = datetime.fromisoformat(updated.get("entry_time")) if updated.get("entry_time") else now_ts
    held_seconds = max(0, int((now_ts - entry_time).total_seconds()))

    if favourable > prior_mfe:
        updated["mfe_points"] = round(favourable, 4)
        updated["mfe_r"] = round(favourable / sl_distance, 3)
        updated["time_to_mfe_seconds"] = held_seconds
    if adverse < prior_mae:
        updated["mae_points"] = round(adverse, 4)
        updated["mae_r"] = round(adverse / sl_distance, 3)
        updated["time_to_mae_seconds"] = held_seconds

    if held_seconds <= 60:
        updated["price_immediately_moved_in_favour"] = favourable > 0
        updated["price_immediately_moved_against"] = adverse < 0

    current_spread = safe_float((market_data.get("tick") or {}).get("spread"), 0.0)
    updated["spread_widened"] = current_spread > safe_float((updated.get("open_context") or {}).get("spread_at_signal"), 0.0) + 0.05
    initial_market_state = str((updated.get("open_context") or {}).get("market_state") or "")
    updated["market_state_changed"] = initial_market_state not in {"", str((market_data.get("regime") or {}).get("state") or "")}
    tick_pressure = market_data.get("tick_pressure") or {}
    original_tick = ((updated.get("open_context") or {}).get("tick_pressure") or {}).get("directional_bias")
    updated["micro_trend_flipped"] = bool(original_tick and tick_pressure.get("directional_bias") and original_tick != tick_pressure.get("directional_bias"))
    updated["vwap_ema_relationship_changed"] = str((updated.get("open_context") or {}).get("price_vs_ema20")) != str(build_market_context({}, market_data).get("price_vs_ema20"))
    updated["last_updated_at"] = _utc_now_iso()
    return updated


def build_close_trade_snapshot(signal_snapshot: Dict[str, Any], live_snapshot: Dict[str, Any], db_order: Dict[str, Any], mt5_trade: Dict[str, Any]) -> Dict[str, Any]:
    entry_time = live_snapshot.get("entry_time") or signal_snapshot.get("timestamp") or _utc_now_iso()
    close_time = str(db_order.get("close_time") or mt5_trade.get("close_time") or _utc_now_iso())
    try:
        held_seconds = int((datetime.fromisoformat(close_time) - datetime.fromisoformat(entry_time)).total_seconds())
    except Exception:
        held_seconds = int(safe_float(db_order.get("held_seconds"), 0))

    pnl = safe_float(db_order.get("final_pnl"), safe_float(mt5_trade.get("pnl"), 0.0))
    pnl_r = safe_float(db_order.get("final_r"), pnl / max(0.01, safe_float(signal_snapshot.get("rr"), 1.0)))
    close_reason = str(db_order.get("close_reason") or db_order.get("mt5_close_reason") or mt5_trade.get("comment") or "")
    exit_price = safe_float(db_order.get("exit_price"), safe_float(mt5_trade.get("exit_price"), 0.0))
    peak_r = safe_float(db_order.get("peak_r"), safe_float(live_snapshot.get("mfe_r"), 0.0))

    # TODO: Post-exit path tracking is only available while this service stays online.
    # The nearest safe integration point is the runtime cycle updater in service_runner.py.
    close_snapshot = {
        "exit_time": close_time,
        "exit_price": round(exit_price, 2) if exit_price else 0.0,
        "pnl": round(pnl, 2),
        "pnl_r": round(pnl_r, 3),
        "exit_reason": close_reason,
        "hold_time_seconds": held_seconds,
        "sl_hit": str(db_order.get("close_reason_category") or "") in {"sl", "breakeven_stop"},
        "tp_hit": str(db_order.get("close_reason_category") or "") == "tp",
        "early_exit_hit": str(db_order.get("close_reason_category") or "") not in {"sl", "tp", ""},
        "price_later_moved_in_intended_direction_after_exit": peak_r > max(0.2, pnl_r + 0.25),
        "max_favourable_move_after_exit": round(max(0.0, peak_r - pnl_r), 3),
        "max_adverse_move_after_exit": round(abs(min(0.0, safe_float(live_snapshot.get("mae_r"), 0.0))), 3),
        "waiting_one_candle_would_have_helped": bool(live_snapshot.get("price_immediately_moved_against")) and peak_r > max(0.3, pnl_r + 0.3),
        "retest_entry_would_have_helped": bool(signal_snapshot.get("liquidity_sweep")) and peak_r > max(0.4, pnl_r + 0.25),
        "wider_sl_beyond_structure_would_have_helped": safe_float(live_snapshot.get("mae_r"), 0.0) < -1.0 and peak_r > 0.5,
        "smaller_tp_or_trailing_would_have_helped": peak_r > 0.5 and pnl_r < peak_r,
        "peak_r": peak_r,
        "mae_r": safe_float(live_snapshot.get("mae_r"), 0.0),
        "mfe_r": safe_float(live_snapshot.get("mfe_r"), 0.0),
    }
    return close_snapshot
