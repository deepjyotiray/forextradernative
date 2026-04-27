"""
Centralized execution gate for all live trade candidates.

This module intentionally runs immediately before order send so every strategy
passes through one final, config-driven quality and safety checkpoint.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import config as cfg
from .session_filter import get_session_state_at


def master_trade_gate(
    signal: Dict[str, Any],
    market_state: Dict[str, Any],
    cfg_module=cfg,
    risk_manager=None,
    m15_context_provider=None,
) -> Dict[str, Any]:
    """
    Returns:
        {
            "allowed": bool,
            "reason": str,
            "score": int,
            "adjusted_signal": dict,
            "passed_gates": list[str],
            "failed_gates": list[str],
            "gate_trace": list[dict],
            ...
        }
    """
    signal = dict(signal or {})
    now_utc = _resolve_now(market_state.get("now_utc"))
    tick = market_state.get("tick") or {}
    indicators = market_state.get("indicators") or {}
    account = market_state.get("account") or {}
    positions = market_state.get("positions") or []
    direction = str(signal.get("signal") or "").upper()
    trade_dir = "LONG" if direction == "BUY" else "SHORT" if direction == "SELL" else "NEUTRAL"
    bias_dir = _resolve_bias_direction(signal, market_state)
    counter_trend = bias_dir not in ("", "NEUTRAL", trade_dir) and trade_dir in ("LONG", "SHORT")
    tick_snap = signal.get("_tick_snapshot") or market_state.get("tick_snapshot") or {}
    current_rr = float(signal.get("rr", 0.0) or 0.0)

    result = {
        "allowed": True,
        "reason": "MASTER_GATE_PASS",
        "score": 0,
        "adjusted_signal": signal,
        "passed_gates": [],
        "failed_gates": [],
        "gate_trace": [],
        "session_state": {},
        "m15_context": {},
        "confirmations": {},
        "auto_relax": {"active": False, "reason": "AUTO_RELAX_DISABLED"},
        "counter_trend": counter_trend,
        "required_rr": current_rr,
        "premium_only": False,
    }

    def block(gate_name: str, reason: str, score: Optional[int] = None) -> Dict[str, Any]:
        result["allowed"] = False
        result["reason"] = reason
        if score is not None:
            result["score"] = int(score)
        result["failed_gates"].append(gate_name)
        result["gate_trace"].append({"gate": gate_name, "allowed": False, "reason": reason})
        return result

    def mark_pass(gate_name: str, reason: str) -> None:
        result["passed_gates"].append(gate_name)
        result["gate_trace"].append({"gate": gate_name, "allowed": True, "reason": reason})

    # 1. Risk gate
    if risk_manager is not None:
        allowed, reason = risk_manager.can_trade(account, len(positions))
        if not allowed:
            return block("risk", reason)
        mark_pass("risk", "RISK_PASS")

    # 2. Session gate
    session_state = get_session_state_at(now_utc)
    result["session_state"] = session_state
    result["premium_only"] = bool(session_state.get("premium_only"))
    if not session_state.get("allowed", False):
        return block("session", str(session_state.get("reason") or "SESSION_BLOCK"))
    mark_pass("session", str(session_state.get("reason") or "SESSION_PASS"))

    # 3. Spread gate
    spread_limit = _resolve_spread_limit(signal, cfg_module)
    spread_value = float(tick.get("spread", signal.get("_entry_spread", 0.0)) or 0.0)
    spread_stable = tick_snap.get("spread_stable", True)
    if spread_value > spread_limit:
        return block("spread", f"SPREAD_BLOCK: {spread_value:.3f} > {spread_limit:.3f}")
    if cfg_module.AUTO_RELAX_REQUIRE_STABLE_SPREAD and tick_snap and not spread_stable:
        return block("spread", "SPREAD_BLOCK: spread unstable")
    mark_pass("spread", f"SPREAD_PASS: {spread_value:.3f} <= {spread_limit:.3f}")

    # 4. Spike gate
    spike_reason = "SPIKE_PASS"
    spike_blocked = False
    spike_context = {}
    if cfg_module.TIER1_BLOCK_SPIKE_ENTRY and m15_context_provider and hasattr(m15_context_provider, "evaluate_spike_context"):
        spike_context = m15_context_provider.evaluate_spike_context(market_state)
        spike_blocked = not spike_context.get("allowed", True)
        spike_reason = str(spike_context.get("reason") or ("SPIKE_BLOCK" if spike_blocked else "SPIKE_PASS"))
    result["spike_context"] = spike_context
    if spike_blocked:
        return block("spike", spike_reason)
    mark_pass("spike", spike_reason)

    # 5. M15 context gate
    m15_context = {"allowed": True, "reason": "M15_CONTEXT_PASS", "distance_atr": None, "candle_confirmation": False}
    if m15_context_provider and hasattr(m15_context_provider, "evaluate_trade_context"):
        m15_context = m15_context_provider.evaluate_trade_context(market_state, direction, signal)
    result["m15_context"] = m15_context
    if not m15_context.get("allowed", True):
        return block("m15_context", str(m15_context.get("reason") or "M15_CONTEXT_BLOCK"))
    mark_pass("m15_context", str(m15_context.get("reason") or "M15_CONTEXT_PASS"))

    # 6-8. Signal, candle, RR, score, and confirmation gates
    sweep_confirmed = bool(signal.get("_sweep_confirmed"))
    candle_confirmed = bool(signal.get("_candle_confirmation")) or bool(m15_context.get("candle_confirmation"))
    ema_aligned = bool(signal.get("_ema_aligned", True))
    bias_confirmed = bias_dir == trade_dir and trade_dir in ("LONG", "SHORT")
    session_bonus_ok = session_state.get("session") in ("LONDON", "NEW_YORK")
    atr_healthy = _atr_is_healthy(indicators, spike_context)
    if signal.get("_signal_family") == "SMC":
        signal_confirmed = True
    else:
        signal_confirmed = sweep_confirmed
    if not signal_confirmed:
        return block("signal", "SIGNAL_BLOCK: no sweep/smc confirmation")
    mark_pass("signal", "SIGNAL_PASS")

    if not candle_confirmed:
        return block("candle_confirmation", "CANDLE_BLOCK: confirmation missing")
    mark_pass("candle_confirmation", "CANDLE_PASS")

    confirmations = {
        "bias": bias_confirmed,
        "sweep": sweep_confirmed,
        "m15_context": bool(m15_context.get("allowed")),
        "candle": candle_confirmed,
    }
    confirmation_count = sum(1 for passed in confirmations.values() if passed)
    result["confirmations"] = confirmations
    result["confirmation_count"] = confirmation_count
    if not any((sweep_confirmed, bool(m15_context.get("allowed")), candle_confirmed)):
        return block("confirmation", "CONFIRMATION_BLOCK: no sweep + no M15 zone + no candle confirmation")
    if cfg_module.CONFIRMATION_3_OF_4_ENABLED and confirmation_count < 3:
        return block("confirmation", f"CONFIRMATION_BLOCK: {confirmation_count}/4")
    mark_pass("confirmation", f"CONFIRMATION_PASS: {confirmation_count}/4")

    auto_relax = evaluate_auto_relax(market_state, risk_manager, tick_snap, session_state, spike_context, cfg_module)
    result["auto_relax"] = auto_relax

    score = 0
    if cfg_module.TRADE_SCORE_ENABLED:
        if sweep_confirmed:
            score += 20
        if m15_context.get("allowed"):
            score += 20
        if ema_aligned:
            score += 15
        if _strong_candle_body(signal, indicators):
            score += 15
        if spread_value <= spread_limit:
            score += 10
        if atr_healthy:
            score += 10
        if session_bonus_ok:
            score += 10
    result["score"] = score

    min_score = int(cfg_module.TRADE_SCORE_MIN)
    if auto_relax.get("active"):
        min_score = int(cfg_module.AUTO_RELAX_MIN_SCORE)
    if result["premium_only"]:
        min_score = max(min_score, int(cfg_module.TRADE_SCORE_PREMIUM))
    if risk_manager is not None:
        streak = int((risk_manager.daily_status or {}).get("consecutive_losses", 0))
        if streak >= 2:
            min_score = max(min_score, int(cfg_module.TRADE_SCORE_STRICT_AFTER_LOSS))
    result["required_score"] = min_score
    if score < min_score:
        return block("trade_score", f"TRADE_SCORE_BLOCK: score={score} min={min_score}", score=score)
    mark_pass("trade_score", f"TRADE_SCORE_PASS: score={score}")

    required_rr = float(signal.get("_min_rr_required", current_rr) or current_rr)
    if auto_relax.get("active"):
        required_rr = min(required_rr, float(cfg_module.AUTO_RELAX_RR))
    result["required_rr"] = round(required_rr, 2)
    if current_rr < required_rr:
        return block("rr", f"RR_BLOCK: rr={current_rr:.2f} min={required_rr:.2f}", score=score)
    mark_pass("rr", f"RR_PASS: rr={current_rr:.2f}")

    if cfg_module.BLOCK_WEAK_COUNTER_TREND and counter_trend:
        if score < int(cfg_module.COUNTER_TREND_MIN_SCORE):
            return block("counter_trend", "COUNTER_TREND_BLOCK: weak counter-trend setup", score=score)
        if cfg_module.COUNTER_TREND_REQUIRE_SWEEP and not sweep_confirmed:
            return block("counter_trend", "COUNTER_TREND_BLOCK: weak counter-trend setup", score=score)
        if cfg_module.COUNTER_TREND_REQUIRE_M15_ZONE and not m15_context.get("allowed"):
            return block("counter_trend", "COUNTER_TREND_BLOCK: weak counter-trend setup", score=score)
        if cfg_module.COUNTER_TREND_REQUIRE_CANDLE_CONFIRMATION and not candle_confirmed:
            return block("counter_trend", "COUNTER_TREND_BLOCK: weak counter-trend setup", score=score)
    mark_pass("counter_trend", "COUNTER_TREND_PASS")

    # 9. Tier 1 execution gate
    signal_spread = float(signal.get("_entry_spread", spread_value) or spread_value)
    if (spread_value - signal_spread) > float(cfg_module.TIER1_SPREAD_RECHECK_MAX_DELTA):
        return block("tier1", "TIER1_BLOCK: Spread widened since signal", score=score)

    signal_ts_ms = float(signal.get("_signal_generated_ts_ms", 0.0) or 0.0)
    if signal_ts_ms > 0:
        delay_ms = max(0.0, time.time() * 1000.0 - signal_ts_ms)
        result["entry_delay_ms"] = round(delay_ms, 2)
        if delay_ms > float(cfg_module.TIER1_MAX_ENTRY_DELAY_MS):
            return block("tier1", "TIER1_BLOCK: Entry delay exceeded", score=score)
    mark_pass("tier1", "TIER1_PASS")

    result["adjusted_signal"] = signal
    return result


def evaluate_auto_relax(
    market_state: Dict[str, Any],
    risk_manager,
    tick_snapshot: Dict[str, Any],
    session_state: Dict[str, Any],
    spike_context: Dict[str, Any],
    cfg_module=cfg,
) -> Dict[str, Any]:
    if not cfg_module.AUTO_RELAX_ENABLED:
        return {"active": False, "reason": "AUTO_RELAX_DISABLED"}

    if cfg_module.AUTO_RELAX_ONLY_GOOD_SESSION and session_state.get("session") not in ("LONDON", "NEW_YORK"):
        return {"active": False, "reason": "AUTO_RELAX_BLOCK: session not eligible"}

    if cfg_module.AUTO_RELAX_REQUIRE_STABLE_SPREAD and tick_snapshot and not tick_snapshot.get("spread_stable", True):
        return {"active": False, "reason": "AUTO_RELAX_BLOCK: spread unstable"}

    if spike_context and not spike_context.get("allowed", True):
        return {"active": False, "reason": "AUTO_RELAX_BLOCK: recent spike"}

    indicators = market_state.get("indicators") or {}
    if not _atr_is_healthy(indicators, spike_context):
        return {"active": False, "reason": "AUTO_RELAX_BLOCK: ATR abnormal"}

    if risk_manager is None:
        return {"active": False, "reason": "AUTO_RELAX_BLOCK: risk state unavailable"}

    daily_status = risk_manager.daily_status or {}
    if cfg_module.AUTO_RELAX_BLOCK_DRAWDOWN:
        combined_pnl = float(daily_status.get("pnl", 0.0) or 0.0) + float(getattr(risk_manager, "_floating_pnl", 0.0) or 0.0)
        if combined_pnl < 0:
            return {"active": False, "reason": "AUTO_RELAX_BLOCK: drawdown active"}

    if cfg_module.AUTO_RELAX_BLOCK_AFTER_LOSS:
        if int(daily_status.get("consecutive_losses", 0) or 0) >= 1:
            return {"active": False, "reason": "AUTO_RELAX_BLOCK: loss streak active"}
        if str(getattr(risk_manager, "last_trade_outcome", "") or "") == "LOSS":
            return {"active": False, "reason": "AUTO_RELAX_BLOCK: last trade was a loss"}

    last_trade_time = float(getattr(risk_manager, "last_trade_time", 0.0) or 0.0)
    if last_trade_time <= 0:
        return {"active": False, "reason": "AUTO_RELAX_BLOCK: no reference trade yet"}
    minutes_since_trade = (time.time() - last_trade_time) / 60.0
    if minutes_since_trade < float(cfg_module.AUTO_RELAX_AFTER_MINUTES):
        return {"active": False, "reason": f"AUTO_RELAX_BLOCK: only {minutes_since_trade:.0f}m since last trade"}

    return {
        "active": True,
        "reason": f"AUTO_RELAX_ON: no trade for {cfg_module.AUTO_RELAX_AFTER_MINUTES} minutes, conditions stable",
        "minutes_since_trade": round(minutes_since_trade, 1),
    }


def _resolve_now(now_utc: Any) -> datetime:
    if isinstance(now_utc, datetime):
        return now_utc.astimezone(timezone.utc) if now_utc.tzinfo else now_utc.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _resolve_bias_direction(signal: Dict[str, Any], market_state: Dict[str, Any]) -> str:
    bias = signal.get("bias") or market_state.get("bias") or {}
    direction = str(bias.get("direction") or signal.get("_bias_direction") or "").upper()
    if direction == "BUY":
        return "LONG"
    if direction == "SELL":
        return "SHORT"
    return direction


def _resolve_spread_limit(signal: Dict[str, Any], cfg_module=cfg) -> float:
    family = str(signal.get("_signal_family") or "").upper()
    if family == "SWEEP":
        return float(cfg_module.M15_SR_MAX_SPREAD)
    if family == "SMC":
        return min(float(cfg_module.M15_SR_MAX_SPREAD), float(cfg_module.SMC_SPREAD_MEAN_MAX))
    return float(cfg_module.M15_SR_MAX_SPREAD)


def _strong_candle_body(signal: Dict[str, Any], indicators: Dict[str, Any]) -> bool:
    body_ratio = signal.get("_body_ratio")
    if body_ratio is None:
        body_ratio = indicators.get("body_ratio", 0.0)
    try:
        return float(body_ratio or 0.0) > 0.60
    except Exception:
        return False


def _atr_is_healthy(indicators: Dict[str, Any], spike_context: Dict[str, Any]) -> bool:
    if spike_context and spike_context.get("spike_detected"):
        return False
    atr_ratio = float(indicators.get("atr_ratio", 1.0) or 1.0)
    atr_value = float(indicators.get("atr14", indicators.get("atr", 0.0)) or 0.0)
    return atr_value > 0 and 0.6 <= atr_ratio <= 1.8
