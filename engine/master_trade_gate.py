"""
Centralized execution gate for all live trade candidates.

This module intentionally runs immediately before order send so every strategy
passes through one final, config-driven quality and safety checkpoint.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, Optional

import config as cfg
from .session_filter import get_session_state_at

_IST = timezone(timedelta(hours=5, minutes=30))

# Optional callback set by auto_trader: fn(tag, msg) -> None
_gate_log_fn: Optional[Callable[[str, str], None]] = None


def set_gate_log_fn(fn: Callable[[str, str], None]) -> None:
    global _gate_log_fn
    _gate_log_fn = fn


def _gate_log(tag: str, msg: str) -> None:
    if _gate_log_fn is not None:
        try:
            _gate_log_fn(tag, msg)
        except Exception:
            pass


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
    signal_family = str(signal.get("_signal_family") or "").upper()
    signal_family_bucket = _family_bucket(signal_family)
    tick_snap = signal.get("_tick_snapshot") or market_state.get("tick_snapshot") or {}
    current_rr = float(signal.get("rr", 0.0) or 0.0)

    strategy_name = str(
        signal.get("_strategy_name")
        or signal.get("strategy")
        or ""
    ).upper()
    strategy = strategy_name or signal_family or "UNKNOWN"

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
        _gate_log("GATE", f"[{strategy}|{direction}] BLOCKED gate={gate_name} reason={reason}")
        return result

    def mark_pass(gate_name: str, reason: str) -> None:
        result["passed_gates"].append(gate_name)
        result["gate_trace"].append({"gate": gate_name, "allowed": True, "reason": reason})

    if bool(getattr(cfg_module, "TEMP_DISABLE_GLOBAL_BLOCKS", False)):
        result["score"] = 100
        result["confirmation_count"] = 4
        result["auto_relax"] = {"active": False, "reason": "GLOBAL_BLOCKS_DISABLED"}
        result["required_rr"] = current_rr
        mark_pass("global_bypass", "GLOBAL_BLOCKS_DISABLED")
        return result

    # 1. Risk gate
    if risk_manager is not None:
        allowed, reason = risk_manager.can_trade(
            account,
            len(positions),
            strategy_name=strategy_name,
            strategy_trade_counts=market_state.get("strategy_trade_counts") or {},
        )
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

    # For SWING and INTRADAY families: all filtering is done inside the strategy.
    # Only enforce risk, session, and spread gates — skip score/confirmation/counter-trend.
    if signal_family in {"SWING", "INTRADAY"}:
        if not result["allowed"]:
            return result
        # spread already checked above, risk + session already checked above
        result["score"] = 100
        result["confirmation_count"] = 4
        mark_pass("m15_context", "M15_CONTEXT_SKIPPED")
        mark_pass("opposing_zone", "OPPOSING_ZONE_SKIPPED")
        mark_pass("signal", "SIGNAL_PASS")
        mark_pass("candle_confirmation", "CANDLE_PASS")
        mark_pass("confirmation", "CONFIRMATION_PASS: 4/4")
        mark_pass("trade_score", "TRADE_SCORE_SKIPPED")
        mark_pass("rr", "RR_PASS")
        mark_pass("counter_trend", "COUNTER_TREND_SKIPPED")
        mark_pass("tier1", "TIER1_PASS")
        return result

    # 5. M15 context gate
    m15_context = {
        "allowed": True,
        "reason": "M15_CONTEXT_PASS",
        "distance_atr": None,
        "candle_confirmation": False,
        "zone": None,
        "support_zone": None,
        "resistance_zone": None,
    }
    skip_m15_context = signal_family_bucket in {"SWEEP", "M15", "TREND", "SWING", "INTRADAY"}
    provider_context = None
    if m15_context_provider and hasattr(m15_context_provider, "evaluate_trade_context"):
        provider_context = m15_context_provider.evaluate_trade_context(market_state, direction, signal)
    if provider_context is not None:
        if not skip_m15_context:
            m15_context = provider_context
        else:
            m15_context.update(
                {
                    "reason": "M15_CONTEXT_SKIPPED",
                    "distance_atr": provider_context.get("distance_atr"),
                    "candle_confirmation": bool(signal.get("_candle_confirmation")) or bool(provider_context.get("candle_confirmation")),
                    "zone": provider_context.get("zone"),
                    "support_zone": provider_context.get("support_zone"),
                    "resistance_zone": provider_context.get("resistance_zone"),
                    "lookback_hours_used": provider_context.get("lookback_hours_used"),
                    "breakout_confirmation": provider_context.get("breakout_confirmation"),
                }
            )
    result["m15_context"] = m15_context
    if not skip_m15_context and not m15_context.get("allowed", True):
        return block("m15_context", str(m15_context.get("reason") or "M15_CONTEXT_BLOCK"))
    mark_pass("m15_context", str(m15_context.get("reason") or "M15_CONTEXT_PASS"))

    zone_ok, zone_reason = _opposing_zone_entry_check(signal, tick, indicators, m15_context)
    if not zone_ok:
        return block("opposing_zone", zone_reason)
    mark_pass("opposing_zone", zone_reason)

    # 6-8. Signal, candle, RR, score, and confirmation gates
    sweep_confirmed = bool(signal.get("_sweep_confirmed"))
    m15_zone_confirmed = bool(signal.get("_m15_zone_confirmed"))
    candle_confirmed = bool(signal.get("_candle_confirmation")) or bool(m15_context.get("candle_confirmation"))
    ema_aligned = bool(signal.get("_ema_aligned", True))
    bias_confirmed = bias_dir == trade_dir and trade_dir in ("LONG", "SHORT")
    session_bonus_ok = session_state.get("session") in ("LONDON", "NEW_YORK")
    atr_healthy = _atr_is_healthy(indicators, spike_context)
    signal_confirmed = _signal_is_confirmed(signal_family, sweep_confirmed, candle_confirmed)
    if not signal_confirmed:
        return block("signal", f"SIGNAL_BLOCK: {signal_family or 'UNKNOWN'} confirmation missing")
    mark_pass("signal", "SIGNAL_PASS")

    if not candle_confirmed:
        return block("candle_confirmation", "CANDLE_BLOCK: confirmation missing")
    mark_pass("candle_confirmation", "CANDLE_PASS")

    confirmations = {
        "bias": bias_confirmed,
        "setup": sweep_confirmed or m15_zone_confirmed,
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
        if sweep_confirmed or m15_zone_confirmed:
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
        if cfg_module.COUNTER_TREND_REQUIRE_SWEEP and not _counter_trend_confirmation_passes(signal_family, sweep_confirmed, candle_confirmed):
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
    family = _family_bucket(str(signal.get("_signal_family") or "").upper())
    if family == "SWEEP":
        return float(getattr(cfg_module, "SCALPER_SPREAD_MEAN_MAX", cfg_module.M15_SR_MAX_SPREAD))
    if family == "SMC":
        return float(getattr(cfg_module, "SMC_SPREAD_MEAN_MAX", cfg_module.M15_SR_MAX_SPREAD))
    if family == "TREND":
        return float(getattr(cfg_module, "TREND_CHANNEL_MAX_SPREAD", cfg_module.M15_SR_MAX_SPREAD))
    return float(cfg_module.M15_SR_MAX_SPREAD)


def _opposing_zone_entry_check(
    signal: Dict[str, Any],
    tick: Dict[str, Any],
    indicators: Dict[str, Any],
    m15_context: Dict[str, Any],
) -> tuple[bool, str]:
    direction = str(signal.get("signal") or "").upper()
    if direction not in {"BUY", "SELL"}:
        return True, "OPPOSING_ZONE_PASS"

    if direction == "SELL":
        entry = float(signal.get("entry", tick.get("bid", 0.0)) or 0.0)
        opposing_zone = m15_context.get("support_zone")
        zone_label = "support"
    else:
        entry = float(signal.get("entry", tick.get("ask", 0.0)) or 0.0)
        opposing_zone = m15_context.get("resistance_zone")
        zone_label = "resistance"

    if entry <= 0 or not isinstance(opposing_zone, dict):
        return True, "OPPOSING_ZONE_PASS"

    zone_low = float(opposing_zone.get("zone_low", 0.0) or 0.0)
    zone_high = float(opposing_zone.get("zone_high", 0.0) or 0.0)
    if zone_high <= zone_low:
        return True, "OPPOSING_ZONE_PASS"

    spread = float(tick.get("spread", signal.get("_entry_spread", 0.0)) or 0.0)
    atr = float(indicators.get("atr14", indicators.get("atr", signal.get("_atr14", 0.0))) or 0.0)
    near_buffer = max(spread * 2.0, atr * 0.12, 0.25)

    if zone_low <= entry <= zone_high:
        return False, f"OPPOSING_ZONE_BLOCK: entry inside {zone_label} zone [{zone_low:.2f}-{zone_high:.2f}]"

    if direction == "SELL":
        distance = entry - zone_high
        if distance >= 0 and distance < near_buffer:
            return False, f"OPPOSING_ZONE_BLOCK: short too close to support [{zone_low:.2f}-{zone_high:.2f}] ({distance:.2f} pts)"
    else:
        distance = zone_low - entry
        if distance >= 0 and distance < near_buffer:
            return False, f"OPPOSING_ZONE_BLOCK: long too close to resistance [{zone_low:.2f}-{zone_high:.2f}] ({distance:.2f} pts)"

    return True, "OPPOSING_ZONE_PASS"


def _signal_is_confirmed(signal_family: str, sweep_confirmed: bool, candle_confirmed: bool) -> bool:
    family = _family_bucket(str(signal_family or "").upper())
    if family == "SMC":
        return True
    if family == "SWEEP":
        return sweep_confirmed
    if family in {"M15", "TREND"}:
        return candle_confirmed or sweep_confirmed
    return sweep_confirmed or candle_confirmed


def _counter_trend_confirmation_passes(signal_family: str, sweep_confirmed: bool, candle_confirmed: bool) -> bool:
    family = _family_bucket(str(signal_family or "").upper())
    if family == "SWEEP":
        return sweep_confirmed
    if family in {"M15", "TREND"}:
        return sweep_confirmed or candle_confirmed
    if family == "SMC":
        return sweep_confirmed or candle_confirmed
    return sweep_confirmed


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


def _family_bucket(signal_family: str) -> str:
    family = str(signal_family or "").upper()
    if family.startswith("M15_SR_") or family.startswith("M15_ZONE"):
        return "M15"
    return family
