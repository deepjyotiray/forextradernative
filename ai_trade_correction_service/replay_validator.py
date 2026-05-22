from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Tuple

from .decision_schema import AIDecisionType, LossClassification


def build_counterfactuals(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    signal = snapshot.get("signal_snapshot") or snapshot
    close_snapshot = snapshot.get("close_snapshot") or {}
    live_snapshot = snapshot.get("live_snapshot") or {}
    peak_r = float(close_snapshot.get("peak_r") or close_snapshot.get("mfe_r") or live_snapshot.get("mfe_r") or 0.0)
    final_r = float(close_snapshot.get("pnl_r") or 0.0)
    mae_r = float(close_snapshot.get("mae_r") or live_snapshot.get("mae_r") or 0.0)
    price_vs_vwap = str(signal.get("price_vs_vwap") or "UNKNOWN")
    price_vs_ema20 = str(signal.get("price_vs_ema20") or "UNKNOWN")
    direction = str(signal.get("direction") or "")

    vwap_misaligned = (direction == "BUY" and price_vs_vwap == "BELOW") or (direction == "SELL" and price_vs_vwap == "ABOVE")
    ema_misaligned = (direction == "BUY" and price_vs_ema20 == "BELOW") or (direction == "SELL" and price_vs_ema20 == "ABOVE")

    return {
        "skip_trade": {"estimated_effect_r": round(max(0.0, -final_r), 3), "would_help": final_r < 0},
        "wait_one_candle": {
            "estimated_effect_r": 0.6 if close_snapshot.get("waiting_one_candle_would_have_helped") else 0.0,
            "would_help": bool(close_snapshot.get("waiting_one_candle_would_have_helped")),
        },
        "wait_for_candle_close": {
            "estimated_effect_r": 0.5 if live_snapshot.get("price_immediately_moved_against") else 0.0,
            "would_help": bool(live_snapshot.get("price_immediately_moved_against")),
        },
        "wait_for_retest": {
            "estimated_effect_r": 0.55 if close_snapshot.get("retest_entry_would_have_helped") else 0.0,
            "would_help": bool(close_snapshot.get("retest_entry_would_have_helped")),
        },
        "require_sweep_reclaim": {
            "estimated_effect_r": 0.4 if signal.get("liquidity_sweep") else 0.0,
            "would_help": bool(signal.get("liquidity_sweep") and live_snapshot.get("price_immediately_moved_against")),
        },
        "wider_sl": {
            "estimated_effect_r": round(max(0.0, peak_r), 3) if close_snapshot.get("wider_sl_beyond_structure_would_have_helped") else 0.0,
            "would_help": bool(close_snapshot.get("wider_sl_beyond_structure_would_have_helped")),
        },
        "smaller_tp": {
            "estimated_effect_r": round(max(0.0, min(peak_r, 1.0) - final_r), 3),
            "would_help": peak_r >= 0.6 and final_r < peak_r,
        },
        "trailing_0_5r": {"estimated_effect_r": round(max(0.0, min(peak_r, 0.5) - final_r), 3), "would_help": peak_r >= 0.5 and final_r < 0.5},
        "trailing_0_7r": {"estimated_effect_r": round(max(0.0, min(peak_r, 0.7) - final_r), 3), "would_help": peak_r >= 0.7 and final_r < 0.7},
        "partial_exit": {"estimated_effect_r": round(max(0.0, min(peak_r, 0.5) - final_r), 3), "would_help": peak_r >= 0.5 and final_r < 0.5},
        "require_vwap_alignment": {"estimated_effect_r": 0.35 if vwap_misaligned else 0.0, "would_help": vwap_misaligned},
        "require_pressure_confirmation": {"estimated_effect_r": 0.35 if ema_misaligned else 0.0, "would_help": ema_misaligned},
        "market_changed_after_entry": {"would_help": bool(live_snapshot.get("market_state_changed")), "estimated_effect_r": 0.2 if live_snapshot.get("market_state_changed") else 0.0},
        "spread_or_execution_damage": {"would_help": bool(live_snapshot.get("spread_widened")), "estimated_effect_r": 0.15 if live_snapshot.get("spread_widened") else 0.0},
    }


def heuristic_classification(snapshot: Dict[str, Any], counterfactuals: Dict[str, Any]) -> Tuple[LossClassification, AIDecisionType, str]:
    signal = snapshot.get("signal_snapshot") or snapshot
    close_snapshot = snapshot.get("close_snapshot") or {}
    live_snapshot = snapshot.get("live_snapshot") or {}
    final_r = float(close_snapshot.get("pnl_r") or 0.0)

    if final_r >= 0:
        return LossClassification.UNKNOWN, AIDecisionType.NO_CHANGE, "Trade did not lose."
    if counterfactuals["spread_or_execution_damage"]["would_help"]:
        return LossClassification.SPREAD_OR_EXECUTION_DAMAGE, AIDecisionType.NO_CHANGE, "Spread widened materially after entry."
    if counterfactuals["wait_one_candle"]["would_help"] or counterfactuals["wait_for_candle_close"]["would_help"]:
        return LossClassification.CORRECT_IDEA_WRONG_TIMING, AIDecisionType.WAIT_FOR_CONFIRMATION, "Setup later moved correctly after early adverse move."
    if counterfactuals["wider_sl"]["would_help"]:
        return LossClassification.CORRECT_DIRECTION_BAD_SL, AIDecisionType.ADJUST_SL_PLACEMENT, "Trade reached favourable excursion after stop-sized noise."
    if counterfactuals["smaller_tp"]["would_help"] or counterfactuals["trailing_0_5r"]["would_help"]:
        return LossClassification.VALID_ENTRY_BAD_TP_OR_EXIT, AIDecisionType.ADJUST_TP_BEHAVIOUR, "Trade found profit but exit plan gave too much back."
    if counterfactuals["market_changed_after_entry"]["would_help"]:
        return LossClassification.MARKET_CHANGED_AFTER_ENTRY, AIDecisionType.NO_CHANGE, "Market regime changed after entry."
    if counterfactuals["require_vwap_alignment"]["would_help"] or counterfactuals["require_pressure_confirmation"]["would_help"]:
        return LossClassification.AVOIDABLE_BAD_TRADE, AIDecisionType.SKIP_SIMILAR_SETUP, "Directional alignment was weak at signal time."
    if not signal.get("rejection_candle") and not signal.get("candle_confirmation"):
        return LossClassification.AVOIDABLE_BAD_TRADE, AIDecisionType.REQUIRE_CANDLE_CLOSE_CONFIRMATION, "No strong confirmation existed at entry."
    return LossClassification.RANDOM_NOISE, AIDecisionType.MARK_PATTERN_AS_RANDOM_NOISE, "Loss does not show a stable corrective pattern."


def build_default_expiry(minutes: int = 30) -> Dict[str, Any]:
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    return {"type": "after_minutes", "value": minutes, "reference": expires_at.isoformat()}
