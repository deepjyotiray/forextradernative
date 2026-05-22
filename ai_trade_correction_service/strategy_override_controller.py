from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from .decision_schema import AIDecisionType, PreExecutionAction


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class StrategyOverrideController:
    def evaluate_signal(
        self,
        *,
        signal: Dict[str, Any],
        signal_snapshot: Dict[str, Any],
        materialized_rules: Dict[str, List[Dict[str, Any]]],
        similar_trades: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        matched = self._matching_rules(materialized_rules, signal_snapshot)
        signal_updates: Dict[str, Any] = {}

        if matched["setup_skip_rules"] or matched["avoided_patterns"]:
            item = (matched["setup_skip_rules"] or matched["avoided_patterns"])[0]
            return {
                "action": PreExecutionAction.SKIP.value,
                "reason": item.get("rule_description") or "Matched previously avoidable losing pattern.",
                "signal_updates": {},
                "pending": None,
                "matched_modifications": [item],
                "similar_trades": similar_trades[:5],
            }

        if matched["confirmation_requirements"] or matched["setup_wait_rules"]:
            item = (matched["confirmation_requirements"] or matched["setup_wait_rules"])[0]
            pending = self._build_pending_requirements(item, signal_snapshot)
            return {
                "action": pending["wait_action"],
                "reason": item.get("rule_description") or "Pending confirmation required by live AI correction rule.",
                "signal_updates": {},
                "pending": pending,
                "matched_modifications": [item],
                "similar_trades": similar_trades[:5],
            }

        if matched["sl_adjustment_rules"]:
            item = matched["sl_adjustment_rules"][0]
            signal_updates.update(self._apply_sl_adjustment(signal, item, signal_snapshot))
        if matched["tp_adjustment_rules"]:
            item = matched["tp_adjustment_rules"][0]
            signal_updates.update(self._apply_tp_adjustment(signal, item))
        if matched["trailing_exit_rules"]:
            item = matched["trailing_exit_rules"][0]
            signal_updates.setdefault("_ai_feature_overrides", {}).update(dict(item.get("exact_runtime_override_value") or {}))
        if matched["partial_exit_rules"]:
            item = matched["partial_exit_rules"][0]
            signal_updates.setdefault("_ai_feature_overrides", {}).update(dict(item.get("exact_runtime_override_value") or {}))

        if signal_updates:
            return {
                "action": PreExecutionAction.MODIFY_SL.value if "sl" in signal_updates else PreExecutionAction.MODIFY_TP.value,
                "reason": "Applied active AI live override.",
                "signal_updates": signal_updates,
                "pending": None,
                "matched_modifications": [
                    *(matched["sl_adjustment_rules"][:1]),
                    *(matched["tp_adjustment_rules"][:1]),
                    *(matched["trailing_exit_rules"][:1]),
                    *(matched["partial_exit_rules"][:1]),
                ],
                "similar_trades": similar_trades[:5],
            }

        if similar_trades:
            top = similar_trades[0]
            classification = str(top.get("loss_classification") or "")
            if classification == "CORRECT_IDEA_WRONG_TIMING":
                return {
                    "action": PreExecutionAction.WAIT_FOR_CONFIRMATION.value,
                    "reason": "Similar winning idea previously failed on timing; wait for confirmation.",
                    "signal_updates": {},
                    "pending": self._default_pending(signal_snapshot),
                    "matched_modifications": [],
                    "similar_trades": similar_trades[:5],
                }
            if classification == "AVOIDABLE_BAD_TRADE":
                return {
                    "action": PreExecutionAction.SKIP.value,
                    "reason": "Very similar pattern matches an avoidable prior loser.",
                    "signal_updates": {},
                    "pending": None,
                    "matched_modifications": [],
                    "similar_trades": similar_trades[:5],
                }

        return {
            "action": PreExecutionAction.TAKE_NOW.value,
            "reason": "No active AI correction rule blocks this setup.",
            "signal_updates": {},
            "pending": None,
            "matched_modifications": [],
            "similar_trades": similar_trades[:5],
        }

    def check_pending_signal(self, pending_signal: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        conditions = dict(pending_signal.get("conditions") or {})
        signal_snapshot = pending_signal.get("signal_snapshot") or {}
        direction = str(signal_snapshot.get("direction") or "")
        price_vs_ema20 = str((market_data.get("signal_context") or {}).get("price_vs_ema20") or signal_snapshot.get("price_vs_ema20") or "UNKNOWN")
        price_vs_vwap = str((market_data.get("signal_context") or {}).get("price_vs_vwap") or signal_snapshot.get("price_vs_vwap") or "UNKNOWN")
        tick_pressure = market_data.get("tick_pressure") or {}
        bias = str(tick_pressure.get("directional_bias") or "").upper()
        now = datetime.now(timezone.utc)
        expires_at = pending_signal.get("expires_at")
        if expires_at:
            try:
                if now >= datetime.fromisoformat(expires_at):
                    return {"state": "EXPIRED", "reason": "Confirmation window expired."}
            except Exception:
                pass

        if conditions.get("require_ema_alignment"):
            if direction == "BUY" and price_vs_ema20 not in {"ABOVE", "AT_LEVEL"}:
                return {"state": "WAITING", "reason": "Waiting for EMA alignment."}
            if direction == "SELL" and price_vs_ema20 not in {"BELOW", "AT_LEVEL"}:
                return {"state": "WAITING", "reason": "Waiting for EMA alignment."}
        if conditions.get("require_vwap_alignment"):
            if direction == "BUY" and price_vs_vwap not in {"ABOVE", "AT_LEVEL"}:
                return {"state": "WAITING", "reason": "Waiting for VWAP alignment."}
            if direction == "SELL" and price_vs_vwap not in {"BELOW", "AT_LEVEL"}:
                return {"state": "WAITING", "reason": "Waiting for VWAP alignment."}
        if conditions.get("require_microflow_confirmation"):
            if direction == "BUY" and bias != "LONG":
                return {"state": "WAITING", "reason": "Waiting for bullish microflow confirmation."}
            if direction == "SELL" and bias != "SHORT":
                return {"state": "WAITING", "reason": "Waiting for bearish microflow confirmation."}
        if conditions.get("require_candle_confirmation") and not signal_snapshot.get("candle_confirmation"):
            return {"state": "WAITING", "reason": "Waiting for candle close confirmation."}
        if conditions.get("require_retest"):
            zone = signal_snapshot.get("nearest_support_zone") if direction == "BUY" else signal_snapshot.get("nearest_resistance_zone")
            low = float((zone or {}).get("zone_low") or 0.0)
            high = float((zone or {}).get("zone_high") or 0.0)
            bid = float(((market_data.get("tick") or {}).get("bid") or 0.0))
            if not (low <= bid <= high if low and high else False):
                return {"state": "WAITING", "reason": "Waiting for zone retest."}
        return {"state": "READY", "reason": "Confirmation satisfied."}

    def _matching_rules(self, rules: Dict[str, List[Dict[str, Any]]], signal_snapshot: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        result = {key: [] for key in rules}
        for key, items in rules.items():
            for item in items:
                if self._rule_matches(item, signal_snapshot):
                    result[key].append(item)
        return result

    def _rule_matches(self, item: Dict[str, Any], signal_snapshot: Dict[str, Any]) -> bool:
        setup_ok = str(item.get("target_setup_type") or "ANY") in {"ANY", signal_snapshot.get("setup_type")}
        direction_ok = str(item.get("target_direction") or "ANY") in {"ANY", signal_snapshot.get("direction")}
        market_ok = str(item.get("target_market_state") or "ANY") in {"ANY", signal_snapshot.get("market_state")}
        return setup_ok and direction_ok and market_ok

    def _build_pending_requirements(self, item: Dict[str, Any], signal_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        raw_value = dict(item.get("exact_runtime_override_value") or {})
        wait_action = PreExecutionAction.WAIT_FOR_RETEST.value if item.get("action_type") == AIDecisionType.REQUIRE_RETEST.value else PreExecutionAction.WAIT_FOR_CONFIRMATION.value
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=int(raw_value.get("expiry_minutes", 15)))
        return {
            "wait_action": wait_action,
            "conditions": {
                "require_ema_alignment": bool(raw_value.get("require_ema_alignment")),
                "require_vwap_alignment": bool(raw_value.get("require_vwap_alignment")),
                "require_microflow_confirmation": bool(raw_value.get("require_microflow_confirmation")),
                "require_candle_confirmation": bool(raw_value.get("require_candle_confirmation", True)),
                "require_retest": bool(raw_value.get("require_retest")) or item.get("action_type") == AIDecisionType.REQUIRE_RETEST.value,
            },
            "expires_at": expires_at.isoformat(),
            "reason": item.get("rule_description") or "AI correction wait rule",
            "signal_snapshot": copy.deepcopy(signal_snapshot),
        }

    def _default_pending(self, signal_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=15)
        return {
            "wait_action": PreExecutionAction.WAIT_FOR_CONFIRMATION.value,
            "conditions": {
                "require_ema_alignment": False,
                "require_vwap_alignment": False,
                "require_microflow_confirmation": False,
                "require_candle_confirmation": True,
                "require_retest": False,
            },
            "expires_at": expires_at.isoformat(),
            "reason": "Similar losing trade improved when delayed.",
            "signal_snapshot": copy.deepcopy(signal_snapshot),
        }

    def _apply_sl_adjustment(self, signal: Dict[str, Any], item: Dict[str, Any], signal_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        value = dict(item.get("exact_runtime_override_value") or {})
        new_signal: Dict[str, Any] = {"_ai_feature_overrides": {}}
        entry = float(signal.get("entry") or signal_snapshot.get("entry_price") or 0.0)
        sl = float(signal.get("sl") or signal_snapshot.get("proposed_sl") or 0.0)
        zone = signal_snapshot.get("nearest_support_zone") if signal_snapshot.get("direction") == "BUY" else signal_snapshot.get("nearest_resistance_zone")
        if str(value.get("mode") or "") == "zone_edge_buffer" and zone:
            buffer_points = float(value.get("buffer_points") or 0.25)
            if signal_snapshot.get("direction") == "BUY":
                sl = float(zone.get("zone_low") or sl) - buffer_points
            else:
                sl = float(zone.get("zone_high") or sl) + buffer_points
        elif str(value.get("mode") or "") == "wider_by_r":
            current_r = abs(entry - sl)
            widen = float(value.get("multiplier") or 1.2)
            sl = entry - (current_r * widen) if signal_snapshot.get("direction") == "BUY" else entry + (current_r * widen)
        new_signal["sl"] = round(sl, 2)
        new_signal["_ai_feature_overrides"].update(value)
        return new_signal

    def _apply_tp_adjustment(self, signal: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
        value = dict(item.get("exact_runtime_override_value") or {})
        new_signal: Dict[str, Any] = {"_ai_feature_overrides": {}}
        entry = float(signal.get("entry") or 0.0)
        sl = float(signal.get("sl") or 0.0)
        current_tp = float(signal.get("tp") or 0.0)
        risk = abs(entry - sl) if entry and sl else 0.0
        if str(value.get("mode") or "") == "rr_target" and risk > 0:
            rr = float(value.get("rr") or 1.0)
            current_tp = entry + (risk * rr) if str(signal.get("signal") or "").upper() == "BUY" else entry - (risk * rr)
        new_signal["tp"] = round(current_tp, 2)
        new_signal["_ai_feature_overrides"].update(value)
        return new_signal
