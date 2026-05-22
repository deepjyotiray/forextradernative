from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .audit_logger import AuditLogger
from .config_adapter import ServiceRuntimeConfig
from .decision_schema import (
    AIDecisionType,
    AITradeCorrectionDecision,
    build_response_schema,
    validate_ai_decision,
    validation_error_to_text,
)
from .replay_validator import build_counterfactuals, build_default_expiry, heuristic_classification


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_response_text(payload: Dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    parts = []
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            text = str(content.get("text") or "").strip()
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


class AIForensicClient:
    def __init__(self, runtime_config: ServiceRuntimeConfig, audit_logger: AuditLogger):
        self.runtime_config = runtime_config
        self.audit_logger = audit_logger

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if self.runtime_config.dry_run:
            return self._heuristic_decision(context)
        if not self.runtime_config.api_key_present:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        return self._request_openai_decision(context)

    def _heuristic_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        target_trade = (context.get("recent_trades") or [{}])[0]
        counterfactuals = build_counterfactuals(target_trade)
        classification, decision_type, evidence_text = heuristic_classification(target_trade, counterfactuals)
        signal_snapshot = target_trade.get("signal_snapshot") or {}
        trade_id = str((target_trade.get("close_snapshot") or {}).get("trade_ticket") or target_trade.get("trade_ticket") or signal_snapshot.get("signal_id") or "dry-run-trade")

        modifications = []
        if decision_type != AIDecisionType.NO_CHANGE and decision_type != AIDecisionType.MARK_PATTERN_AS_RANDOM_NOISE:
            modifications.append(
                {
                    "modification_id": f"dry-run-{decision_type.value.lower()}-{trade_id}",
                    "target_strategy": str(context.get("strategy") or signal_snapshot.get("strategy") or ""),
                    "target_setup_type": str(signal_snapshot.get("setup_type") or "ANY"),
                    "target_direction": str(signal_snapshot.get("direction") or "ANY"),
                    "target_market_state": str(signal_snapshot.get("market_state") or "ANY"),
                    "action_type": decision_type.value,
                    "rule_description": evidence_text,
                    "exact_runtime_override_key": self._override_key_for_decision(decision_type),
                    "exact_runtime_override_value": self._override_value_for_decision(decision_type, signal_snapshot),
                    "reason": evidence_text,
                    "evidence_trade_ids": [trade_id],
                    "confidence": 0.82,
                    "expiry": build_default_expiry(30),
                    "rollback_condition": {"type": "after_n_losses_same_type", "value": 2, "reference": None},
                }
            )

        return {
            "analysis_id": f"dry-run-{int(datetime.now(timezone.utc).timestamp())}",
            "timestamp": _utc_now_iso(),
            "trigger_type": str(context.get("trigger_type") or "dry_run"),
            "strategy": str(context.get("strategy") or signal_snapshot.get("strategy") or ""),
            "trade_ids_reviewed": [trade_id],
            "market_window_reviewed": dict(context.get("market_window") or {}),
            "diagnosis_summary": evidence_text,
            "loss_classifications": [
                {
                    "trade_id": trade_id,
                    "classification": classification.value,
                    "evidence": [evidence_text],
                    "confidence": 0.82,
                }
            ],
            "repeated_patterns_found": [f"{classification.value}:{signal_snapshot.get('setup_type') or 'UNKNOWN'}"],
            "counterfactual_findings": [
                {
                    "scenario": name,
                    "outcome": "IMPROVES" if payload.get("would_help") else "NO_MATERIAL_EDGE",
                    "estimated_effect_r": float(payload.get("estimated_effect_r") or 0.0),
                    "supports_decision": bool(payload.get("would_help")),
                }
                for name, payload in counterfactuals.items()
            ],
            "decision": decision_type.value,
            "live_modifications": modifications,
            "confidence": 0.82,
            "evidence": [evidence_text],
            "expected_effect": "Delay, skip, or reshape only similar future setups for the single enabled strategy.",
            "expiry_condition": build_default_expiry(30),
            "rollback_condition": {"type": "after_n_losses_same_type", "value": 2, "reference": None},
            "audit_note": "Dry-run heuristic decision generated without an external API call.",
        }

    def _request_openai_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        system_text = (
            "You are an AI trade forensics service for one enabled trading strategy. "
            "Use only the provided JSON context. "
            "Return a strict JSON object matching the supplied schema. "
            "Never output prose outside the schema. "
            "You may choose only from the allowed decision types. "
            "Do not suggest global risk reduction, strategy disabling, or multi-strategy switching."
        )
        payload = {
            "model": self.runtime_config.model,
            "instructions": system_text,
            "input": json.dumps(context, separators=(",", ":"), ensure_ascii=True),
            "temperature": 0.1,
            "max_output_tokens": 1800,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "ai_trade_correction_decision",
                    "strict": True,
                    "schema": build_response_schema(),
                }
            },
        }
        self.audit_logger.log_ai_metadata(
            {
                "timestamp": _utc_now_iso(),
                "event": "request",
                "provider": "openai",
                "model": self.runtime_config.model,
                "context_summary": {
                    "trigger_type": context.get("trigger_type"),
                    "strategy": context.get("strategy"),
                    "trade_count": len(context.get("recent_trades") or []),
                },
            }
        )
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.runtime_config.api_base}/responses",
            data=body,
            headers={
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '').strip()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.runtime_config.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI API connection failed: {exc.reason}") from exc

        text = _extract_response_text(response_payload)
        if not text:
            raise RuntimeError("OpenAI API returned no structured text output.")
        self.audit_logger.log_ai_metadata(
            {
                "timestamp": _utc_now_iso(),
                "event": "response",
                "provider": "openai",
                "model": self.runtime_config.model,
                "raw_text": text,
            }
        )
        try:
            payload = json.loads(text)
            decision = validate_ai_decision(payload)
            return decision.model_dump(mode="json")
        except Exception as exc:
            detail = validation_error_to_text(exc) if hasattr(exc, "errors") else str(exc)
            raise RuntimeError(f"Structured AI payload validation failed: {detail}") from exc

    @staticmethod
    def _override_key_for_decision(decision_type: AIDecisionType) -> str:
        mapping = {
            AIDecisionType.SKIP_SIMILAR_SETUP: "setup_skip_rules",
            AIDecisionType.MARK_PATTERN_AS_AVOIDABLE: "avoided_patterns",
            AIDecisionType.WAIT_FOR_CONFIRMATION: "confirmation_requirements",
            AIDecisionType.REQUIRE_RETEST: "setup_wait_rules",
            AIDecisionType.REQUIRE_SWEEP_RECLAIM: "confirmation_requirements",
            AIDecisionType.REQUIRE_CANDLE_CLOSE_CONFIRMATION: "confirmation_requirements",
            AIDecisionType.REQUIRE_VWAP_ALIGNMENT: "confirmation_requirements",
            AIDecisionType.REQUIRE_EMA_ALIGNMENT: "confirmation_requirements",
            AIDecisionType.REQUIRE_MICROFLOW_CONFIRMATION: "confirmation_requirements",
            AIDecisionType.ADJUST_SL_PLACEMENT: "sl_adjustment_rules",
            AIDecisionType.ADJUST_TP_BEHAVIOUR: "tp_adjustment_rules",
            AIDecisionType.ENABLE_TRAILING_FOR_SETUP: "trailing_exit_rules",
            AIDecisionType.ENABLE_PARTIAL_EXIT_FOR_SETUP: "partial_exit_rules",
            AIDecisionType.MARK_PATTERN_AS_TAKE_LATER: "preferred_entry_patterns",
            AIDecisionType.MARK_PATTERN_AS_VALID_BUT_BAD_EXIT: "tp_adjustment_rules",
        }
        return mapping.get(decision_type, "confirmation_requirements")

    @staticmethod
    def _override_value_for_decision(decision_type: AIDecisionType, signal_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        if decision_type in {AIDecisionType.WAIT_FOR_CONFIRMATION, AIDecisionType.REQUIRE_CANDLE_CLOSE_CONFIRMATION}:
            return {"require_candle_confirmation": True, "expiry_minutes": 15}
        if decision_type == AIDecisionType.REQUIRE_RETEST:
            return {"require_retest": True, "require_candle_confirmation": True, "expiry_minutes": 15}
        if decision_type == AIDecisionType.REQUIRE_SWEEP_RECLAIM:
            return {"require_retest": True, "require_candle_confirmation": True, "expiry_minutes": 20}
        if decision_type == AIDecisionType.REQUIRE_VWAP_ALIGNMENT:
            return {"require_vwap_alignment": True, "require_candle_confirmation": True, "expiry_minutes": 15}
        if decision_type == AIDecisionType.REQUIRE_EMA_ALIGNMENT:
            return {"require_ema_alignment": True, "require_candle_confirmation": True, "expiry_minutes": 15}
        if decision_type == AIDecisionType.REQUIRE_MICROFLOW_CONFIRMATION:
            return {"require_microflow_confirmation": True, "require_candle_confirmation": True, "expiry_minutes": 10}
        if decision_type == AIDecisionType.ADJUST_SL_PLACEMENT:
            return {"mode": "zone_edge_buffer", "buffer_points": 0.25}
        if decision_type == AIDecisionType.ADJUST_TP_BEHAVIOUR:
            return {"mode": "rr_target", "rr": 1.0}
        if decision_type == AIDecisionType.ENABLE_TRAILING_FOR_SETUP:
            return {"trail_activate_r": 0.5, "trail_lock_r": 0.15}
        if decision_type == AIDecisionType.ENABLE_PARTIAL_EXIT_FOR_SETUP:
            return {"profit_lock_1_arm_r": 0.5, "profit_lock_1_r": 0.2}
        return {"note": signal_snapshot.get("setup_type") or "pattern_memory"}
