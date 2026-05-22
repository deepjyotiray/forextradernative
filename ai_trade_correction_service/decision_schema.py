from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class LossClassification(str, Enum):
    AVOIDABLE_BAD_TRADE = "AVOIDABLE_BAD_TRADE"
    CORRECT_IDEA_WRONG_TIMING = "CORRECT_IDEA_WRONG_TIMING"
    CORRECT_DIRECTION_BAD_SL = "CORRECT_DIRECTION_BAD_SL"
    VALID_ENTRY_BAD_TP_OR_EXIT = "VALID_ENTRY_BAD_TP_OR_EXIT"
    MARKET_CHANGED_AFTER_ENTRY = "MARKET_CHANGED_AFTER_ENTRY"
    SPREAD_OR_EXECUTION_DAMAGE = "SPREAD_OR_EXECUTION_DAMAGE"
    RANDOM_NOISE = "RANDOM_NOISE"
    UNKNOWN = "UNKNOWN"


class AIDecisionType(str, Enum):
    NO_CHANGE = "NO_CHANGE"
    SKIP_SIMILAR_SETUP = "SKIP_SIMILAR_SETUP"
    WAIT_FOR_CONFIRMATION = "WAIT_FOR_CONFIRMATION"
    REQUIRE_RETEST = "REQUIRE_RETEST"
    REQUIRE_SWEEP_RECLAIM = "REQUIRE_SWEEP_RECLAIM"
    REQUIRE_CANDLE_CLOSE_CONFIRMATION = "REQUIRE_CANDLE_CLOSE_CONFIRMATION"
    REQUIRE_VWAP_ALIGNMENT = "REQUIRE_VWAP_ALIGNMENT"
    REQUIRE_EMA_ALIGNMENT = "REQUIRE_EMA_ALIGNMENT"
    REQUIRE_MICROFLOW_CONFIRMATION = "REQUIRE_MICROFLOW_CONFIRMATION"
    ADJUST_ENTRY_TIMING = "ADJUST_ENTRY_TIMING"
    ADJUST_SL_PLACEMENT = "ADJUST_SL_PLACEMENT"
    ADJUST_TP_BEHAVIOUR = "ADJUST_TP_BEHAVIOUR"
    ENABLE_TRAILING_FOR_SETUP = "ENABLE_TRAILING_FOR_SETUP"
    ENABLE_PARTIAL_EXIT_FOR_SETUP = "ENABLE_PARTIAL_EXIT_FOR_SETUP"
    MARK_PATTERN_AS_AVOIDABLE = "MARK_PATTERN_AS_AVOIDABLE"
    MARK_PATTERN_AS_TAKE_LATER = "MARK_PATTERN_AS_TAKE_LATER"
    MARK_PATTERN_AS_VALID_BUT_BAD_EXIT = "MARK_PATTERN_AS_VALID_BUT_BAD_EXIT"
    MARK_PATTERN_AS_RANDOM_NOISE = "MARK_PATTERN_AS_RANDOM_NOISE"


class PreExecutionAction(str, Enum):
    TAKE_NOW = "TAKE_NOW"
    WAIT_FOR_CONFIRMATION = "WAIT_FOR_CONFIRMATION"
    WAIT_FOR_RETEST = "WAIT_FOR_RETEST"
    MODIFY_SL = "MODIFY_SL"
    MODIFY_TP = "MODIFY_TP"
    USE_TRAILING = "USE_TRAILING"
    SKIP = "SKIP"


class ExpiryCondition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: str
    value: Optional[float | int | str] = None
    reference: Optional[str] = None


class RollbackCondition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: str
    value: Optional[float | int | str] = None
    reference: Optional[str] = None


class LossClassificationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    trade_id: str
    classification: LossClassification
    evidence: List[str] = Field(default_factory=list)
    confidence: float


class CounterfactualFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    scenario: str
    outcome: str
    estimated_effect_r: float
    supports_decision: bool


class LiveModification(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    modification_id: str
    target_strategy: str
    target_setup_type: str
    target_direction: str
    target_market_state: str
    action_type: AIDecisionType
    rule_description: str
    exact_runtime_override_key: str
    exact_runtime_override_value: Any
    reason: str
    evidence_trade_ids: List[str] = Field(default_factory=list)
    confidence: float
    expiry: ExpiryCondition
    rollback_condition: RollbackCondition


class AITradeCorrectionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    analysis_id: str
    timestamp: str
    trigger_type: str
    strategy: str
    trade_ids_reviewed: List[str] = Field(default_factory=list)
    market_window_reviewed: Dict[str, Any] = Field(default_factory=dict)
    diagnosis_summary: str
    loss_classifications: List[LossClassificationRecord] = Field(default_factory=list)
    repeated_patterns_found: List[str] = Field(default_factory=list)
    counterfactual_findings: List[CounterfactualFinding] = Field(default_factory=list)
    decision: AIDecisionType
    live_modifications: List[LiveModification] = Field(default_factory=list)
    confidence: float
    evidence: List[str] = Field(default_factory=list)
    expected_effect: str
    expiry_condition: ExpiryCondition
    rollback_condition: RollbackCondition
    audit_note: str


def validate_ai_decision(payload: Dict[str, Any]) -> AITradeCorrectionDecision:
    return AITradeCorrectionDecision.model_validate(payload)


def validation_error_to_text(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(item) for item in err.get("loc", []))
        parts.append(f"{loc}: {err.get('msg')}")
    return "; ".join(parts)


def build_live_modification_schema() -> Dict[str, Any]:
    expiry_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "type": {"type": "string"},
            "value": {
                "anyOf": [
                    {"type": "number"},
                    {"type": "integer"},
                    {"type": "string"},
                    {"type": "null"},
                ]
            },
            "reference": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["type", "value", "reference"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "modification_id": {"type": "string"},
            "target_strategy": {"type": "string"},
            "target_setup_type": {"type": "string"},
            "target_direction": {"type": "string"},
            "target_market_state": {"type": "string"},
            "action_type": {"type": "string", "enum": [member.value for member in AIDecisionType]},
            "rule_description": {"type": "string"},
            "exact_runtime_override_key": {"type": "string"},
            "exact_runtime_override_value": {
                "anyOf": [
                    {"type": "object", "additionalProperties": True},
                    {"type": "array"},
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "integer"},
                    {"type": "boolean"},
                    {"type": "null"},
                ]
            },
            "reason": {"type": "string"},
            "evidence_trade_ids": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
            "expiry": expiry_schema,
            "rollback_condition": expiry_schema,
        },
        "required": [
            "modification_id",
            "target_strategy",
            "target_setup_type",
            "target_direction",
            "target_market_state",
            "action_type",
            "rule_description",
            "exact_runtime_override_key",
            "exact_runtime_override_value",
            "reason",
            "evidence_trade_ids",
            "confidence",
            "expiry",
            "rollback_condition",
        ],
    }


def build_response_schema() -> Dict[str, Any]:
    expiry_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "type": {"type": "string"},
            "value": {
                "anyOf": [
                    {"type": "number"},
                    {"type": "integer"},
                    {"type": "string"},
                    {"type": "null"},
                ]
            },
            "reference": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["type", "value", "reference"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "analysis_id": {"type": "string"},
            "timestamp": {"type": "string"},
            "trigger_type": {"type": "string"},
            "strategy": {"type": "string"},
            "trade_ids_reviewed": {"type": "array", "items": {"type": "string"}},
            "market_window_reviewed": {"type": "object", "additionalProperties": True},
            "diagnosis_summary": {"type": "string"},
            "loss_classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "trade_id": {"type": "string"},
                        "classification": {"type": "string", "enum": [member.value for member in LossClassification]},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "number"},
                    },
                    "required": ["trade_id", "classification", "evidence", "confidence"],
                },
            },
            "repeated_patterns_found": {"type": "array", "items": {"type": "string"}},
            "counterfactual_findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "scenario": {"type": "string"},
                        "outcome": {"type": "string"},
                        "estimated_effect_r": {"type": "number"},
                        "supports_decision": {"type": "boolean"},
                    },
                    "required": ["scenario", "outcome", "estimated_effect_r", "supports_decision"],
                },
            },
            "decision": {"type": "string", "enum": [member.value for member in AIDecisionType]},
            "live_modifications": {"type": "array", "items": build_live_modification_schema()},
            "confidence": {"type": "number"},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "expected_effect": {"type": "string"},
            "expiry_condition": expiry_schema,
            "rollback_condition": expiry_schema,
            "audit_note": {"type": "string"},
        },
        "required": [
            "analysis_id",
            "timestamp",
            "trigger_type",
            "strategy",
            "trade_ids_reviewed",
            "market_window_reviewed",
            "diagnosis_summary",
            "loss_classifications",
            "repeated_patterns_found",
            "counterfactual_findings",
            "decision",
            "live_modifications",
            "confidence",
            "evidence",
            "expected_effect",
            "expiry_condition",
            "rollback_condition",
            "audit_note",
        ],
    }
