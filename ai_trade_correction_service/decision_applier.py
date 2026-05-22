from __future__ import annotations

from typing import Any, Dict, List

from .audit_logger import AuditLogger
from .decision_schema import AITradeCorrectionDecision
from .live_rule_store import LiveRuleStore


class DecisionApplier:
    def __init__(self, rule_store: LiveRuleStore, audit_logger: AuditLogger):
        self.rule_store = rule_store
        self.audit_logger = audit_logger

    def apply(self, decision: AITradeCorrectionDecision) -> List[Dict[str, Any]]:
        applied = []
        for modification in decision.live_modifications:
            payload = self.rule_store.apply_modification(modification, decision.analysis_id)
            self.audit_logger.log_live_modification(
                {
                    "timestamp": decision.timestamp,
                    "trigger_type": decision.trigger_type,
                    "analysis_id": decision.analysis_id,
                    "decision": decision.decision.value,
                    "modification": payload,
                    "confidence": modification.confidence,
                    "evidence": modification.evidence_trade_ids,
                }
            )
            applied.append(payload)
        return applied
