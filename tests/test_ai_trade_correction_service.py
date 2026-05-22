from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from ai_trade_correction_service import AITradeCorrectionService
from ai_trade_correction_service.decision_schema import (
    AIDecisionType,
    ExpiryCondition,
    LiveModification,
    RollbackCondition,
    validate_ai_decision,
)
from ai_trade_correction_service.live_rule_store import LiveRuleStore
from ai_trade_correction_service.replay_validator import build_counterfactuals, heuristic_classification
from ai_trade_correction_service.strategy_override_controller import StrategyOverrideController


def _snapshot(direction="SELL", setup_type="SUPPLY", market_state="RANGING", pnl_r=-1.0, peak_r=0.8, mae_r=-1.2):
    return {
        "signal_snapshot": {
            "signal_id": "sig-1",
            "strategy": "M15_ZONE_SCALP",
            "direction": direction,
            "setup_type": setup_type,
            "market_state": market_state,
            "session": "LONDON",
            "price_vs_vwap": "ABOVE" if direction == "SELL" else "BELOW",
            "price_vs_ema20": "ABOVE" if direction == "SELL" else "BELOW",
            "liquidity_sweep": True,
            "candle_confirmation": False,
            "nearest_support_zone": {"zone_low": 100.0, "zone_high": 100.5},
            "nearest_resistance_zone": {"zone_low": 101.0, "zone_high": 101.5},
        },
        "live_snapshot": {
            "mfe_r": peak_r,
            "mae_r": mae_r,
            "price_immediately_moved_against": True,
            "spread_widened": False,
            "market_state_changed": False,
        },
        "close_snapshot": {
            "pnl_r": pnl_r,
            "peak_r": peak_r,
            "mae_r": mae_r,
            "waiting_one_candle_would_have_helped": True,
            "retest_entry_would_have_helped": False,
            "wider_sl_beyond_structure_would_have_helped": mae_r < -1.0,
        },
    }


def test_invalid_ai_output_is_rejected():
    with pytest.raises(ValidationError):
        validate_ai_decision({"analysis_id": "missing-many-fields"})


def test_losing_trade_is_classified_by_dry_run_heuristic():
    snapshot = _snapshot()
    counterfactuals = build_counterfactuals(snapshot)
    classification, decision_type, _ = heuristic_classification(snapshot, counterfactuals)

    assert classification.value == "CORRECT_IDEA_WRONG_TIMING"
    assert decision_type == AIDecisionType.WAIT_FOR_CONFIRMATION


def test_similar_future_signal_is_skipped_or_delayed(tmp_path: Path):
    controller = StrategyOverrideController()
    signal_snapshot = _snapshot()["signal_snapshot"]
    rules = {
        "setup_skip_rules": [],
        "setup_wait_rules": [],
        "sl_adjustment_rules": [],
        "tp_adjustment_rules": [],
        "trailing_exit_rules": [],
        "partial_exit_rules": [],
        "confirmation_requirements": [
            {
                "target_setup_type": "SUPPLY",
                "target_direction": "SELL",
                "target_market_state": "RANGING",
                "action_type": "WAIT_FOR_CONFIRMATION",
                "rule_description": "Wait for confirmation.",
                "exact_runtime_override_value": {"require_candle_confirmation": True, "expiry_minutes": 15},
            }
        ],
        "avoided_patterns": [],
        "preferred_entry_patterns": [],
    }

    result = controller.evaluate_signal(
        signal={"signal": "SELL", "entry": 101.2, "sl": 102.0, "tp": 100.2},
        signal_snapshot=signal_snapshot,
        materialized_rules=rules,
        similar_trades=[],
    )

    assert result["action"] == "WAIT_FOR_CONFIRMATION"


def test_sl_tp_modification_is_applied_to_matching_setup():
    controller = StrategyOverrideController()
    signal_snapshot = _snapshot()["signal_snapshot"]
    rules = {
        "setup_skip_rules": [],
        "setup_wait_rules": [],
        "sl_adjustment_rules": [
            {
                "target_setup_type": "SUPPLY",
                "target_direction": "SELL",
                "target_market_state": "RANGING",
                "action_type": "ADJUST_SL_PLACEMENT",
                "rule_description": "Move stop outside the resistance zone.",
                "exact_runtime_override_value": {"mode": "zone_edge_buffer", "buffer_points": 0.3},
            }
        ],
        "tp_adjustment_rules": [
            {
                "target_setup_type": "SUPPLY",
                "target_direction": "SELL",
                "target_market_state": "RANGING",
                "action_type": "ADJUST_TP_BEHAVIOUR",
                "rule_description": "Reduce TP to 1R.",
                "exact_runtime_override_value": {"mode": "rr_target", "rr": 1.0},
            }
        ],
        "trailing_exit_rules": [],
        "partial_exit_rules": [],
        "confirmation_requirements": [],
        "avoided_patterns": [],
        "preferred_entry_patterns": [],
    }
    signal = {"signal": "SELL", "entry": 101.2, "sl": 102.0, "tp": 99.4}

    result = controller.evaluate_signal(
        signal=signal,
        signal_snapshot=signal_snapshot,
        materialized_rules=rules,
        similar_trades=[],
    )

    assert result["action"] in {"MODIFY_SL", "MODIFY_TP"}
    assert round(float(result["signal_updates"]["sl"]), 2) == 101.8
    assert round(float(result["signal_updates"]["tp"]), 2) == 100.4


def test_expired_modification_is_removed(tmp_path: Path):
    store = LiveRuleStore(tmp_path / "rules.json")
    mod = LiveModification(
        modification_id="mod-expire",
        target_strategy="M15_ZONE_SCALP",
        target_setup_type="SUPPLY",
        target_direction="SELL",
        target_market_state="RANGING",
        action_type=AIDecisionType.WAIT_FOR_CONFIRMATION,
        rule_description="Expire quickly.",
        exact_runtime_override_key="confirmation_requirements",
        exact_runtime_override_value={"require_candle_confirmation": True},
        reason="test",
        evidence_trade_ids=["t1"],
        confidence=0.8,
        expiry=ExpiryCondition(type="after_minutes", value=1, reference=None),
        rollback_condition=RollbackCondition(type="after_n_losses_same_type", value=2, reference=None),
    )
    payload = store.apply_modification(mod, "analysis-1")
    payload["applied_at"] = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
    store.state["active_modifications"] = [payload]
    store._save()

    expired = store.cleanup_expired(current_market_state="RANGING")
    assert len(expired) == 1
    assert store.list_active("M15_ZONE_SCALP") == []


def test_rollback_condition_works(tmp_path: Path):
    store = LiveRuleStore(tmp_path / "rules.json")
    mod = LiveModification(
        modification_id="mod-rollback",
        target_strategy="M15_ZONE_SCALP",
        target_setup_type="SUPPLY",
        target_direction="SELL",
        target_market_state="RANGING",
        action_type=AIDecisionType.WAIT_FOR_CONFIRMATION,
        rule_description="Rollback after two losses.",
        exact_runtime_override_key="confirmation_requirements",
        exact_runtime_override_value={"require_candle_confirmation": True},
        reason="test",
        evidence_trade_ids=["t1"],
        confidence=0.8,
        expiry=ExpiryCondition(type="after_n_trades", value=10, reference=None),
        rollback_condition=RollbackCondition(type="after_n_losses_same_type", value=2, reference=None),
    )
    store.apply_modification(mod, "analysis-1")

    signal_snapshot = _snapshot()["signal_snapshot"]
    close_snapshot = {"pnl_r": -0.8, "pnl": -1.0}
    rolled_one = store.record_trade_result("M15_ZONE_SCALP", signal_snapshot, close_snapshot)
    rolled_two = store.record_trade_result("M15_ZONE_SCALP", signal_snapshot, close_snapshot)

    assert rolled_one == []
    assert len(rolled_two) == 1
    assert store.list_active("M15_ZONE_SCALP") == []


def test_dashboard_status_exposes_recent_analysis_and_modifications(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AI_TRADE_CORRECTION_ENABLED", "1")
    monkeypatch.setenv("AI_TRADE_CORRECTION_DRY_RUN", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    cfg = SimpleNamespace(DEFAULT_STRATEGY=["M15_ZONE_SCALP"])
    strat_mgr = SimpleNamespace(active_name="M15_ZONE_SCALP", selected=["M15_ZONE_SCALP"])
    service = AITradeCorrectionService(tmp_path, cfg, strategy_manager=strat_mgr, log_fn=None)

    mod = LiveModification(
        modification_id="mod-status",
        target_strategy="M15_ZONE_SCALP",
        target_setup_type="SUPPLY",
        target_direction="SELL",
        target_market_state="RANGING",
        action_type=AIDecisionType.WAIT_FOR_CONFIRMATION,
        rule_description="Wait for bearish confirmation.",
        exact_runtime_override_key="confirmation_requirements",
        exact_runtime_override_value={"require_candle_confirmation": True},
        reason="test",
        evidence_trade_ids=["t1"],
        confidence=0.9,
        expiry=ExpiryCondition(type="after_n_trades", value=5, reference=None),
        rollback_condition=RollbackCondition(type="after_n_losses_same_type", value=2, reference=None),
    )
    service.rule_store.apply_modification(mod, "analysis-status")
    service.audit_logger.log_ai_decision(
        {
            "analysis_id": "analysis-status",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trigger_type": "AFTER_CLOSED_TRADE",
            "strategy": "M15_ZONE_SCALP",
            "trade_ids_reviewed": ["t1"],
            "decision": "WAIT_FOR_CONFIRMATION",
            "diagnosis_summary": "Entry was early; wait for confirmation next time.",
        }
    )
    service.audit_logger.log_live_modification(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "APPLIED",
            "modification_id": "mod-status",
            "reason": "Wait for bearish confirmation.",
        }
    )
    service.audit_logger.log_trade_forensic(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal_id": "sig-status",
            "trade_ticket": 123,
            "classification": "CORRECT_IDEA_WRONG_TIMING",
        }
    )
    service._set_analysis_status(
        running=False,
        last_trigger="AFTER_CLOSED_TRADE",
        last_status="COMPLETED",
        last_decision="WAIT_FOR_CONFIRMATION",
        last_analysis_id="analysis-status",
    )

    status = service.get_dashboard_status(include_recent=True, limit=5)

    assert status["active"] is True
    assert status["mode"] == "DRY_RUN"
    assert status["counts"]["active_modifications"] == 1
    assert status["analysis_state"]["last_analysis_id"] == "analysis-status"
    assert status["recent_analyses"][0]["analysis_id"] == "analysis-status"
    assert status["recent_modifications"][0]["modification_id"] == "mod-status"
    assert status["recent_forensics"][0]["classification"] == "CORRECT_IDEA_WRONG_TIMING"
