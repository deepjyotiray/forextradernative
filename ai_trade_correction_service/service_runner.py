from __future__ import annotations

import copy
import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ai_forensic_client import AIForensicClient
from .audit_logger import AuditLogger
from .config_adapter import load_service_runtime_config, resolve_enabled_strategy, session_is_active
from .decision_applier import DecisionApplier
from .decision_schema import validate_ai_decision
from .live_rule_store import LiveRuleStore
from .recent_trade_window import RecentTradeWindow
from .replay_validator import build_counterfactuals, heuristic_classification
from .strategy_override_controller import StrategyOverrideController
from .trade_snapshot_builder import (
    build_close_trade_snapshot,
    build_open_trade_snapshot,
    build_signal_snapshot,
    update_open_trade_snapshot,
)


class AITradeCorrectionService:
    def __init__(self, base_dir: str | Path, cfg_module, strategy_manager=None, log_fn=None):
        self.base_dir = Path(base_dir)
        self.cfg_module = cfg_module
        self.strategy_manager = strategy_manager
        self.log_fn = log_fn
        self.runtime_config = load_service_runtime_config()
        self.enabled_strategy_info = resolve_enabled_strategy(cfg_module, strategy_manager)
        self.store = RecentTradeWindow(self.base_dir / "ai_trade_correction.db")
        self.audit_logger = AuditLogger(self.base_dir)
        self.rule_store = LiveRuleStore(self.base_dir / "ai_trade_correction_live_rules.json")
        self.override_controller = StrategyOverrideController()
        self.ai_client = AIForensicClient(self.runtime_config, self.audit_logger)
        self.decision_applier = DecisionApplier(self.rule_store, self.audit_logger)
        self._analysis_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._last_15m_bucket = ""
        self._last_session = ""
        self._last_day = datetime.now(timezone.utc).date().isoformat()
        self._analysis_status = {
            "running": False,
            "current_trigger": "",
            "current_started_at": "",
            "last_trigger": "",
            "last_started_at": "",
            "last_completed_at": "",
            "last_status": "IDLE",
            "last_error": "",
            "last_decision": "",
            "last_analysis_id": "",
            "last_trade_ids_reviewed": [],
            "last_modification_ids": [],
        }

    @property
    def is_active(self) -> bool:
        return bool(self.runtime_config.enabled and self.enabled_strategy_info.get("single_strategy_mode"))

    def capture_signal(self, strategy_name: str, signal: Dict[str, Any], market_data: Dict[str, Any], setup_signature: str) -> str:
        if not self.is_active:
            return ""
        if str(strategy_name or "").upper() != str(self.enabled_strategy_info.get("enabled_strategy") or "").upper():
            return ""
        snapshot = build_signal_snapshot(strategy_name, signal, market_data, setup_signature)
        return self.store.upsert_signal_snapshot(snapshot)

    def update_signal_status(self, signal_id: str, *, status: str, reason: str = "", signal_snapshot: Optional[Dict[str, Any]] = None) -> None:
        if signal_id:
            self.store.update_signal_status(signal_id, status=status, reason=reason, signal_snapshot=signal_snapshot)

    def pre_execution_check(
        self,
        *,
        strategy_name: str,
        signal_id: str,
        signal: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self.is_active:
            return {"action": "TAKE_NOW", "reason": "Service inactive.", "signal": signal, "signal_snapshot": {}, "matched_modifications": []}
        if str(strategy_name or "").upper() != str(self.enabled_strategy_info.get("enabled_strategy") or "").upper():
            return {"action": "TAKE_NOW", "reason": "Signal does not belong to enabled strategy.", "signal": signal, "signal_snapshot": {}, "matched_modifications": []}

        stored = self.store.get_signal_snapshot(signal_id) or {}
        signal_snapshot = copy.deepcopy(stored.get("signal_snapshot") or {})
        similar = self.store.find_similar_trades(strategy_name, signal_snapshot, limit=10)
        rules = self.rule_store.materialize(strategy_name)
        evaluation = self.override_controller.evaluate_signal(
            signal=signal,
            signal_snapshot=signal_snapshot,
            materialized_rules=rules,
            similar_trades=similar,
        )
        updated_signal = copy.deepcopy(signal)
        updated_signal.update(evaluation.get("signal_updates") or {})
        if evaluation.get("pending"):
            pending = evaluation["pending"]
            self.store.add_pending_signal(
                signal_id=signal_id,
                strategy=strategy_name,
                setup_type=str(signal_snapshot.get("setup_type") or ""),
                direction=str(signal_snapshot.get("direction") or ""),
                expires_at=pending.get("expires_at"),
                wait_action=pending.get("wait_action"),
                conditions=pending.get("conditions") or {},
                signal_snapshot=signal_snapshot,
                signal_payload=updated_signal,
            )
        return {
            "action": evaluation.get("action"),
            "reason": evaluation.get("reason"),
            "signal": updated_signal,
            "signal_snapshot": signal_snapshot,
            "matched_modifications": evaluation.get("matched_modifications") or [],
            "similar_trades": evaluation.get("similar_trades") or [],
        }

    def release_pending_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        strategy_name = str(self.enabled_strategy_info.get("enabled_strategy") or "")
        if not self.is_active or not strategy_name:
            return []
        current_context = build_signal_snapshot(strategy_name, {"signal": "NO_TRADE"}, market_data, "pending-context")
        enriched_market_data = dict(market_data)
        enriched_market_data["signal_context"] = current_context
        released = []
        for pending in self.store.list_pending_signals(strategy_name):
            state = self.override_controller.check_pending_signal(pending, enriched_market_data)
            if state.get("state") == "READY":
                payload = dict(pending.get("signal_payload") or {})
                payload["_ai_pending_released"] = True
                payload["reason"] = f"{payload.get('reason', '')} | AI confirmation released".strip()
                released.append({"strategy": strategy_name, "signal": payload})
                self.store.resolve_pending_signal(pending["signal_id"], "RELEASED")
                self.store.update_signal_status(pending["signal_id"], status="PENDING_RELEASED", reason=state.get("reason", "Released"))
            elif state.get("state") == "EXPIRED":
                self.store.resolve_pending_signal(pending["signal_id"], "EXPIRED")
                self.store.update_signal_status(pending["signal_id"], status="PENDING_EXPIRED", reason=state.get("reason", "Expired"))
        return released

    def on_trade_opened(self, signal_id: str, signal: Dict[str, Any], trade_ticket: int, fill_price: float) -> None:
        if not signal_id:
            return
        snapshot = (self.store.get_signal_snapshot(signal_id) or {}).get("signal_snapshot") or {}
        live_snapshot = build_open_trade_snapshot(snapshot, signal, trade_ticket, fill_price)
        self.store.attach_trade_open(signal_id, trade_ticket, live_snapshot)

    def on_cycle(self, market_data: Dict[str, Any], open_trades: Dict[int, Any]) -> None:
        strategy_name = str(self.enabled_strategy_info.get("enabled_strategy") or "")
        if not strategy_name:
            return
        self.rule_store.cleanup_expired(current_market_state=str((market_data.get("regime") or {}).get("state") or ""))
        for ticket, trade in (open_trades or {}).items():
            snapshot = self.store.get_snapshot_by_ticket(int(ticket))
            if not snapshot:
                continue
            live_snapshot = update_open_trade_snapshot(snapshot.get("live_snapshot") or {}, trade, market_data)
            self.store.update_live_snapshot(snapshot["signal_id"], live_snapshot)

        self._check_periodic_triggers(strategy_name, market_data)

    def on_trade_closed(self, trade_ticket: int, db_order: Dict[str, Any], mt5_trade: Dict[str, Any]) -> None:
        snapshot = self.store.get_snapshot_by_ticket(int(trade_ticket))
        if not snapshot:
            return
        close_snapshot = build_close_trade_snapshot(
            snapshot.get("signal_snapshot") or {},
            snapshot.get("live_snapshot") or {},
            db_order or {},
            mt5_trade or {},
        )
        counterfactuals = build_counterfactuals({"signal_snapshot": snapshot.get("signal_snapshot"), "live_snapshot": snapshot.get("live_snapshot"), "close_snapshot": close_snapshot})
        classification, _, explanation = heuristic_classification({"signal_snapshot": snapshot.get("signal_snapshot"), "live_snapshot": snapshot.get("live_snapshot"), "close_snapshot": close_snapshot}, counterfactuals)
        self.store.close_trade_snapshot(
            snapshot["signal_id"],
            close_snapshot=close_snapshot,
            counterfactuals=counterfactuals,
            loss_classification=classification.value,
        )
        self.audit_logger.log_trade_forensic(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "signal_id": snapshot["signal_id"],
                "trade_ticket": trade_ticket,
                "strategy": snapshot.get("strategy"),
                "classification": classification.value,
                "explanation": explanation,
                "counterfactuals": counterfactuals,
            }
        )
        rolled_back = self.rule_store.record_trade_result(snapshot.get("strategy") or "", snapshot.get("signal_snapshot") or {}, close_snapshot)
        if rolled_back:
            for item in rolled_back:
                self.audit_logger.log_live_modification(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "event": "ROLLBACK",
                        "modification_id": item.get("modification_id"),
                        "reason": item.get("rollback_condition"),
                    }
                )
        self._schedule_analysis("AFTER_CLOSED_TRADE", strategy_name=snapshot.get("strategy") or "")
        losses = self.store.get_consecutive_losses(snapshot.get("strategy") or "", limit=3)
        if len(losses) >= 2:
            self._schedule_analysis("AFTER_2_CONSECUTIVE_LOSSES", strategy_name=snapshot.get("strategy") or "")
        if self.store.count_closed(snapshot.get("strategy") or "") % 5 == 0:
            self._schedule_analysis("AFTER_5_CLOSED_TRADES", strategy_name=snapshot.get("strategy") or "")

    def _check_periodic_triggers(self, strategy_name: str, market_data: Dict[str, Any]) -> None:
        now = market_data.get("now_utc") or datetime.now(timezone.utc)
        session_label = str(market_data.get("session") or "").upper()
        bucket = now.strftime("%Y-%m-%dT%H:%M")
        bucket_15m = bucket[:-1] + "0"
        if session_is_active(session_label) and bucket_15m != self._last_15m_bucket and now.minute % 15 == 0:
            self._last_15m_bucket = bucket_15m
            self._schedule_analysis("EVERY_15_MINUTES", strategy_name=strategy_name, market_window={"session": session_label, "bucket": bucket_15m})
        if self._last_session and self._last_session != session_label:
            self._schedule_analysis("END_OF_SESSION", strategy_name=strategy_name, market_window={"previous_session": self._last_session, "new_session": session_label})
        self._last_session = session_label
        today = now.date().isoformat()
        if self._last_day != today:
            self._schedule_analysis("END_OF_DAY", strategy_name=strategy_name, market_window={"previous_day": self._last_day, "new_day": today})
            self._last_day = today

    def _schedule_analysis(self, trigger_type: str, *, strategy_name: str, market_window: Optional[Dict[str, Any]] = None) -> None:
        if not self.is_active:
            return
        worker = threading.Thread(
            target=self._run_analysis_if_idle,
            kwargs={"trigger_type": trigger_type, "strategy_name": strategy_name, "market_window": market_window or {}},
            daemon=True,
            name=f"AITradeCorrection-{trigger_type}",
        )
        worker.start()

    def _run_analysis_if_idle(self, *, trigger_type: str, strategy_name: str, market_window: Dict[str, Any]) -> None:
        if not self._analysis_lock.acquire(blocking=False):
            return
        try:
            self.run_analysis_now(trigger_type=trigger_type, strategy_name=strategy_name, market_window=market_window)
        finally:
            self._analysis_lock.release()

    def run_analysis_now(self, *, trigger_type: str, strategy_name: str, market_window: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        recent_trades = self.store.get_recent_closed(strategy_name, limit=5)
        if not recent_trades:
            return None
        started_at = datetime.now(timezone.utc).isoformat()
        self._set_analysis_status(
            running=True,
            current_trigger=trigger_type,
            current_started_at=started_at,
            last_trigger=trigger_type,
            last_started_at=started_at,
            last_status="RUNNING",
            last_error="",
        )
        context = {
            "trigger_type": trigger_type,
            "strategy": strategy_name,
            "market_window": market_window or {},
            "recent_trades": recent_trades,
            "similar_winners": [row for row in self.store.find_similar_trades(strategy_name, (recent_trades[0].get("signal_snapshot") or {}), limit=5) if float(((row.get("close_snapshot") or {}).get("pnl") or 0.0)) > 0],
            "similar_losers": [row for row in self.store.find_similar_trades(strategy_name, (recent_trades[0].get("signal_snapshot") or {}), limit=5) if float(((row.get("close_snapshot") or {}).get("pnl") or 0.0)) <= 0],
            "active_overrides": self.rule_store.list_active(strategy_name),
        }
        try:
            payload = self.ai_client.analyze(context)
            decision = validate_ai_decision(payload)
        except Exception as exc:
            self._set_analysis_status(
                running=False,
                current_trigger="",
                current_started_at="",
                last_completed_at=datetime.now(timezone.utc).isoformat(),
                last_status="FAILED",
                last_error=str(exc),
            )
            if self.log_fn:
                self.log_fn("AI_CORRECT", f"Analysis skipped: {exc}")
            return None

        applied = self.decision_applier.apply(decision)
        self.audit_logger.log_ai_decision(decision.model_dump(mode="json"))
        self._set_analysis_status(
            running=False,
            current_trigger="",
            current_started_at="",
            last_completed_at=datetime.now(timezone.utc).isoformat(),
            last_status="COMPLETED",
            last_error="",
            last_decision=decision.decision.value,
            last_analysis_id=decision.analysis_id,
            last_trade_ids_reviewed=list(decision.trade_ids_reviewed or []),
            last_modification_ids=[item.get("modification_id", "") for item in applied if item.get("modification_id")],
        )
        self._refresh_daily_report(strategy_name, applied)
        return decision.model_dump(mode="json")

    def get_dashboard_status(self, *, include_recent: bool = True, limit: int = 8) -> Dict[str, Any]:
        strategy_name = str(self.enabled_strategy_info.get("enabled_strategy") or "")
        with self._status_lock:
            analysis_state = dict(self._analysis_status)

        payload = {
            "enabled": bool(self.runtime_config.enabled),
            "active": bool(self.is_active),
            "mode": "DRY_RUN" if self.runtime_config.dry_run else "OPENAI",
            "api_configured": bool(self.runtime_config.api_key_present),
            "fail_open": bool(self.runtime_config.fail_open),
            "enabled_strategy": strategy_name,
            "single_strategy_mode": bool(self.enabled_strategy_info.get("single_strategy_mode")),
            "reason": str(self.enabled_strategy_info.get("reason") or ""),
            "analysis_state": analysis_state,
            "counts": {
                "closed_trades": self.store.count_closed(strategy_name) if strategy_name else 0,
                "pending_signals": len(self.store.list_pending_signals(strategy_name)) if strategy_name else 0,
                "active_modifications": len(self.rule_store.list_active(strategy_name)) if strategy_name else 0,
                "rolled_back_modifications": len(self.rule_store.state.get("rolled_back", []) or []),
            },
            "active_modifications": self.rule_store.list_active(strategy_name)[:5] if strategy_name else [],
        }
        if include_recent:
            payload["recent_analyses"] = self._read_recent_jsonl(self.audit_logger.ai_decisions_path, limit=limit)
            payload["recent_modifications"] = self._read_recent_jsonl(self.audit_logger.live_modifications_path, limit=limit)
            payload["recent_forensics"] = self._read_recent_jsonl(self.audit_logger.trade_forensics_path, limit=limit)
        else:
            payload["recent_analyses"] = []
            payload["recent_modifications"] = []
            payload["recent_forensics"] = []
        return payload

    def _set_analysis_status(self, **updates: Any) -> None:
        with self._status_lock:
            self._analysis_status.update({k: v for k, v in updates.items() if v is not None})

    def _read_recent_jsonl(self, path: Path, *, limit: int) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        rows.append(json.loads(text))
                    except Exception:
                        continue
        except Exception:
            return []
        rows = rows[-max(1, int(limit)) :]
        rows.reverse()
        return rows

    def _refresh_daily_report(self, strategy_name: str, applied_modifications: List[Dict[str, Any]]) -> None:
        closed = self.store.get_recent_closed(strategy_name, limit=100)
        skipped = []
        timing = []
        bad_sl = []
        bad_exit = []
        patterns = []
        for row in closed:
            trade_ticket = str(row.get("trade_ticket") or row.get("signal_id") or "")
            classification = str(row.get("loss_classification") or "")
            if classification == "AVOIDABLE_BAD_TRADE":
                skipped.append(trade_ticket)
            elif classification == "CORRECT_IDEA_WRONG_TIMING":
                timing.append(trade_ticket)
            elif classification == "CORRECT_DIRECTION_BAD_SL":
                bad_sl.append(trade_ticket)
            elif classification == "VALID_ENTRY_BAD_TP_OR_EXIT":
                bad_exit.append(trade_ticket)
            if classification:
                patterns.append(f"{classification}:{(row.get('signal_snapshot') or {}).get('setup_type') or 'UNKNOWN'}")
        active = self.rule_store.list_active(strategy_name)
        self.audit_logger.write_daily_report(
            enabled_strategy=strategy_name,
            closed_trade_count=len(closed),
            skipped_trades=skipped,
            timing_trades=timing,
            bad_sl_trades=bad_sl,
            bad_exit_trades=bad_exit,
            repeated_patterns=patterns[:5],
            applied_modifications=applied_modifications,
            kept_modifications=[item.get("modification_id") for item in active],
            rolled_back_modifications=[item.get("modification_id") for item in self.rule_store.state.get("rolled_back", [])],
        )
