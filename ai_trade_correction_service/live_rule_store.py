from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from .decision_schema import LiveModification


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_iso(value: str | None) -> datetime | None:
    try:
        if not value:
            return None
        dt = datetime.fromisoformat(value)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


class LiveRuleStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.state = {"active_modifications": [], "rolled_back": []}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self.state = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.state = {"active_modifications": [], "rolled_back": []}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.state, indent=2, default=str), encoding="utf-8")

    def list_active(self, strategy: str) -> List[Dict[str, Any]]:
        active = []
        for item in list(self.state.get("active_modifications", [])):
            if str(item.get("target_strategy") or "").upper() != str(strategy or "").upper():
                continue
            if self._is_expired(item):
                continue
            active.append(item)
        return active

    def apply_modification(self, modification: LiveModification, analysis_id: str) -> Dict[str, Any]:
        payload = modification.model_dump(mode="json")
        payload["analysis_id"] = analysis_id
        payload["applied_at"] = _utc_now().isoformat()
        payload.setdefault("observed_trade_results", [])
        payload.setdefault("successful_trade_count", 0)
        payload.setdefault("loss_count", 0)

        active = [item for item in self.state.get("active_modifications", []) if item.get("modification_id") != payload["modification_id"]]
        active.append(payload)
        self.state["active_modifications"] = active
        self._save()
        return payload

    def cleanup_expired(self, *, current_market_state: str = "") -> List[Dict[str, Any]]:
        remaining = []
        expired = []
        for item in self.state.get("active_modifications", []):
            if self._is_expired(item, current_market_state=current_market_state):
                expired.append(item)
            else:
                remaining.append(item)
        if expired:
            self.state["active_modifications"] = remaining
            self._save()
        return expired

    def record_trade_result(self, strategy: str, signal_snapshot: Dict[str, Any], close_snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        rolled_back = []
        for item in self.state.get("active_modifications", []):
            if str(item.get("target_strategy") or "").upper() != str(strategy or "").upper():
                continue
            if not self._matches(item, signal_snapshot):
                continue
            item.setdefault("observed_trade_results", []).append(
                {
                    "signal_id": signal_snapshot.get("signal_id"),
                    "pnl_r": close_snapshot.get("pnl_r"),
                    "won": float(close_snapshot.get("pnl") or 0.0) > 0,
                }
            )
            if float(close_snapshot.get("pnl") or 0.0) > 0:
                item["successful_trade_count"] = int(item.get("successful_trade_count", 0)) + 1
            else:
                item["loss_count"] = int(item.get("loss_count", 0)) + 1
            if self._should_roll_back(item):
                item["rolled_back_at"] = _utc_now().isoformat()
                rolled_back.append(item)
        if rolled_back:
            active = [item for item in self.state.get("active_modifications", []) if item.get("modification_id") not in {row["modification_id"] for row in rolled_back}]
            self.state["active_modifications"] = active
            self.state.setdefault("rolled_back", []).extend(rolled_back)
        self._save()
        return rolled_back

    def materialize(self, strategy: str) -> Dict[str, List[Dict[str, Any]]]:
        result = {
            "setup_skip_rules": [],
            "setup_wait_rules": [],
            "sl_adjustment_rules": [],
            "tp_adjustment_rules": [],
            "trailing_exit_rules": [],
            "partial_exit_rules": [],
            "confirmation_requirements": [],
            "avoided_patterns": [],
            "preferred_entry_patterns": [],
        }
        for item in self.list_active(strategy):
            key = str(item.get("exact_runtime_override_key") or "")
            if key in result:
                result[key].append(item)
        return result

    def _matches(self, item: Dict[str, Any], signal_snapshot: Dict[str, Any]) -> bool:
        if item.get("target_setup_type") not in {"", "ANY", signal_snapshot.get("setup_type")}:
            return False
        if item.get("target_direction") not in {"", "ANY", signal_snapshot.get("direction")}:
            return False
        if item.get("target_market_state") not in {"", "ANY", signal_snapshot.get("market_state")}:
            return False
        return True

    def _is_expired(self, item: Dict[str, Any], *, current_market_state: str = "") -> bool:
        expiry = item.get("expiry") or {}
        expiry_type = str(expiry.get("type") or "").strip().lower()
        if expiry_type == "after_minutes":
            applied_at = _safe_iso(item.get("applied_at"))
            minutes = float(expiry.get("value") or 0)
            return bool(applied_at and minutes > 0 and _utc_now() >= applied_at + timedelta(minutes=minutes))
        if expiry_type == "after_n_trades":
            limit = int(expiry.get("value") or 0)
            return limit > 0 and len(item.get("observed_trade_results", []) or []) >= limit
        if expiry_type == "after_n_successes":
            limit = int(expiry.get("value") or 0)
            return limit > 0 and int(item.get("successful_trade_count", 0)) >= limit
        if expiry_type == "session_end":
            reference = str(expiry.get("reference") or "").upper()
            return bool(reference and current_market_state and reference != current_market_state)
        if expiry_type == "market_state_change":
            reference = str(expiry.get("reference") or "")
            return bool(reference and current_market_state and reference != current_market_state)
        if expiry_type == "absolute_time":
            ref_dt = _safe_iso(str(expiry.get("reference") or ""))
            return bool(ref_dt and _utc_now() >= ref_dt)
        return False

    def _should_roll_back(self, item: Dict[str, Any]) -> bool:
        rollback = item.get("rollback_condition") or {}
        rollback_type = str(rollback.get("type") or "").strip().lower()
        if rollback_type == "after_n_losses_same_type":
            limit = int(rollback.get("value") or 0)
            return limit > 0 and int(item.get("loss_count", 0)) >= limit
        if rollback_type == "confidence_below":
            threshold = float(rollback.get("value") or 0)
            return threshold > 0 and float(item.get("confidence", 0.0)) < threshold
        return False
