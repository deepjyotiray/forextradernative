from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditLogger:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.ai_decisions_path = self.base_dir / "ai_decisions.jsonl"
        self.live_modifications_path = self.base_dir / "live_modifications.jsonl"
        self.trade_forensics_path = self.base_dir / "trade_forensics.jsonl"
        self.ai_metadata_path = self.base_dir / "ai_request_response_meta.jsonl"
        self.daily_report_path = self.base_dir / "daily_ai_report.md"

    def _append_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str, ensure_ascii=True) + "\n")

    def log_ai_decision(self, payload: Dict[str, Any]) -> None:
        self._append_jsonl(self.ai_decisions_path, payload)

    def log_live_modification(self, payload: Dict[str, Any]) -> None:
        self._append_jsonl(self.live_modifications_path, payload)

    def log_trade_forensic(self, payload: Dict[str, Any]) -> None:
        self._append_jsonl(self.trade_forensics_path, payload)

    def log_ai_metadata(self, payload: Dict[str, Any]) -> None:
        self._append_jsonl(self.ai_metadata_path, payload)

    def write_daily_report(
        self,
        *,
        enabled_strategy: str,
        closed_trade_count: int,
        skipped_trades: Iterable[str],
        timing_trades: Iterable[str],
        bad_sl_trades: Iterable[str],
        bad_exit_trades: Iterable[str],
        repeated_patterns: Iterable[str],
        applied_modifications: List[Dict[str, Any]],
        kept_modifications: Iterable[str],
        rolled_back_modifications: Iterable[str],
    ) -> None:
        skipped_list = list(skipped_trades)
        timing_list = list(timing_trades)
        bad_sl_list = list(bad_sl_trades)
        bad_exit_list = list(bad_exit_trades)
        repeated_list = list(repeated_patterns)
        kept_list = list(kept_modifications)
        rolled_back_list = list(rolled_back_modifications)
        lines = [
            "# Daily AI Trade Correction Report",
            "",
            f"- Generated at: {_utc_now_iso()}",
            f"- Enabled strategy: {enabled_strategy or 'UNKNOWN'}",
            f"- Closed trades reviewed: {closed_trade_count}",
            "",
            "## Daily Answers",
            "",
            f"1. Why did the strategy lose today? {'; '.join(repeated_list) if repeated_list else 'No repeated loss pattern identified.'}",
            f"2. Which trades should have been skipped? {', '.join(skipped_list) if skipped_list else 'None flagged for skipping.'}",
            f"3. Which trades were valid but entered wrongly? {', '.join(timing_list) if timing_list else 'None flagged for entry timing correction.'}",
            f"4. Which trades needed different SL? {', '.join(bad_sl_list) if bad_sl_list else 'None flagged for SL adjustment.'}",
            f"5. Which trades needed different TP/exit? {', '.join(bad_exit_list) if bad_exit_list else 'None flagged for exit adjustment.'}",
            f"6. Which pattern repeated? {'; '.join(repeated_list) if repeated_list else 'No stable repeated pattern found.'}",
            f"7. Which live modifications were applied? {', '.join(mod.get('modification_id', '') for mod in applied_modifications) if applied_modifications else 'No live modifications applied.'}",
            f"8. Did the modifications improve results? {'Pending more samples.' if applied_modifications else 'No active modifications to evaluate yet.'}",
            f"9. Which modifications should stay tomorrow? {', '.join(kept_list) if kept_list else 'None retained yet.'}",
            f"10. Which should be rolled back? {', '.join(rolled_back_list) if rolled_back_list else 'No rollback triggered.'}",
            "",
        ]
        self.daily_report_path.write_text("\n".join(lines), encoding="utf-8")
