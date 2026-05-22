from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RecentTradeWindow:
    def __init__(self, db_path: Path):
        self.db_path = str(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_snapshots (
                    signal_id TEXT PRIMARY KEY,
                    trade_ticket INTEGER,
                    strategy TEXT NOT NULL,
                    setup_type TEXT,
                    direction TEXT,
                    signal_time TEXT,
                    status TEXT,
                    session_label TEXT,
                    market_state TEXT,
                    pnl REAL,
                    pnl_r REAL,
                    loss_classification TEXT,
                    signal_snapshot TEXT NOT NULL,
                    live_snapshot TEXT,
                    close_snapshot TEXT,
                    counterfactuals TEXT,
                    analysis_payload TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_signals (
                    signal_id TEXT PRIMARY KEY,
                    strategy TEXT NOT NULL,
                    setup_type TEXT,
                    direction TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    wait_action TEXT NOT NULL,
                    conditions_json TEXT NOT NULL,
                    signal_snapshot TEXT NOT NULL,
                    signal_payload TEXT NOT NULL,
                    state TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_snapshots_strategy_time ON trade_snapshots(strategy, signal_time DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_snapshots_ticket ON trade_snapshots(trade_ticket)")

    def upsert_signal_snapshot(self, snapshot: Dict[str, Any]) -> str:
        signal_id = str(snapshot.get("signal_id") or snapshot.get("trade_id") or "")
        payload = json.dumps(snapshot, default=str, ensure_ascii=True)
        now_ts = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trade_snapshots (
                    signal_id, trade_ticket, strategy, setup_type, direction, signal_time, status,
                    session_label, market_state, pnl, pnl_r, loss_classification, signal_snapshot,
                    live_snapshot, close_snapshot, counterfactuals, analysis_payload, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(signal_id) DO UPDATE SET
                    trade_ticket=excluded.trade_ticket,
                    strategy=excluded.strategy,
                    setup_type=excluded.setup_type,
                    direction=excluded.direction,
                    signal_time=excluded.signal_time,
                    status=excluded.status,
                    session_label=excluded.session_label,
                    market_state=excluded.market_state,
                    signal_snapshot=excluded.signal_snapshot,
                    updated_at=excluded.updated_at
                """,
                (
                    signal_id,
                    snapshot.get("trade_ticket"),
                    snapshot.get("strategy"),
                    snapshot.get("setup_type"),
                    snapshot.get("direction"),
                    snapshot.get("timestamp"),
                    snapshot.get("status", "SIGNAL_CAPTURED"),
                    snapshot.get("session"),
                    snapshot.get("market_state"),
                    snapshot.get("pnl"),
                    snapshot.get("pnl_r"),
                    snapshot.get("loss_classification"),
                    payload,
                    None,
                    None,
                    None,
                    None,
                    now_ts,
                    now_ts,
                ),
            )
        return signal_id

    def update_signal_status(self, signal_id: str, *, status: str, reason: str = "", signal_snapshot: Optional[Dict[str, Any]] = None) -> None:
        with self._connect() as conn:
            current = self.get_signal_snapshot(signal_id) or {}
            snapshot = dict(signal_snapshot or current or {})
            if snapshot:
                snapshot["status"] = status
                if reason:
                    snapshot["final_signal_reason"] = reason
            conn.execute(
                """
                UPDATE trade_snapshots
                SET status = ?, signal_snapshot = ?, updated_at = ?
                WHERE signal_id = ?
                """,
                (
                    status,
                    json.dumps(snapshot, default=str, ensure_ascii=True)
                    if snapshot
                    else json.dumps(current.get("signal_snapshot") or {}, default=str, ensure_ascii=True),
                    time.time(),
                    signal_id,
                ),
            )

    def attach_trade_open(self, signal_id: str, trade_ticket: int, live_snapshot: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE trade_snapshots
                SET trade_ticket = ?, status = ?, live_snapshot = ?, updated_at = ?
                WHERE signal_id = ?
                """,
                (
                    int(trade_ticket),
                    "TRADE_OPEN",
                    json.dumps(live_snapshot, default=str, ensure_ascii=True),
                    time.time(),
                    signal_id,
                ),
            )

    def update_live_snapshot(self, signal_id: str, live_snapshot: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE trade_snapshots SET live_snapshot = ?, updated_at = ? WHERE signal_id = ?",
                (json.dumps(live_snapshot, default=str, ensure_ascii=True), time.time(), signal_id),
            )

    def close_trade_snapshot(
        self,
        signal_id: str,
        *,
        close_snapshot: Dict[str, Any],
        counterfactuals: Dict[str, Any],
        loss_classification: str,
        analysis_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE trade_snapshots
                SET status = ?, pnl = ?, pnl_r = ?, loss_classification = ?, close_snapshot = ?,
                    counterfactuals = ?, analysis_payload = ?, updated_at = ?
                WHERE signal_id = ?
                """,
                (
                    "TRADE_CLOSED",
                    close_snapshot.get("pnl"),
                    close_snapshot.get("pnl_r"),
                    loss_classification,
                    json.dumps(close_snapshot, default=str, ensure_ascii=True),
                    json.dumps(counterfactuals, default=str, ensure_ascii=True),
                    json.dumps(analysis_payload, default=str, ensure_ascii=True) if analysis_payload else None,
                    time.time(),
                    signal_id,
                ),
            )

    def get_signal_snapshot(self, signal_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM trade_snapshots WHERE signal_id = ?", (signal_id,)).fetchone()
            return self._row_to_snapshot(row) if row else None

    def get_snapshot_by_ticket(self, trade_ticket: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM trade_snapshots WHERE trade_ticket = ? ORDER BY updated_at DESC LIMIT 1",
                (int(trade_ticket),),
            ).fetchone()
            return self._row_to_snapshot(row) if row else None

    def get_recent_closed(self, strategy: str, limit: int = 5) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM trade_snapshots
                WHERE strategy = ? AND status = 'TRADE_CLOSED'
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (strategy, int(limit)),
            ).fetchall()
        return [self._row_to_snapshot(row) for row in rows]

    def get_recent_all(self, strategy: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trade_snapshots WHERE strategy = ? ORDER BY updated_at DESC LIMIT ?",
                (strategy, int(limit)),
            ).fetchall()
        return [self._row_to_snapshot(row) for row in rows]

    def count_closed(self, strategy: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM trade_snapshots WHERE strategy = ? AND status = 'TRADE_CLOSED'",
                (strategy,),
            ).fetchone()
        return int((row["c"] if row is not None else 0) or 0)

    def get_consecutive_losses(self, strategy: str, limit: int = 5) -> List[Dict[str, Any]]:
        recent = self.get_recent_closed(strategy, limit=limit)
        losses = []
        for item in recent:
            if float(((item.get("close_snapshot") or {}).get("pnl") or 0.0)) >= 0:
                break
            losses.append(item)
        return losses

    def find_similar_trades(self, strategy: str, snapshot: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        rows = [row for row in self.get_recent_closed(strategy, limit=250) if self._similarity_score(snapshot, row) > 0]
        rows.sort(key=lambda item: item.get("_similarity_score", 0.0), reverse=True)
        return rows[:limit]

    def add_pending_signal(
        self,
        *,
        signal_id: str,
        strategy: str,
        setup_type: str,
        direction: str,
        expires_at: Optional[str],
        wait_action: str,
        conditions: Dict[str, Any],
        signal_snapshot: Dict[str, Any],
        signal_payload: Dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pending_signals (
                    signal_id, strategy, setup_type, direction, created_at, expires_at,
                    wait_action, conditions_json, signal_snapshot, signal_payload, state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal_id,
                    strategy,
                    setup_type,
                    direction,
                    _utc_now_iso(),
                    expires_at,
                    wait_action,
                    json.dumps(conditions, default=str, ensure_ascii=True),
                    json.dumps(signal_snapshot, default=str, ensure_ascii=True),
                    json.dumps(signal_payload, default=str, ensure_ascii=True),
                    "PENDING",
                ),
            )

    def list_pending_signals(self, strategy: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM pending_signals WHERE strategy = ? AND state = 'PENDING' ORDER BY created_at ASC",
                (strategy,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["conditions"] = json.loads(item.pop("conditions_json") or "{}")
            item["signal_snapshot"] = json.loads(item.get("signal_snapshot") or "{}")
            item["signal_payload"] = json.loads(item.get("signal_payload") or "{}")
            result.append(item)
        return result

    def resolve_pending_signal(self, signal_id: str, state: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE pending_signals SET state = ? WHERE signal_id = ?", (state, signal_id))

    def _row_to_snapshot(self, row: sqlite3.Row) -> Dict[str, Any]:
        item = dict(row)
        item["signal_snapshot"] = json.loads(item.get("signal_snapshot") or "{}")
        item["live_snapshot"] = json.loads(item.get("live_snapshot") or "{}") if item.get("live_snapshot") else {}
        item["close_snapshot"] = json.loads(item.get("close_snapshot") or "{}") if item.get("close_snapshot") else {}
        item["counterfactuals"] = json.loads(item.get("counterfactuals") or "{}") if item.get("counterfactuals") else {}
        item["analysis_payload"] = json.loads(item.get("analysis_payload") or "{}") if item.get("analysis_payload") else {}
        return item

    def _similarity_score(self, baseline: Dict[str, Any], candidate: Dict[str, Any]) -> float:
        candidate_signal = candidate.get("signal_snapshot") or {}
        score = 0.0
        checks = [
            (baseline.get("setup_type"), candidate_signal.get("setup_type"), 2.0),
            (baseline.get("direction"), candidate_signal.get("direction"), 2.0),
            (baseline.get("market_state"), candidate_signal.get("market_state"), 1.0),
            (baseline.get("session"), candidate_signal.get("session"), 1.0),
            (baseline.get("price_vs_ema20"), candidate_signal.get("price_vs_ema20"), 1.0),
            (baseline.get("price_vs_vwap"), candidate_signal.get("price_vs_vwap"), 1.0),
            (baseline.get("liquidity_sweep"), candidate_signal.get("liquidity_sweep"), 1.0),
            (baseline.get("candle_confirmation"), candidate_signal.get("candle_confirmation"), 1.0),
        ]
        for left, right, weight in checks:
            if left == right and left not in ("", None):
                score += weight
        if score > 0:
            candidate["_similarity_score"] = round(score, 3)
        return score
