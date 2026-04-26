"""
Data Pipeline - Convert trade logs to structured format for analysis.

Processes:
- Trade attribution logs (JSON lines)
- Decision logs (including skipped trades)
- Converts to structured CSV/SQLite format
- Provides clean data for dashboard and analysis
"""
import json
import sqlite3
import pandas as pd
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional
from .trade_attribution import get_recent_attributions


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH = os.path.join(_BASE_DIR, "analytics.db")
_ORDERS_DB_PATH = os.path.join(_BASE_DIR, "orders.db")
_CSV_PATH = os.path.join(_BASE_DIR, "trade_data.csv")


class DataPipeline:
    def __init__(self):
        self._init_database()

    @staticmethod
    def _parse_iso_to_datetime(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            normalized = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    @staticmethod
    def _extract_percent(reason: Optional[str], label: str) -> Optional[float]:
        if not reason:
            return None
        match = re.search(rf"{label}\s*([0-9]+(?:\.[0-9]+)?)%", reason, flags=re.IGNORECASE)
        if not match:
            return None
        try:
            return float(match.group(1)) / 100.0
        except Exception:
            return None

    @staticmethod
    def _extract_quality_score(reason: Optional[str]) -> Optional[float]:
        if not reason:
            return None
        match = re.search(r"\[Q:([0-9]+(?:\.[0-9]+)?)%", reason, flags=re.IGNORECASE)
        if not match:
            return None
        try:
            return float(match.group(1)) / 100.0
        except Exception:
            return None

    @staticmethod
    def _normalize_session(session_value: Optional[str]) -> str:
        session = (session_value or "UNKNOWN").upper().strip()
        if session in ("NEW_YORK", "NY"):
            return "NY"
        return session if session else "UNKNOWN"

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _load_orders_fallback_dataframe(self, days: int = 30) -> pd.DataFrame:
        """Build analytics-compatible dataframe directly from orders.db closed orders."""
        if not os.path.exists(_ORDERS_DB_PATH):
            return pd.DataFrame()

        cutoff_time = datetime.now(timezone.utc).timestamp() - (days * 24 * 3600)
        rows: List[Dict] = []

        conn = sqlite3.connect(_ORDERS_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT
                ticket, strategy, direction, reason, features,
                open_time, close_time, entry_price, exit_price, sl, tp, final_pnl
            FROM orders
            WHERE status = 'CLOSED' AND close_time IS NOT NULL
            ORDER BY close_time DESC
            """
        )
        for row in cursor.fetchall():
            features = {}
            try:
                features = json.loads(row["features"] or "{}")
            except Exception:
                features = {}

            close_dt = self._parse_iso_to_datetime(row["close_time"])
            if not close_dt:
                continue
            close_unix = close_dt.timestamp()
            if close_unix < cutoff_time:
                continue

            open_dt = self._parse_iso_to_datetime(row["open_time"])
            duration = None
            if open_dt:
                duration = max(int((close_dt - open_dt).total_seconds()), 0)

            pnl = self._safe_float(row["final_pnl"]) or 0.0
            if pnl > 0:
                outcome = "WIN"
            elif pnl < 0:
                outcome = "LOSS"
            else:
                outcome = "BE"

            direction_raw = (row["direction"] or "").upper()
            setup_direction = "LONG" if direction_raw in ("BUY", "LONG") else "SHORT"

            session_raw = features.get("session")
            session = self._normalize_session(session_raw)

            quality_score = self._safe_float(features.get("quality_score"))
            if quality_score is None:
                quality_score = self._extract_quality_score(row["reason"])

            tick_ratio = self._safe_float(features.get("tick_ratio"))
            if tick_ratio is None:
                tick_ratio = self._extract_percent(row["reason"], "Tick\\s*ratio")

            entry_tick_velocity = self._safe_float(features.get("entry_tick_velocity"))
            spread_mean = self._safe_float(features.get("spread"))
            atr_value = self._safe_float(features.get("atr"))

            reason_text = row["reason"] or ""
            trade_type = "WITH_TREND"
            if "counter-trend" in reason_text.lower() or "ctr" in reason_text.lower():
                trade_type = "COUNTER_TREND"

            rows.append(
                {
                    "timestamp": close_dt,
                    "unix_time": close_unix,
                    "strategy": row["strategy"] or "UNKNOWN",
                    "setup_direction": setup_direction,
                    "bias_direction": None,
                    "trade_type": trade_type,
                    "quality_score": quality_score,
                    "threshold_used": None,
                    "spread_mean": spread_mean,
                    "spread_std": None,
                    "spread_percentile": None,
                    "atr_value": atr_value,
                    "compression_flag": False,
                    "ltf_conflict_flag": False,
                    "tick_ratio": tick_ratio,
                    "tick_velocity": entry_tick_velocity,
                    "session": session,
                    "decision": "TRADE_TAKEN",
                    "reason": reason_text,
                    "price": self._safe_float(row["entry_price"]),
                    "trade_id": f"order_{row['ticket']}",
                    "entry_price": self._safe_float(row["entry_price"]),
                    "exit_price": self._safe_float(row["exit_price"]),
                    "sl": self._safe_float(row["sl"]),
                    "tp": self._safe_float(row["tp"]),
                    "outcome": outcome,
                    "pnl": pnl,
                    "trade_duration": duration,
                    "exit_reason": "ORDER_DB",
                    "completion_time": close_dt,
                }
            )
        conn.close()

        if not rows:
            return pd.DataFrame()

        fallback_df = pd.DataFrame(rows)
        fallback_df["timestamp"] = pd.to_datetime(fallback_df["timestamp"], errors="coerce", utc=True, format="ISO8601")
        fallback_df["completion_time"] = pd.to_datetime(fallback_df["completion_time"], errors="coerce", utc=True, format="ISO8601")
        fallback_df["compression_flag"] = fallback_df["compression_flag"].astype(bool)
        fallback_df["ltf_conflict_flag"] = fallback_df["ltf_conflict_flag"].astype(bool)
        fallback_df = fallback_df.sort_values("unix_time", ascending=False).reset_index(drop=True)
        return fallback_df
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        conn = sqlite3.connect(_DB_PATH)
        cursor = conn.cursor()
        
        # Trade decisions table (all decisions - taken and skipped)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                unix_time REAL NOT NULL,
                strategy TEXT NOT NULL,
                setup_direction TEXT,
                bias_direction TEXT,
                trade_type TEXT,
                quality_score REAL,
                threshold_used REAL,
                spread_mean REAL,
                spread_std REAL,
                spread_percentile REAL,
                atr_value REAL,
                compression_flag INTEGER,
                ltf_conflict_flag INTEGER,
                tick_ratio REAL,
                tick_velocity REAL,
                session TEXT,
                decision TEXT NOT NULL,
                reason TEXT,
                price REAL,
                trade_id TEXT
            )
        """)
        
        # Trade outcomes table (only completed trades)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT NOT NULL UNIQUE,
                entry_price REAL,
                exit_price REAL,
                sl REAL,
                tp REAL,
                outcome TEXT,
                pnl REAL,
                trade_duration INTEGER,
                exit_reason TEXT,
                completion_time TEXT,
                FOREIGN KEY (trade_id) REFERENCES trade_decisions (trade_id)
            )
        """)
        
        # Create indexes for better query performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON trade_decisions(unix_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_strategy ON trade_decisions(strategy)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_session ON trade_decisions(session)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_trade_id ON trade_outcomes(trade_id)")
        
        conn.commit()
        conn.close()
    
    def sync_from_attribution_logs(self, hours: int = 168) -> Dict:
        """Sync data from attribution engine logs to structured database."""
        
        # Get recent attributions
        attributions = get_recent_attributions(hours)
        
        if not attributions:
            return {"synced": 0, "message": "No attribution data found"}
        
        conn = sqlite3.connect(_DB_PATH)
        cursor = conn.cursor()
        
        decisions_synced = 0
        outcomes_synced = 0
        
        for attr in attributions:
            try:
                # Insert/update trade decision
                cursor.execute("""
                    INSERT OR REPLACE INTO trade_decisions (
                        timestamp, unix_time, strategy, setup_direction, bias_direction,
                        trade_type, quality_score, threshold_used, spread_mean, spread_std,
                        spread_percentile, atr_value, compression_flag, ltf_conflict_flag,
                        tick_ratio, tick_velocity, session, decision, reason, price, trade_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    attr.get("timestamp"),
                    attr.get("unix_time", 0),
                    attr.get("strategy"),
                    attr.get("setup_direction"),
                    attr.get("bias_direction"),
                    attr.get("trade_type"),
                    attr.get("quality_score"),
                    attr.get("threshold_used"),
                    attr.get("spread_mean"),
                    attr.get("spread_std"),
                    attr.get("spread_percentile"),
                    attr.get("atr_value"),
                    1 if attr.get("compression_flag") else 0,
                    1 if attr.get("ltf_conflict_flag") else 0,
                    attr.get("tick_ratio"),
                    attr.get("tick_velocity"),
                    attr.get("session"),
                    attr.get("decision"),
                    attr.get("reason"),
                    attr.get("price"),
                    attr.get("trade_id")
                ))
                decisions_synced += 1
                
                # Insert trade outcome if completed
                if attr.get("trade_completed") and attr.get("trade_id"):
                    cursor.execute("""
                        INSERT OR REPLACE INTO trade_outcomes (
                            trade_id, entry_price, exit_price, sl, tp, outcome,
                            pnl, trade_duration, exit_reason, completion_time
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        attr.get("trade_id"),
                        attr.get("entry_price"),
                        attr.get("exit_price"),
                        attr.get("sl"),
                        attr.get("tp"),
                        attr.get("outcome"),
                        attr.get("pnl"),
                        attr.get("trade_duration"),
                        attr.get("exit_reason"),
                        attr.get("completion_time")
                    ))
                    outcomes_synced += 1
                    
            except Exception as e:
                print(f"Error syncing attribution: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        return {
            "synced": True,
            "decisions_synced": decisions_synced,
            "outcomes_synced": outcomes_synced,
            "total_attributions": len(attributions)
        }
    
    def get_trade_data(self, days: int = 30) -> pd.DataFrame:
        """Get structured trade data as pandas DataFrame."""
        
        conn = sqlite3.connect(_DB_PATH)
        
        # Join decisions with outcomes for completed trades
        query = """
            SELECT 
                d.timestamp,
                d.unix_time,
                d.strategy,
                d.setup_direction,
                d.bias_direction,
                d.trade_type,
                d.quality_score,
                d.threshold_used,
                d.spread_mean,
                d.spread_std,
                d.spread_percentile,
                d.atr_value,
                d.compression_flag,
                d.ltf_conflict_flag,
                d.tick_ratio,
                d.tick_velocity,
                d.session,
                d.decision,
                d.reason,
                d.price,
                d.trade_id,
                o.entry_price,
                o.exit_price,
                o.sl,
                o.tp,
                o.outcome,
                o.pnl,
                o.trade_duration,
                o.exit_reason,
                o.completion_time
            FROM trade_decisions d
            LEFT JOIN trade_outcomes o ON d.trade_id = o.trade_id
            WHERE d.unix_time >= ?
            ORDER BY d.unix_time DESC
        """
        
        cutoff_time = datetime.now(timezone.utc).timestamp() - (days * 24 * 3600)
        
        df = pd.read_sql_query(query, conn, params=[cutoff_time])
        conn.close()
        
        if df.empty:
            return self._load_orders_fallback_dataframe(days)

        # Convert timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True, format='ISO8601')
        df['completion_time'] = pd.to_datetime(df['completion_time'], errors='coerce', utc=True, format='ISO8601')
        
        # Convert boolean flags
        df['compression_flag'] = df['compression_flag'].astype(bool)
        df['ltf_conflict_flag'] = df['ltf_conflict_flag'].astype(bool)
        
        return df
    
    def get_completed_trades_only(self, days: int = 30) -> pd.DataFrame:
        """Get only completed trades with outcomes."""
        df = self.get_trade_data(days)
        completed = df[df['decision'] == 'TRADE_TAKEN'].dropna(subset=['outcome'])
        if completed.empty:
            return self._load_orders_fallback_dataframe(days)
        return completed
    
    def get_decision_summary(self, days: int = 7) -> Dict:
        """Get summary of all decisions (taken/skipped)."""
        df = self.get_trade_data(days)
        
        total_decisions = len(df)
        taken = len(df[df['decision'] == 'TRADE_TAKEN'])
        skipped = len(df[df['decision'] == 'TRADE_SKIPPED'])
        completed = len(df.dropna(subset=['outcome']))
        
        # Skip reasons analysis
        skipped_df = df[df['decision'] == 'TRADE_SKIPPED']
        skip_reasons = skipped_df['reason'].value_counts().head(10).to_dict()
        
        return {
            "total_decisions": total_decisions,
            "trades_taken": taken,
            "trades_skipped": skipped,
            "trades_completed": completed,
            "conversion_rate": taken / total_decisions if total_decisions > 0 else 0,
            "completion_rate": completed / taken if taken > 0 else 0,
            "top_skip_reasons": skip_reasons,
            "period_days": days
        }
    
    def export_to_csv(self, days: int = 30) -> str:
        """Export trade data to CSV file."""
        df = self.get_trade_data(days)
        df.to_csv(_CSV_PATH, index=False)
        return _CSV_PATH
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        conn = sqlite3.connect(_DB_PATH)
        cursor = conn.cursor()
        
        # Count records
        cursor.execute("SELECT COUNT(*) FROM trade_decisions")
        decisions_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM trade_outcomes")
        outcomes_count = cursor.fetchone()[0]
        
        # Date range
        cursor.execute("SELECT MIN(unix_time), MAX(unix_time) FROM trade_decisions")
        min_time, max_time = cursor.fetchone()
        
        conn.close()
        
        date_range = None
        if min_time and max_time:
            min_date = datetime.fromtimestamp(min_time, timezone.utc)
            max_date = datetime.fromtimestamp(max_time, timezone.utc)
            date_range = {
                "start": min_date.isoformat(),
                "end": max_date.isoformat(),
                "days": (max_date - min_date).days
            }
        
        return {
            "decisions_count": decisions_count,
            "outcomes_count": outcomes_count,
            "completion_rate": outcomes_count / decisions_count if decisions_count > 0 else 0,
            "date_range": date_range,
            "database_path": _DB_PATH
        }


# Singleton instance
data_pipeline = DataPipeline()


def sync_data(hours: int = 168) -> Dict:
    """Sync data from attribution logs."""
    return data_pipeline.sync_from_attribution_logs(hours)


def get_trade_data(days: int = 30) -> pd.DataFrame:
    """Get structured trade data."""
    return data_pipeline.get_trade_data(days)


def get_completed_trades_only(days: int = 30) -> pd.DataFrame:
    """Get only completed trades."""
    return data_pipeline.get_completed_trades_only(days)


def get_decision_summary(days: int = 7) -> Dict:
    """Get decision summary."""
    return data_pipeline.get_decision_summary(days)


def export_to_csv(days: int = 30) -> str:
    """Export to CSV."""
    return data_pipeline.export_to_csv(days)


def get_database_stats() -> Dict:
    """Get database statistics."""
    return data_pipeline.get_database_stats()
