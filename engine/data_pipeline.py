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
from datetime import datetime, timezone
from typing import Dict, List, Optional
from .trade_attribution import get_recent_attributions


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH = os.path.join(_BASE_DIR, "analytics.db")
_CSV_PATH = os.path.join(_BASE_DIR, "trade_data.csv")


class DataPipeline:
    def __init__(self):
        self._init_database()
    
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
        
        # Convert timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['completion_time'] = pd.to_datetime(df['completion_time'])
        
        # Convert boolean flags
        df['compression_flag'] = df['compression_flag'].astype(bool)
        df['ltf_conflict_flag'] = df['ltf_conflict_flag'].astype(bool)
        
        return df
    
    def get_completed_trades_only(self, days: int = 30) -> pd.DataFrame:
        """Get only completed trades with outcomes."""
        df = self.get_trade_data(days)
        return df[df['decision'] == 'TRADE_TAKEN'].dropna(subset=['outcome'])
    
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