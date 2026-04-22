"""
Order Database - Complete local storage for all trade data.
Stores comprehensive order information with session tracking.
Uses MT5 only for live PnL updates.
"""
import sqlite3
import json
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import config as cfg

_IST = timezone(timedelta(hours=5, minutes=30))
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH = os.path.join(_BASE_DIR, "orders.db")


class OrderDatabase:
    def __init__(self):
        self.db_path = _DB_PATH
        self._init_database()
    
    def _init_database(self):
        """Initialize database with orders table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket INTEGER UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    volume REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    sl REAL,
                    tp REAL,
                    sl_distance REAL,
                    strategy TEXT,
                    confidence REAL,
                    reason TEXT,
                    scalp INTEGER DEFAULT 0,
                    be_trigger_price REAL DEFAULT 0,
                    timeout_seconds INTEGER DEFAULT 0,
                    
                    -- Timing
                    open_time TEXT NOT NULL,
                    open_time_ist TEXT NOT NULL,
                    fill_timestamp REAL NOT NULL,
                    close_time TEXT,
                    close_time_ist TEXT,
                    
                    -- Session info
                    session_type TEXT,
                    market_phase TEXT,
                    
                    -- PnL tracking
                    live_pnl REAL DEFAULT 0,
                    peak_pnl REAL DEFAULT 0,
                    final_pnl REAL,
                    swap REAL DEFAULT 0,
                    commission REAL DEFAULT 0,
                    
                    -- Management flags
                    sl_breakeven INTEGER DEFAULT 0,
                    partial_closed INTEGER DEFAULT 0,
                    trail_active INTEGER DEFAULT 0,
                    initial_volume REAL,
                    
                    -- Status
                    status TEXT DEFAULT 'OPEN',
                    close_reason TEXT,
                    
                    -- Features (JSON)
                    features TEXT,
                    
                    -- Metadata
                    magic_number INTEGER,
                    created_at REAL DEFAULT (strftime('%s', 'now')),
                    updated_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)
            
            # Add exit_price column if missing (migration)
            try:
                conn.execute("ALTER TABLE orders ADD COLUMN exit_price REAL")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ticket ON orders(ticket)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON orders(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_open_time ON orders(open_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_strategy ON orders(strategy)")
    
    def store_order(self, ticket: int, direction: str, volume: float, entry_price: float,
                   sl: float, tp: float, sl_distance: float, strategy: str = "",
                   confidence: float = 0, reason: str = "", scalp: bool = False,
                   be_trigger: float = 0, timeout: int = 0, features: Dict = None,
                   session_type: str = "", market_phase: str = "") -> bool:
        """Store new order in database."""
        now = datetime.now(timezone.utc)
        ist_now = now.astimezone(_IST)
        
        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute("""
                    INSERT INTO orders (
                        ticket, symbol, direction, volume, entry_price, sl, tp, sl_distance,
                        strategy, confidence, reason, scalp, be_trigger_price, timeout_seconds,
                        open_time, open_time_ist, fill_timestamp, session_type, market_phase,
                        initial_volume, features, magic_number
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticket, cfg.SYMBOL, direction, volume, entry_price, sl, tp, sl_distance,
                    strategy, confidence, reason, int(scalp), be_trigger, timeout,
                    now.isoformat(), ist_now.isoformat(), time.time(), session_type, market_phase,
                    volume, json.dumps(features or {}), cfg.MAGIC_NUMBER
                ))
                return True
            except sqlite3.IntegrityError:
                return False  # Ticket already exists
    
    def update_live_pnl(self, ticket: int, live_pnl: float, current_volume: float = None,
                       current_sl: float = None, current_tp: float = None) -> bool:
        """Update live PnL and position details from MT5."""
        with sqlite3.connect(self.db_path) as conn:
            # Get current peak PnL
            cursor = conn.execute("SELECT peak_pnl FROM orders WHERE ticket = ?", (ticket,))
            row = cursor.fetchone()
            if not row:
                return False
            
            current_peak = row[0] or 0
            new_peak = max(current_peak, live_pnl)
            
            # Update query parts
            updates = ["live_pnl = ?", "peak_pnl = ?", "updated_at = strftime('%s', 'now')"]
            params = [live_pnl, new_peak]
            
            if current_volume is not None:
                updates.append("volume = ?")
                params.append(current_volume)
            
            if current_sl is not None:
                updates.append("sl = ?")
                params.append(current_sl)
            
            if current_tp is not None:
                updates.append("tp = ?")
                params.append(current_tp)
            
            params.append(ticket)
            
            conn.execute(f"""
                UPDATE orders SET {', '.join(updates)}
                WHERE ticket = ?
            """, params)
            
            return conn.total_changes > 0
    
    def close_order(self, ticket: int, final_pnl: float, close_reason: str = "",
                   swap: float = 0, commission: float = 0, exit_price: float = 0) -> bool:
        """Mark order as closed with final PnL."""
        now = datetime.now(timezone.utc)
        ist_now = now.astimezone(_IST)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE orders SET 
                    status = 'CLOSED',
                    close_time = ?,
                    close_time_ist = ?,
                    final_pnl = ?,
                    close_reason = ?,
                    swap = ?,
                    commission = ?,
                    exit_price = CASE WHEN ? > 0 THEN ? ELSE exit_price END,
                    updated_at = strftime('%s', 'now')
                WHERE ticket = ?
            """, (now.isoformat(), ist_now.isoformat(), final_pnl, close_reason,
                  swap, commission, exit_price, exit_price, ticket))
            
            return conn.total_changes > 0
    
    def update_management_flags(self, ticket: int, sl_breakeven: bool = None,
                              partial_closed: bool = None, trail_active: bool = None) -> bool:
        """Update trade management flags."""
        updates = []
        params = []
        
        if sl_breakeven is not None:
            updates.append("sl_breakeven = ?")
            params.append(int(sl_breakeven))
        
        if partial_closed is not None:
            updates.append("partial_closed = ?")
            params.append(int(partial_closed))
        
        if trail_active is not None:
            updates.append("trail_active = ?")
            params.append(int(trail_active))
        
        if not updates:
            return False
        
        updates.append("updated_at = strftime('%s', 'now')")
        params.append(ticket)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"""
                UPDATE orders SET {', '.join(updates)}
                WHERE ticket = ?
            """, params)
            
            return conn.total_changes > 0
    
    def update_exit_price(self, ticket: int, exit_price: float) -> bool:
        """Store exit price immediately when known (before final P&L is available)."""
        if not exit_price or exit_price <= 0:
            return False
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE orders SET exit_price = ?, updated_at = strftime('%s', 'now')
                WHERE ticket = ? AND (exit_price IS NULL OR exit_price = 0)
            """, (exit_price, ticket))
            return conn.total_changes > 0

    def get_stale_closed_orders(self, mt5_map: dict) -> dict:
        """Find closed orders where DB data differs from MT5 deal history.
        Returns {ticket: db_row} for orders needing reconciliation."""
        if not mt5_map:
            return {}
        tickets = list(mt5_map.keys())
        stale = {}
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            placeholders = ','.join('?' * len(tickets))
            cursor = conn.execute(f"""
                SELECT ticket, final_pnl, exit_price, swap, commission
                FROM orders WHERE status = 'CLOSED' AND ticket IN ({placeholders})
            """, tickets)
            for row in cursor.fetchall():
                t = row['ticket']
                mt5t = mt5_map[t]
                mt5_pnl = mt5t.get('pnl', 0)
                mt5_exit = mt5t.get('exit_price', 0)
                db_pnl = row['final_pnl'] or 0
                db_exit = row['exit_price'] or 0
                if (abs(db_pnl - mt5_pnl) > 0.005
                        or (mt5_exit and abs(db_exit - mt5_exit) > 0.005)):
                    stale[t] = dict(row)
        return stale

    def reconcile_order(self, ticket: int, final_pnl: float,
                       exit_price: float = 0, swap: float = 0,
                       commission: float = 0) -> bool:
        """Overwrite closed order data with authoritative MT5 values."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE orders SET
                    final_pnl = ?,
                    exit_price = CASE WHEN ? > 0 THEN ? ELSE exit_price END,
                    swap = ?,
                    commission = ?,
                    updated_at = strftime('%s', 'now')
                WHERE ticket = ? AND status = 'CLOSED'
            """, (final_pnl, exit_price, exit_price, swap, commission, ticket))
            return conn.total_changes > 0
    
    def get_open_orders(self) -> List[Dict]:
        """Get all open orders."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM orders 
                WHERE status = 'OPEN' 
                ORDER BY open_time DESC
            """)
            
            orders = []
            for row in cursor.fetchall():
                order = dict(row)
                order['features'] = json.loads(order['features'] or '{}')
                order['scalp'] = bool(order['scalp'])
                order['sl_breakeven'] = bool(order['sl_breakeven'])
                order['partial_closed'] = bool(order['partial_closed'])
                order['trail_active'] = bool(order['trail_active'])
                orders.append(order)
            
            return orders
    
    def get_closed_orders(self, days: int = 30) -> List[Dict]:
        """Get closed orders from last N days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM orders 
                WHERE status = 'CLOSED' AND open_time >= ?
                ORDER BY close_time DESC
            """, (cutoff.isoformat(),))
            
            orders = []
            for row in cursor.fetchall():
                order = dict(row)
                order['features'] = json.loads(order['features'] or '{}')
                order['scalp'] = bool(order['scalp'])
                order['sl_breakeven'] = bool(order['sl_breakeven'])
                order['partial_closed'] = bool(order['partial_closed'])
                order['trail_active'] = bool(order['trail_active'])
                orders.append(order)
            
            return orders
    
    def get_order(self, ticket: int) -> Optional[Dict]:
        """Get specific order by ticket."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM orders WHERE ticket = ?", (ticket,))
            row = cursor.fetchone()
            
            if row:
                order = dict(row)
                order['features'] = json.loads(order['features'] or '{}')
                order['scalp'] = bool(order['scalp'])
                order['sl_breakeven'] = bool(order['sl_breakeven'])
                order['partial_closed'] = bool(order['partial_closed'])
                order['trail_active'] = bool(order['trail_active'])
                return order
            
            return None
    
    def get_today_stats(self) -> Dict:
        """Get today's trading statistics using IST date from close_time_ist."""
        ist_now = datetime.now(timezone.utc).astimezone(_IST)
        today_ist = ist_now.strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT COUNT(*) as trades, 
                       SUM(CASE WHEN final_pnl > 0 THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN final_pnl < 0 THEN 1 ELSE 0 END) as losses,
                       SUM(final_pnl) as total_pnl,
                       AVG(final_pnl) as avg_pnl,
                       MAX(final_pnl) as best_trade,
                       MIN(final_pnl) as worst_trade
                FROM orders 
                WHERE status = 'CLOSED' AND substr(close_time_ist, 1, 10) = ?
            """, (today_ist,))
            
            row = cursor.fetchone()
            stats = dict(row) if row else {}
            
            cursor = conn.execute("""
                SELECT COUNT(*) as open_count, SUM(live_pnl) as open_pnl
                FROM orders WHERE status = 'OPEN'
            """)
            row = cursor.fetchone()
            stats.update(dict(row) if row else {})
            
            for key in ['trades', 'wins', 'losses', 'open_count']:
                stats[key] = stats[key] or 0
            for key in ['total_pnl', 'avg_pnl', 'best_trade', 'worst_trade', 'open_pnl']:
                stats[key] = round(stats[key] or 0, 2)
            
            return stats
    
    def get_daily_pnl(self, days: int = 30) -> List[Dict]:
        """Get daily PNL breakdown using IST dates from close_time_ist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT 
                    substr(close_time_ist, 1, 10) as ist_date,
                    COUNT(*) as trades,
                    SUM(CASE WHEN final_pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN final_pnl < 0 THEN 1 ELSE 0 END) as losses,
                    ROUND(SUM(final_pnl), 2) as daily_pnl,
                    ROUND(AVG(final_pnl), 2) as avg_pnl,
                    ROUND(MAX(final_pnl), 2) as best_trade,
                    ROUND(MIN(final_pnl), 2) as worst_trade
                FROM orders
                WHERE status = 'CLOSED' AND close_time_ist IS NOT NULL
                GROUP BY ist_date
                ORDER BY ist_date DESC
                LIMIT ?
            """, (days,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_strategy_performance(self, days: int = 30) -> List[Dict]:
        """Get performance breakdown by strategy."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT strategy,
                       COUNT(*) as trades,
                       SUM(CASE WHEN final_pnl > 0 THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN final_pnl < 0 THEN 1 ELSE 0 END) as losses,
                       SUM(final_pnl) as total_pnl,
                       AVG(final_pnl) as avg_pnl,
                       MAX(final_pnl) as best_trade,
                       MIN(final_pnl) as worst_trade
                FROM orders 
                WHERE status = 'CLOSED' AND close_time >= ?
                GROUP BY strategy
                ORDER BY total_pnl DESC
            """, (cutoff.isoformat(),))
            
            results = []
            for row in cursor.fetchall():
                stats = dict(row)
                stats['win_rate'] = round((stats['wins'] / stats['trades']) * 100, 1) if stats['trades'] > 0 else 0
                for key in ['total_pnl', 'avg_pnl', 'best_trade', 'worst_trade']:
                    stats[key] = round(stats[key] or 0, 2)
                results.append(stats)
            
            return results
    
    def cleanup_old_data(self, days: int = 90):
        """Remove orders older than specified days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM orders 
                WHERE status = 'CLOSED' AND close_time < ?
            """, (cutoff.isoformat(),))
            
            return cursor.rowcount
    
    def export_snapshot(self, filename: str = None) -> str:
        """Export current database state to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"orders_snapshot_{timestamp}.json"
        
        filepath = os.path.join(_BASE_DIR, filename)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM orders ORDER BY open_time DESC")
            
            orders = []
            for row in cursor.fetchall():
                order = dict(row)
                order['features'] = json.loads(order['features'] or '{}')
                orders.append(order)
        
        snapshot = {
            "export_time": datetime.now(timezone.utc).isoformat(),
            "total_orders": len(orders),
            "orders": orders
        }
        
        with open(filepath, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)
        
        return filepath