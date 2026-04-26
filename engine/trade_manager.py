"""
Trade Manager — persistent state, correct P&L tracking.
Saves open trades to disk so restarts don't lose tracking.
Updates P&L from live positions every cycle.
Integrated with comprehensive order database and PnL validation.
"""
import time
import json
import os
import MetaTrader5 as mt5
from typing import Dict, List
from datetime import datetime, timezone, timedelta
import config as cfg
from .order_database import OrderDatabase
from .pnl_validator import PnLValidator

_IST = timezone(timedelta(hours=5, minutes=30))

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STATE_FILE = os.path.join(_BASE_DIR, "open_trades.json")


class TradeRecord:
    __slots__ = (
        "ticket", "direction", "volume", "entry", "sl", "tp",
        "sl_distance", "strategy", "confidence", "reason",
        "open_time", "open_time_ist", "fill_ts", "peak_pnl", "live_pnl",
        "sl_breakeven", "partial_closed", "trail_active",
        "initial_volume", "scalp", "be_trigger_price", "timeout_seconds",
        "features", "manage_updates", "entry_tick_velocity", "current_price",
        "early_fail_points", "tier1_min_ticks", "tier1_max_ticks",
    )

    def __init__(self, ticket, direction, volume, entry, sl, tp, sl_distance,
                 strategy="", confidence=0, reason="",
                 scalp=False, be_trigger=0, timeout=0, early_fail=0, features=None,
                 tier1_min_ticks=None, tier1_max_ticks=None):
        self.ticket = ticket
        self.direction = direction
        self.volume = volume
        self.initial_volume = volume
        self.entry = entry
        self.sl = sl
        self.tp = tp
        self.sl_distance = sl_distance
        self.strategy = strategy
        self.confidence = confidence
        self.reason = reason
        now = datetime.now(timezone.utc)
        self.open_time = now.isoformat()
        self.open_time_ist = now.astimezone(_IST).isoformat()
        self.fill_ts = time.time()
        self.peak_pnl = 0.0
        self.live_pnl = 0.0
        self.sl_breakeven = False
        self.partial_closed = False
        self.trail_active = False
        self.scalp = scalp
        self.be_trigger_price = be_trigger
        self.timeout_seconds = timeout
        self.features = features or {}
        self.manage_updates = 0
        self.entry_tick_velocity = float(self.features.get("entry_tick_velocity", 0) or 0)
        self.current_price = entry
        self.early_fail_points = float(early_fail or self.features.get("early_fail_points", 0) or 0)
        self.tier1_min_ticks = int(tier1_min_ticks if tier1_min_ticks is not None else cfg.SMC_TIER1_MIN_TICKS)
        self.tier1_max_ticks = int(tier1_max_ticks if tier1_max_ticks is not None else cfg.SMC_TIER1_MAX_TICKS)

    def to_dict(self) -> Dict:
        return {s: getattr(self, s) for s in self.__slots__}

    @classmethod
    def from_dict(cls, d: Dict) -> "TradeRecord":
        t = cls(d["ticket"], d["direction"], d["volume"], d["entry"],
                d["sl"], d["tp"], d["sl_distance"], d.get("strategy", ""),
                d.get("confidence", 0), d.get("reason", ""),
                d.get("scalp", False), d.get("be_trigger_price", 0),
                d.get("timeout_seconds", 0), d.get("early_fail_points", 0))
        t.initial_volume = d.get("initial_volume", d["volume"])
        t.open_time = d.get("open_time", "")
        t.open_time_ist = d.get("open_time_ist", "")
        t.fill_ts = d.get("fill_ts", time.time())
        t.peak_pnl = d.get("peak_pnl", 0)
        t.live_pnl = d.get("live_pnl", 0)
        t.sl_breakeven = d.get("sl_breakeven", False)
        t.partial_closed = d.get("partial_closed", False)
        t.trail_active = d.get("trail_active", False)
        t.features = d.get("features", {})
        t.manage_updates = d.get("manage_updates", 0)
        t.entry_tick_velocity = float(d.get("entry_tick_velocity", t.features.get("entry_tick_velocity", 0)) or 0)
        t.current_price = d.get("current_price", t.entry)
        t.tier1_min_ticks = int(d.get("tier1_min_ticks", cfg.SMC_TIER1_MIN_TICKS))
        t.tier1_max_ticks = int(d.get("tier1_max_ticks", cfg.SMC_TIER1_MAX_TICKS))
        return t


class TradeManager:
    def __init__(self, mt5_bridge):
        self.bridge = mt5_bridge
        self.open_trades: Dict[int, TradeRecord] = {}
        self.closed_trades: List[Dict] = []  # Deprecated - use MT5 history instead
        self._closing_tickets: set = set()
        self._pending_close_reasons: Dict[int, str] = {}  # ticket -> reason from _close_early
        self.order_db = OrderDatabase()
        self.pnl_validator = PnLValidator(mt5_bridge)
        self._load_state()

    def register_trade(self, ticket, direction, volume, entry, sl, tp, sl_distance,
                       strategy="", confidence=0, reason="",
                       scalp=False, be_trigger=0, timeout=0, early_fail=0, features=None,
                       session_type="", market_phase=""):
        # Store in memory for active management
        self.open_trades[ticket] = TradeRecord(
            ticket, direction, volume, entry, sl, tp, sl_distance,
            strategy, confidence, reason, scalp, be_trigger, timeout, early_fail, features,
        )
        
        # Store in database for permanent record
        self.order_db.store_order(
            ticket, direction, volume, entry, sl, tp, sl_distance,
            strategy, confidence, reason, scalp, be_trigger, timeout,
            features, session_type, market_phase
        )
        
        self._save_state()

    def manage_all(self, live_positions: List[Dict], tick_metrics: Dict = None):
        live_map = {p["ticket"]: p for p in live_positions}
        closed = []
        tick_metrics = tick_metrics or {}

        # Auto-adopt bot positions that exist in MT5 but aren't tracked
        for p in live_positions:
            if p["ticket"] not in self.open_trades and p.get("magic") == cfg.MAGIC_NUMBER:
                self.register_trade(
                    p["ticket"], p["type"], p["volume"], p["open_price"],
                    p.get("sl") or 0, p.get("tp") or 0,
                    abs(p["open_price"] - (p.get("sl") or p["open_price"])),
                    strategy="adopted", reason="auto-adopted from MT5",
                )
                self._log_adopt(p["ticket"])

        for ticket, trade in list(self.open_trades.items()):
            if ticket not in live_map:
                # Position gone — fetch P&L from MT5 deal history (authoritative)
                # Retry up to 3 times with short delay for deal propagation
                close_data = {"pnl": 0.0, "exit_price": 0.0, "swap": 0.0, "commission": 0.0}
                for _attempt in range(3):
                    close_data = self._fetch_closed_pnl(ticket)
                    if close_data["exit_price"] > 0:
                        break
                    time.sleep(0.3)

                pnl = close_data["pnl"]
                exit_price = close_data["exit_price"]

                # Fallback: compute P&L from prices if deal history unavailable
                if exit_price == 0.0 and trade.live_pnl != 0.0:
                    # Use last known live P&L as exit price hint
                    pnl = trade.live_pnl
                elif exit_price > 0 and pnl == 0.0:
                    # Have exit price but no deal P&L — compute it
                    if trade.direction == "BUY":
                        pnl = round((exit_price - trade.entry) * trade.initial_volume * cfg.PIP_VALUE_PER_LOT, 2)
                    else:
                        pnl = round((trade.entry - exit_price) * trade.initial_volume * cfg.PIP_VALUE_PER_LOT, 2)

                # Determine close reason — preserve _close_early reason if set
                close_reason = self._pending_close_reasons.pop(ticket, "MT5_CLOSED")

                # Update database with final PnL + exit price
                self.order_db.close_order(
                    ticket, pnl, close_reason,
                    swap=close_data["swap"], commission=close_data["commission"],
                    exit_price=exit_price
                )
                
                closed.append((ticket, pnl, pnl > 0))
                self._closing_tickets.discard(ticket)
                del self.open_trades[ticket]
            else:
                pos = live_map[ticket]
                pnl = pos["net_profit"]
                trade.live_pnl = pnl
                trade.volume = pos["volume"]
                trade.current_price = pos.get("current_price", trade.current_price)
                
                # Update SL/TP from MT5 (may have been modified by broker/EA)
                if pos.get("sl"):
                    trade.sl = pos["sl"]
                if pos.get("tp"):
                    trade.tp = pos["tp"]
                if pnl > trade.peak_pnl:
                    trade.peak_pnl = pnl
                
                # Update database with live PnL
                self.order_db.update_live_pnl(
                    ticket, pnl, trade.volume, trade.sl, trade.tp
                )

        for ticket, trade in list(self.open_trades.items()):
            if ticket in self._closing_tickets:
                continue
            self._manage_trade(trade, tick_metrics)

        if closed:
            self._save_state()

        return closed

    def _fetch_closed_pnl(self, ticket: int) -> Dict:
        """Fetch real P&L + exit price from MT5 deal history - AUTHORITATIVE SOURCE.
        Sums profit+swap+commission from ALL deals for this position_id."""
        result = {"pnl": 0.0, "exit_price": 0.0, "swap": 0.0, "commission": 0.0}
        try:
            now = datetime.now(timezone.utc)
            deals = mt5.history_deals_get(now - timedelta(days=7), now + timedelta(hours=1))
            if not deals:
                return result
            total = 0.0
            total_swap = 0.0
            total_comm = 0.0
            found = False
            for d in deals:
                if d.position_id == ticket:
                    total += d.profit + d.swap + d.commission
                    total_swap += d.swap
                    total_comm += d.commission
                    if d.entry in (1, 2):  # OUT deal
                        found = True
                        result["exit_price"] = d.price
            if found:
                result["pnl"] = round(total, 2)
                result["swap"] = round(total_swap, 2)
                result["commission"] = round(total_comm, 2)
            return result
        except Exception:
            return result

    def _manage_trade(self, t: TradeRecord, tick_metrics: Dict):
        if self._apply_universal_management(t, tick_metrics):
            return
        if t.scalp:
            self._manage_scalp(t)
        else:
            self._manage_swing(t)

    def _apply_universal_management(self, t: TradeRecord, tick_metrics: Dict) -> bool:
        age = time.time() - t.fill_ts
        t.manage_updates += 1
        current_price = t.current_price or t.entry
        points_move = (current_price - t.entry) if t.direction == "BUY" else (t.entry - current_price)
        current_tick_count = int(tick_metrics.get("tick_count", 0) or 0)
        entry_tick_count = int(t.features.get("entry_tick_count", 0) or 0)
        ticks_since_entry = (current_tick_count - entry_tick_count) if entry_tick_count and current_tick_count else t.manage_updates

        early_fail = t.early_fail_points or cfg.SMC_EARLY_FAIL_POINTS
        if cfg.TIER1_ENABLED and t.tier1_min_ticks <= ticks_since_entry <= t.tier1_max_ticks and points_move <= -early_fail:
            self._close_early(t, f"Early fail: {points_move:.2f} pts at tick {ticks_since_entry} (window {t.tier1_min_ticks}-{t.tier1_max_ticks})")
            return True

        if t.be_trigger_price > 0 and not t.sl_breakeven and t.live_pnl >= t.be_trigger_price:
            new_sl = round(t.entry, 2)
            res = self.bridge.modify_trade(t.ticket, new_sl, t.tp)
            if res.get("success"):
                t.sl = new_sl
                t.sl_breakeven = True
                self.order_db.update_management_flags(t.ticket, sl_breakeven=True)
            return True

        timeout = t.timeout_seconds or 60
        if age >= timeout and points_move < 0.30:
            self._close_early(t, f"Speed exit: {points_move:.2f} points after {age:.0f}s")
            return True

        current_velocity = float(tick_metrics.get("velocity", 0) or 0)
        if t.entry_tick_velocity > 0 and current_velocity > 0 and age >= 10:
            if current_velocity <= max(1.0, t.entry_tick_velocity * 0.5) and points_move < 0.30:
                self._close_early(
                    t,
                    f"Velocity drop: {current_velocity:.1f} from {t.entry_tick_velocity:.1f} ticks/s",
                )
                return True

        return False

    def _manage_scalp(self, t: TradeRecord):
        pnl = t.live_pnl
        # Reversal protection
        if t.peak_pnl > 2.0 and pnl < 0.50:
            self._close_early(t, f"Scalp reversal (peak ${t.peak_pnl:.2f} -> ${pnl:.2f})")
            return

    def _manage_swing(self, t: TradeRecord):
        r = t.sl_distance * t.initial_volume * cfg.PIP_VALUE_PER_LOT
        if r <= 0:
            r = 1.0
        pnl = t.live_pnl
        peak = t.peak_pnl
        age = time.time() - t.fill_ts

        if pnl >= 0.5 * r and not t.sl_breakeven:
            buf = max(t.sl_distance * 0.15, 0.30)
            new_sl = round((t.entry + buf) if t.direction == "BUY" else (t.entry - buf), 2)
            res = self.bridge.modify_trade(t.ticket, new_sl, t.tp)
            if res.get("success"):
                t.sl = new_sl
                t.sl_breakeven = True
                self.order_db.update_management_flags(t.ticket, sl_breakeven=True)
            return

        if pnl >= 1.0 * r and not t.partial_closed:
            t.partial_closed = True
            close_vol = round(t.initial_volume * 0.5, 2)
            if cfg.TIER1_ENABLED:
                # Tier 1: skip partial close at min lot, tighten SL instead
                if close_vol >= cfg.MIN_LOT and (t.volume - close_vol) >= cfg.MIN_LOT:
                    self._partial_close(t, close_vol, f"Partial 50% at 1R")
            else:
                if close_vol >= cfg.MIN_LOT:
                    self._partial_close(t, close_vol, f"Partial 50% at 1R")
            # Set runner TP at 2R + tighten SL to lock 0.5R
            rr = 2.0
            if t.direction == "BUY":
                new_tp = round(t.entry + t.sl_distance * rr, 2)
                new_sl = round(max(t.sl, t.entry + t.sl_distance * 0.5), 2)
            else:
                new_tp = round(t.entry - t.sl_distance * rr, 2)
                new_sl = round(min(t.sl, t.entry - t.sl_distance * 0.5), 2)
            self.bridge.modify_trade(t.ticket, new_sl, new_tp)
            t.tp = new_tp
            t.sl = new_sl
            t.trail_active = True
            self.order_db.update_management_flags(
                t.ticket, partial_closed=True, trail_active=True
            )
            return

        if t.trail_active and pnl >= 1.5 * r:
            if t.direction == "BUY":
                trail_sl = round(max(t.sl, t.entry + t.sl_distance * 1.0), 2)
            else:
                trail_sl = round(min(t.sl, t.entry - t.sl_distance * 1.0), 2)
            if trail_sl != t.sl:
                self.bridge.modify_trade(t.ticket, trail_sl, t.tp)
                t.sl = trail_sl
            return

        if age >= 600 and pnl < 0.2 * r:
            self._close_early(t, f"Time exit {age:.0f}s, P&L ${pnl:.2f}")
            return

        if peak >= 1.2 * r and pnl < 0.3 * r:
            self._close_early(t, f"Reversal (peak ${peak:.2f} -> ${pnl:.2f})")
            return

    def _partial_close(self, t: TradeRecord, volume: float, reason: str):
        tick = mt5.symbol_info_tick(cfg.SYMBOL)
        if not tick:
            return
        close_type = mt5.ORDER_TYPE_SELL if t.direction == "BUY" else mt5.ORDER_TYPE_BUY
        price = tick.bid if t.direction == "BUY" else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": cfg.SYMBOL,
            "volume": round(volume, 2), "type": close_type,
            "position": t.ticket, "price": price,
            "deviation": cfg.DEVIATION, "magic": cfg.MAGIC_NUMBER,
            "comment": "FT_PARTIAL", "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        mt5.order_send(request)

    def _close_early(self, t: TradeRecord, reason: str):
        self._closing_tickets.add(t.ticket)
        res = self.bridge.close_trade(t.ticket)
        if not res.get("success"):
            self._closing_tickets.discard(t.ticket)
        else:
            # Store reason + exit price; manage_all will write authoritative P&L from MT5 deals
            self._pending_close_reasons[t.ticket] = reason
            close_price = res.get("close_price", 0)
            if close_price:
                self.order_db.update_exit_price(t.ticket, close_price)

    def _log_adopt(self, ticket: int):
        try:
            with open(os.path.join(_BASE_DIR, "trader.log"), "a") as f:
                f.write(f"[ADOPT] Auto-adopted position #{ticket} from MT5\n")
        except Exception:
            pass
    
    def _log_pnl_validation(self, ticket: int, pnl_data: Dict):
        """Log PnL validation details for audit trail."""
        try:
            with open(os.path.join(_BASE_DIR, "pnl_validation.log"), "a") as f:
                timestamp = datetime.now(timezone.utc).isoformat()
                f.write(f"[{timestamp}] Ticket {ticket}: ")
                f.write(f"PnL=${pnl_data['final_pnl']}, ")
                f.write(f"Confidence={pnl_data['confidence']}, ")
                f.write(f"Notes={pnl_data['validation_notes']}\n")
        except Exception:
            pass

    def close_all(self):
        for ticket in list(self.open_trades.keys()):
            self.bridge.close_trade(ticket)

    # --- Persistence ---

    def _save_state(self):
        try:
            data = {str(k): v.to_dict() for k, v in self.open_trades.items()}
            with open(_STATE_FILE, "w") as f:
                json.dump(data, f, default=str)
        except Exception:
            pass

    def _load_state(self):
        if not os.path.isfile(_STATE_FILE):
            return
        try:
            with open(_STATE_FILE) as f:
                data = json.load(f)
            for k, v in data.items():
                ticket = int(k)
                self.open_trades[ticket] = TradeRecord.from_dict(v)
        except Exception:
            pass

    @property
    def open_count(self) -> int:
        return len(self.open_trades)

    @property
    def status(self) -> Dict:
        today_stats = self.order_db.get_today_stats() if self.order_db else {}
        return {
            "open_trades": [t.to_dict() for t in self.open_trades.values()],
            "open_count": self.open_count,
            "today_stats": today_stats or {},
        }
    
    def get_order_history(self, days: int = 30) -> List[Dict]:
        """Get order history from database."""
        return self.order_db.get_closed_orders(days)
    
    def get_strategy_performance(self, days: int = 30) -> List[Dict]:
        """Get strategy performance breakdown."""
        return self.order_db.get_strategy_performance(days)
    
    def export_snapshot(self, filename: str = None) -> str:
        """Export complete order database snapshot."""
        return self.order_db.export_snapshot(filename)
    
    def get_pnl_validation_summary(self) -> Dict:
        """Get PnL validation summary for all open positions."""
        summary = {
            "total_positions": len(self.open_trades),
            "validation_results": [],
            "account_summary": self.pnl_validator.get_account_pnl_summary()
        }
        
        for ticket in self.open_trades.keys():
            pnl_data = self.pnl_validator.get_accurate_live_pnl(ticket)
            summary["validation_results"].append({
                "ticket": ticket,
                "pnl": pnl_data['pnl'],
                "confidence": pnl_data['confidence'],
                "source": pnl_data['source']
            })
        
        return summary
