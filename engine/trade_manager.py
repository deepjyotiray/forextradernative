"""
Trade Manager — persistent state, correct P&L tracking.
Saves open trades to disk so restarts don't lose tracking.
Updates P&L from live positions every cycle.
"""
import time
import json
import os
import MetaTrader5 as mt5
from typing import Dict, List
from datetime import datetime, timezone, timedelta
import config as cfg

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
        "features",
    )

    def __init__(self, ticket, direction, volume, entry, sl, tp, sl_distance,
                 strategy="", confidence=0, reason="",
                 scalp=False, be_trigger=0, timeout=0, features=None):
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

    def to_dict(self) -> Dict:
        return {s: getattr(self, s) for s in self.__slots__}

    @classmethod
    def from_dict(cls, d: Dict) -> "TradeRecord":
        t = cls(d["ticket"], d["direction"], d["volume"], d["entry"],
                d["sl"], d["tp"], d["sl_distance"], d.get("strategy", ""),
                d.get("confidence", 0), d.get("reason", ""),
                d.get("scalp", False), d.get("be_trigger_price", 0),
                d.get("timeout_seconds", 0))
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
        return t


class TradeManager:
    def __init__(self, mt5_bridge):
        self.bridge = mt5_bridge
        self.open_trades: Dict[int, TradeRecord] = {}
        self.closed_trades: List[Dict] = []  # Deprecated - use MT5 history instead
        self._closing_tickets: set = set()
        self._load_state()

    def register_trade(self, ticket, direction, volume, entry, sl, tp, sl_distance,
                       strategy="", confidence=0, reason="",
                       scalp=False, be_trigger=0, timeout=0, features=None):
        self.open_trades[ticket] = TradeRecord(
            ticket, direction, volume, entry, sl, tp, sl_distance,
            strategy, confidence, reason, scalp, be_trigger, timeout, features,
        )
        self._save_state()

    def manage_all(self, live_positions: List[Dict]):
        live_map = {p["ticket"]: p for p in live_positions}
        closed = []

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
                # Position gone — ALWAYS fetch P&L from MT5 history (authoritative)
                pnl = self._fetch_closed_pnl(ticket)
                # If MT5 history fetch fails, use last known live P&L as fallback
                if pnl == 0.0 and trade.live_pnl != 0.0:
                    pnl = trade.live_pnl
                now = datetime.now(timezone.utc)
                # Don't add to internal closed_trades - MT5 history is the source of truth
                closed.append((ticket, pnl, pnl > 0))
                self._closing_tickets.discard(ticket)
                del self.open_trades[ticket]
            else:
                pos = live_map[ticket]
                pnl = pos["net_profit"]
                trade.live_pnl = pnl
                trade.volume = pos["volume"]
                # Update SL/TP from MT5 (may have been modified by broker/EA)
                if pos.get("sl"):
                    trade.sl = pos["sl"]
                if pos.get("tp"):
                    trade.tp = pos["tp"]
                if pnl > trade.peak_pnl:
                    trade.peak_pnl = pnl

        for ticket, trade in list(self.open_trades.items()):
            if ticket in self._closing_tickets:
                continue
            self._manage_trade(trade)

        if closed:
            self._save_state()

        return closed

    def _fetch_closed_pnl(self, ticket: int) -> float:
        """Fetch real P&L from MT5 deal history - AUTHORITATIVE SOURCE.
        Sums profit+swap+commission from ALL deals for this position_id."""
        try:
            now = datetime.now(timezone.utc)
            deals = mt5.history_deals_get(now - timedelta(days=7), now + timedelta(hours=1))
            if not deals:
                return 0.0
            total = 0.0
            found = False
            for d in deals:
                if d.position_id == ticket:
                    total += d.profit + d.swap + d.commission
                    if d.entry in (1, 2):  # only mark found if we see an OUT deal
                        found = True
            return round(total, 2) if found else 0.0
        except Exception:
            return 0.0

    def _manage_trade(self, t: TradeRecord):
        if t.scalp:
            self._manage_scalp(t)
        else:
            self._manage_swing(t)

    def _manage_scalp(self, t: TradeRecord):
        pnl = t.live_pnl
        age = time.time() - t.fill_ts
        price_move = pnl / (t.initial_volume * cfg.PIP_VALUE_PER_LOT) if t.initial_volume > 0 else 0

        # Early failure: adverse move > $0.20 in first 15 seconds
        early_fail = getattr(t, 'early_fail', 0) or 0.20
        if age < 15 and price_move < -early_fail:
            self._close_early(t, f"Early fail: ${price_move:.2f} in {age:.0f}s")
            return

        # BE trigger
        if t.be_trigger_price > 0 and not t.sl_breakeven:
            if price_move >= t.be_trigger_price:
                buf = 0.05
                new_sl = round((t.entry + buf) if t.direction == "BUY" else (t.entry - buf), 2)
                res = self.bridge.modify_trade(t.ticket, new_sl, t.tp)
                if res.get("success"):
                    t.sl = new_sl
                    t.sl_breakeven = True
                return

        # Speed exit: no +$0.30 move in 60s
        timeout = t.timeout_seconds or 60
        if age >= timeout and price_move < 0.30:
            self._close_early(t, f"Speed exit: +${price_move:.2f} in {age:.0f}s")
            return

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
            return

        if pnl >= 1.0 * r and not t.partial_closed:
            t.partial_closed = True
            close_vol = round(t.initial_volume * 0.5, 2)
            # Only partial close if remaining volume stays above min lot
            # Otherwise skip partial and just tighten SL to lock profit
            if close_vol >= cfg.MIN_LOT and (t.volume - close_vol) >= cfg.MIN_LOT:
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

    def _log_adopt(self, ticket: int):
        try:
            with open(os.path.join(_BASE_DIR, "trader.log"), "a") as f:
                f.write(f"[ADOPT] Auto-adopted position #{ticket} from MT5\n")
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
        return {
            "open_trades": [t.to_dict() for t in self.open_trades.values()],
            "open_count": self.open_count,
        }
