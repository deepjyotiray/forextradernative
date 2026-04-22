"""
MT5 Bridge — direct Python connection to MetaTrader 5.
Handles connection, data fetching, and order execution.
"""
import time
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timezone, timedelta
import config as cfg

_IST = timezone(timedelta(hours=5, minutes=30))


class MT5Bridge:
    """Direct MT5 connection for data + execution."""

    def __init__(self):
        self._connected = False
        self._last_tick: Optional[Dict] = None
        self._candle_cache: Dict[str, pd.DataFrame] = {}
        self._candle_mtimes: Dict[str, float] = {}

    # --- Connection ---

    def connect(self) -> bool:
        kwargs = {}
        if cfg.MT5_PATH:
            kwargs["path"] = cfg.MT5_PATH
        if cfg.MT5_LOGIN:
            kwargs["login"] = cfg.MT5_LOGIN
            kwargs["password"] = cfg.MT5_PASSWORD or ""
            kwargs["server"] = cfg.MT5_SERVER or ""

        if not mt5.initialize(**kwargs):
            print(f"[MT5] Init failed: {mt5.last_error()}")
            return False

        # Ensure symbol is visible in Market Watch
        if not mt5.symbol_select(cfg.SYMBOL, True):
            print(f"[MT5] Warning: could not select {cfg.SYMBOL}")

        self._connected = True
        info = mt5.account_info()
        if info:
            print(f"[MT5] Connected: {info.login} @ {info.server} | Balance: ${info.balance:.2f}")
        return True

    def disconnect(self):
        mt5.shutdown()
        self._connected = False

    def ping(self) -> bool:
        """Check if MT5 is actually responsive (detects silent disconnects)."""
        if not self._connected:
            return False
        try:
            info = mt5.terminal_info()
            if info is None:
                self._connected = False
                return False
            return True
        except Exception:
            self._connected = False
            return False

    @property
    def connected(self) -> bool:
        return self._connected

    # --- Account ---

    def get_account(self) -> Dict:
        info = mt5.account_info()
        if not info:
            return {}
        return {
            "login": info.login,
            "server": info.server,
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "profit": info.profit,
            "leverage": info.leverage,
            "currency": info.currency,
        }

    # --- Tick ---

    def get_tick(self) -> Optional[Dict]:
        tick = mt5.symbol_info_tick(cfg.SYMBOL)
        if not tick:
            return self._last_tick
        self._last_tick = {
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": round(tick.ask - tick.bid, 2),
            "time": datetime.fromtimestamp(tick.time, tz=timezone.utc).isoformat(),
            "volume": tick.volume,
        }
        return self._last_tick

    # --- Candle Data ---

    TF_MAP = {
        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1,
    }

    def fetch_candles(self, timeframe: str = "M5", count: int = 0) -> pd.DataFrame:
        if count <= 0:
            count = cfg.MAX_CANDLES
        tf = self.TF_MAP.get(timeframe)
        if tf is None:
            return pd.DataFrame()
        rates = mt5.copy_rates_from_pos(cfg.SYMBOL, tf, 0, count)
        if rates is None or len(rates) == 0:
            return self._candle_cache.get(timeframe, pd.DataFrame())
        df = pd.DataFrame(rates)
        df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"})
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        df["volume"] = df["volume"].astype(float)
        df = df[["datetime", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
        self._candle_cache[timeframe] = df
        return df

    def get_cached(self, timeframe: str) -> pd.DataFrame:
        return self._candle_cache.get(timeframe, pd.DataFrame())

    # --- Positions ---

    def get_positions(self, symbol: str = None) -> List[Dict]:
        symbol = symbol or cfg.SYMBOL
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return []
        result = []
        for p in positions:
            profit = p.profit
            swap = p.swap
            comm = getattr(p, 'commission', 0.0)  # not all brokers have this
            result.append({
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                "volume": p.volume,
                "open_price": p.price_open,
                "current_price": p.price_current,
                "sl": p.sl if p.sl != 0 else None,
                "tp": p.tp if p.tp != 0 else None,
                "profit": profit,
                "swap": swap,
                "commission": comm,
                "net_profit": round(profit + swap + comm, 2),
                "open_time": datetime.fromtimestamp(p.time, tz=timezone.utc).isoformat(),
                "magic": p.magic,
            })
        return result

    def get_my_positions(self) -> List[Dict]:
        """Only positions opened by this bot (matching magic number)."""
        return [p for p in self.get_positions() if p["magic"] == cfg.MAGIC_NUMBER]

    def get_floating_pnl(self) -> Dict:
        """Live floating P&L from all open positions."""
        positions = self.get_positions()
        total = sum(p["net_profit"] for p in positions)
        return {
            "total": round(total, 2),
            "count": len(positions),
            "positions": [{
                "ticket": p["ticket"],
                "direction": p["type"],
                "volume": p["volume"],
                "open_price": p["open_price"],
                "current_price": p["current_price"],
                "pnl": p["net_profit"],
                "sl": p["sl"],
                "tp": p["tp"],
                "open_time": p["open_time"],
                "magic": p["magic"],
            } for p in positions],
        }

    # --- Order Execution ---

    def open_trade(self, direction: str, volume: float, sl: float, tp: float,
                   comment: str = "FT_AUTO") -> Dict:
        tick = mt5.symbol_info_tick(cfg.SYMBOL)
        if not tick:
            return {"success": False, "error": "No tick data"}

        price = tick.ask if direction == "BUY" else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": cfg.SYMBOL,
            "volume": round(volume, 2),
            "type": order_type,
            "price": price,
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "deviation": cfg.DEVIATION,
            "magic": cfg.MAGIC_NUMBER,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            return {"success": False, "error": str(mt5.last_error())}
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"success": False, "error": f"Code {result.retcode}: {result.comment}",
                    "retcode": result.retcode}
        return {
            "success": True,
            "ticket": result.order,
            "price": result.price,
            "volume": result.volume,
        }

    def close_trade(self, ticket: int) -> Dict:
        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            return {"success": False, "error": f"Position {ticket} not found"}
        p = pos[0]
        tick = mt5.symbol_info_tick(p.symbol)
        if not tick:
            return {"success": False, "error": "No tick data"}

        close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if p.type == mt5.ORDER_TYPE_BUY else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": p.symbol,
            "volume": p.volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": cfg.DEVIATION,
            "magic": cfg.MAGIC_NUMBER,
            "comment": "FT_CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            return {"success": False, "error": str(mt5.last_error())}
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"success": False, "error": f"Code {result.retcode}: {result.comment}"}
        return {"success": True, "ticket": ticket, "close_price": result.price}

    def modify_trade(self, ticket: int, sl: float, tp: float) -> Dict:
        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            return {"success": False, "error": f"Position {ticket} not found"}
        p = pos[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": p.symbol,
            "position": ticket,
            "sl": round(sl, 2),
            "tp": round(tp, 2),
        }
        result = mt5.order_send(request)
        if result is None:
            return {"success": False, "error": str(mt5.last_error())}
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"success": False, "error": f"Code {result.retcode}: {result.comment}"}
        return {"success": True, "ticket": ticket, "sl": sl, "tp": tp}

    def close_all(self) -> List[Dict]:
        results = []
        for p in self.get_my_positions():
            results.append(self.close_trade(p["ticket"]))
        return results

    # --- History ---

    def get_history(self, days: int = 30) -> List[Dict]:
        from_date = datetime.now(timezone.utc) - timedelta(days=days)
        to_date = datetime.now(timezone.utc) + timedelta(hours=1)
        # Fetch ALL deals first, then filter — group pattern can miss suffixed symbols
        deals = mt5.history_deals_get(from_date, to_date)
        if not deals:
            return []
        # Filter by symbol match
        deals = [d for d in deals if self._match_symbol(d.symbol)]
        if not deals:
            return []
        return [{
            "ticket": d.ticket,
            "order": d.order,
            "position_id": d.position_id,
            "symbol": d.symbol,
            "type": "BUY" if d.type == 0 else "SELL" if d.type == 1 else str(d.type),
            "volume": d.volume,
            "price": d.price,
            "profit": d.profit,
            "swap": d.swap,
            "commission": d.commission,
            "net_profit": round(d.profit + d.swap + d.commission, 2),
            "time": datetime.fromtimestamp(d.time, tz=timezone.utc).isoformat(),
            "magic": d.magic,
            "comment": d.comment,
            "entry": d.entry,  # 0=in, 1=out, 2=inout, 3=out_by
        } for d in deals]

    def get_closed_trades(self, days: int = 30) -> List[Dict]:
        """Build closed trade records from MT5 history.
        Primary: history_deals grouped by position_id.
        Fallback: history_orders to catch positions with missing deals (demo server quirk).
        FILTERS BY MAGIC NUMBER to only include bot trades."""
        deals = self.get_history(days)
        bot_deals = [d for d in deals if d["magic"] == cfg.MAGIC_NUMBER] if deals else []

        # Group deals by position_id
        by_pos = {}
        for d in bot_deals:
            pid = d["position_id"]
            if pid not in by_pos:
                by_pos[pid] = {"ins": [], "outs": []}
            if d["entry"] == 0:
                by_pos[pid]["ins"].append(d)
            elif d["entry"] in (1, 2):
                by_pos[pid]["outs"].append(d)

        closed = []
        deal_pids = set()  # track which positions we built from deals

        for pid, group in by_pos.items():
            outs = group["outs"]
            if not outs:
                continue
            ins = group["ins"]
            entry = ins[0] if ins else None

            total_pnl = sum(d["profit"] for d in ins + outs)
            total_swap = sum(d["swap"] for d in ins + outs)
            total_comm = sum(d["commission"] for d in ins + outs)
            net_pnl = round(total_pnl + total_swap + total_comm, 2)

            last_out = max(outs, key=lambda d: d["time"])
            direction = entry["type"] if entry else ("BUY" if last_out["type"] == "SELL" else "SELL")
            entry_price = entry["price"] if entry else 0
            volume = entry["volume"] if entry else round(sum(d["volume"] for d in outs), 2)

            closed.append({
                "ticket": pid,
                "symbol": last_out["symbol"],
                "direction": direction,
                "volume": round(volume, 2),
                "entry_price": entry_price,
                "exit_price": last_out["price"],
                "pnl": net_pnl,
                "swap": round(total_swap, 2),
                "commission": round(total_comm, 2),
                "won": net_pnl > 0,
                "open_time": entry["time"] if entry else "",
                "close_time": last_out["time"],
                "reason": last_out["comment"],
                "magic": last_out["magic"],
                "comment": last_out["comment"],
            })
            deal_pids.add(pid)

        # Fallback: scan history_orders for positions missing from deals
        # (MetaQuotes-Demo sometimes doesn't persist deal records)
        try:
            from_date = datetime.now(timezone.utc) - timedelta(days=days)
            to_date = datetime.now(timezone.utc) + timedelta(hours=1)
            orders = mt5.history_orders_get(from_date, to_date)
            if orders:
                # Group orders by position_id
                ord_by_pos = {}
                for o in orders:
                    if o.magic != cfg.MAGIC_NUMBER or not self._match_symbol(o.symbol):
                        continue
                    if o.state != 1:  # 1 = ORDER_STATE_FILLED
                        continue
                    pid = o.position_id
                    if pid in deal_pids:
                        continue  # already have this from deals
                    if pid not in ord_by_pos:
                        ord_by_pos[pid] = []
                    ord_by_pos[pid].append(o)

                for pid, pos_orders in ord_by_pos.items():
                    if len(pos_orders) < 2:
                        continue  # need at least open + close order
                    # Sort by time
                    pos_orders.sort(key=lambda o: o.time_setup)
                    open_ord = pos_orders[0]
                    close_ord = pos_orders[-1]

                    # Determine direction from open order type
                    # ORDER_TYPE_SELL=1, ORDER_TYPE_BUY=0
                    direction = "SELL" if open_ord.type == 1 else "BUY"
                    entry_price = open_ord.price_current if open_ord.price_current > 0 else open_ord.price_open
                    exit_price = close_ord.price_current if close_ord.price_current > 0 else close_ord.price_open
                    volume = open_ord.volume_initial

                    # Compute P&L from prices
                    if direction == "SELL":
                        pnl = round((entry_price - exit_price) * volume * 100, 2)
                    else:
                        pnl = round((exit_price - entry_price) * volume * 100, 2)

                    closed.append({
                        "ticket": pid,
                        "symbol": open_ord.symbol,
                        "direction": direction,
                        "volume": round(volume, 2),
                        "entry_price": round(entry_price, 2),
                        "exit_price": round(exit_price, 2),
                        "pnl": pnl,
                        "swap": 0.0,
                        "commission": 0.0,
                        "won": pnl > 0,
                        "open_time": datetime.fromtimestamp(open_ord.time_setup, tz=timezone.utc).isoformat(),
                        "close_time": datetime.fromtimestamp(close_ord.time_setup, tz=timezone.utc).isoformat(),
                        "reason": close_ord.comment,
                        "magic": open_ord.magic,
                        "comment": close_ord.comment,
                        "source": "orders",  # flag that this came from order fallback
                    })
        except Exception:
            pass

        closed.sort(key=lambda t: t["close_time"])
        return closed

    def _match_symbol(self, symbol: str) -> bool:
        """Check if a deal's symbol matches our configured symbol (handles suffixes like XAUUSDm)."""
        return cfg.SYMBOL in (symbol or "")

    def get_today_pnl(self) -> Dict:
        """Get today's realized P&L from bot trades. Day resets at IST midnight.
        Uses position-level grouping. Only counts positions whose CLOSING deal
        happened after IST midnight. Falls back to orders for missing deals."""
        now = datetime.now(timezone.utc)
        ist_now = now.astimezone(_IST)
        ist_midnight = ist_now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_start = ist_midnight.astimezone(timezone.utc)
        today_start_ts = int(today_start.timestamp())

        # Fetch deals from a wider window to capture full position P&L,
        # but only count positions closed after IST midnight
        fetch_start = today_start - timedelta(days=1)
        deals = mt5.history_deals_get(fetch_start, now + timedelta(hours=1))

        # First pass: find positions with a closing deal AFTER IST midnight
        closed_pids = set()
        for d in (deals or []):
            if not (self._match_symbol(d.symbol) and d.magic == cfg.MAGIC_NUMBER):
                continue
            if d.entry in (1, 2) and d.time >= today_start_ts:
                closed_pids.add(d.position_id)

        # Second pass: sum P&L for only those positions
        pos_pnl = {}
        for d in (deals or []):
            if not (self._match_symbol(d.symbol) and d.magic == cfg.MAGIC_NUMBER):
                continue
            pid = d.position_id
            if pid not in closed_pids:
                continue
            if pid not in pos_pnl:
                pos_pnl[pid] = 0.0
            pos_pnl[pid] += d.profit + d.swap + d.commission

        # Fallback: check orders for positions missing from deals
        try:
            orders = mt5.history_orders_get(fetch_start, now + timedelta(hours=1))
            if orders:
                ord_by_pos = {}
                for o in orders:
                    if o.magic != cfg.MAGIC_NUMBER or not self._match_symbol(o.symbol):
                        continue
                    if o.state != 1:
                        continue
                    pid = o.position_id
                    if pid in closed_pids:
                        continue
                    if pid not in ord_by_pos:
                        ord_by_pos[pid] = []
                    ord_by_pos[pid].append(o)
                for pid, pos_orders in ord_by_pos.items():
                    if len(pos_orders) < 2:
                        continue
                    pos_orders.sort(key=lambda o: o.time_setup)
                    open_ord = pos_orders[0]
                    close_ord = pos_orders[-1]
                    # Only count if closed after IST midnight
                    if close_ord.time_setup < today_start_ts:
                        continue
                    direction = "SELL" if open_ord.type == 1 else "BUY"
                    ep = open_ord.price_current if open_ord.price_current > 0 else open_ord.price_open
                    xp = close_ord.price_current if close_ord.price_current > 0 else close_ord.price_open
                    vol = open_ord.volume_initial
                    pnl = round(((ep - xp) if direction == "SELL" else (xp - ep)) * vol * 100, 2)
                    pos_pnl[pid] = pnl
                    closed_pids.add(pid)
        except Exception:
            pass

        pnl = 0.0
        wins = 0
        losses = 0
        for pid in closed_pids:
            net = round(pos_pnl.get(pid, 0.0), 2)
            pnl += net
            if net > 0:
                wins += 1
            elif net < 0:
                losses += 1

        if closed_pids:
            return {"pnl": round(pnl, 2), "trades": len(closed_pids),
                    "wins": wins, "losses": losses, "source": "deals"}

        return {"pnl": 0.0, "trades": 0, "wins": 0, "losses": 0, "source": "none"}
