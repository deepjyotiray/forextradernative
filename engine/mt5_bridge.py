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
        deals = mt5.history_deals_get(from_date, to_date, group=f"*{cfg.SYMBOL}*")
        if not deals:
            # Fallback: fetch all deals and filter by symbol substring
            deals = mt5.history_deals_get(from_date, to_date)
            if deals:
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
            "entry": d.entry,  # 0=in, 1=out
        } for d in deals]

    def get_closed_trades(self, days: int = 30) -> List[Dict]:
        """Pair IN/OUT deals from MT5 history into closed trade records.
        Handles partial closes (multiple OUT deals per position) by aggregating P&L."""
        deals = self.get_history(days)
        if not deals:
            return []
        entries = {}   # position_id -> IN deal
        exits = {}     # position_id -> aggregated exit info
        for d in deals:
            pid = d["position_id"]
            if d["entry"] == 0:  # IN
                entries[pid] = d
            elif d["entry"] in (1, 2):  # OUT or IN/OUT reversal
                if pid not in exits:
                    exits[pid] = {"pnl": 0.0, "volume": 0.0, "last_deal": d}
                exits[pid]["pnl"] += d["net_profit"]
                exits[pid]["volume"] += d["volume"]
                # Keep the latest exit deal for close_time/price
                if d["time"] >= exits[pid]["last_deal"]["time"]:
                    exits[pid]["last_deal"] = d
        closed = []
        for pid, ex in exits.items():
            e = entries.get(pid)
            d = ex["last_deal"]
            pnl = round(ex["pnl"], 2)
            closed.append({
                "ticket": pid,
                "symbol": d["symbol"],
                "direction": e["type"] if e else ("BUY" if d["type"] == "SELL" else "SELL"),
                "volume": round(ex["volume"], 2),
                "entry_price": e["price"] if e else 0,
                "exit_price": d["price"],
                "pnl": pnl,
                "won": pnl > 0,
                "open_time": e["time"] if e else "",
                "close_time": d["time"],
                "reason": d["comment"],
                "magic": d["magic"],
                "comment": d["comment"],
            })
        closed.sort(key=lambda t: t["close_time"])
        return closed

    def _match_symbol(self, symbol: str) -> bool:
        """Check if a deal's symbol matches our configured symbol (handles suffixes like XAUUSDm)."""
        return cfg.SYMBOL in (symbol or "")

    def get_today_pnl(self) -> Dict:
        """Get today's realized P&L. Day resets at IST midnight."""
        now = datetime.now(timezone.utc)
        ist_now = now.astimezone(_IST)
        ist_midnight = ist_now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_start = ist_midnight.astimezone(timezone.utc)
        
        # Try deal history first
        deals = mt5.history_deals_get(today_start, now + timedelta(hours=1))
        if deals:
            pnl = 0.0
            trades = 0
            wins = 0
            losses = 0
            for d in deals:
                if d.entry in (1, 2) and self._match_symbol(d.symbol):
                    net = d.profit + d.swap + d.commission
                    pnl += net
                    trades += 1
                    if net > 0:
                        wins += 1
                    elif net < 0:
                        losses += 1
            if trades > 0:
                return {"pnl": round(pnl, 2), "trades": trades, "wins": wins, "losses": losses, "source": "deals"}

        # Fallback: balance - initial deposit (crude but works on demo)
        info = mt5.account_info()
        if info:
            all_deals = mt5.history_deals_get(now - timedelta(days=90), now + timedelta(hours=1))
            deposit = 0.0
            if all_deals:
                for d in all_deals:
                    if d.type == 2:  # balance operation (deposit)
                        deposit += d.profit
            if deposit > 0:
                pnl = round(info.balance - deposit, 2)
                return {"pnl": pnl, "trades": 0, "wins": 0, "losses": 0, "source": "balance_delta",
                        "deposit": deposit, "balance": info.balance}

        return {"pnl": 0.0, "trades": 0, "wins": 0, "losses": 0, "source": "none"}
