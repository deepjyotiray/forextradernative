#!/usr/bin/env python3
"""
Compare MT5 History vs Dashboard Data
Fetch actual MT5 deals and compare with current trade tracking
"""
import json
import os
import sys
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import MetaTrader5 as mt5
    import config as cfg
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("[WARNING] MetaTrader5 module not available. Install with: pip install MetaTrader5")

def fetch_mt5_history():
    """Fetch actual MT5 deal history."""
    if not MT5_AVAILABLE:
        return None
    
    # Connect to MT5
    kwargs = {}
    if hasattr(cfg, 'MT5_PATH') and cfg.MT5_PATH:
        kwargs["path"] = cfg.MT5_PATH
    if hasattr(cfg, 'MT5_LOGIN') and cfg.MT5_LOGIN:
        kwargs["login"] = cfg.MT5_LOGIN
        kwargs["password"] = getattr(cfg, 'MT5_PASSWORD', '') or ""
        kwargs["server"] = getattr(cfg, 'MT5_SERVER', '') or ""

    if not mt5.initialize(**kwargs):
        print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
        return None

    print("[INFO] Connected to MT5, fetching deal history...")
    
    # Get deal history for last 7 days
    from_date = datetime.now(timezone.utc) - timedelta(days=7)
    to_date = datetime.now(timezone.utc) + timedelta(hours=1)
    deals = mt5.history_deals_get(from_date, to_date)
    
    if not deals:
        print("[WARN] No deals found in MT5 history")
        mt5.shutdown()
        return []
    
    print(f"[INFO] Found {len(deals)} total deals in MT5 history")
    
    # Filter and process deals
    symbol = getattr(cfg, 'SYMBOL', 'XAUUSD')
    magic = getattr(cfg, 'MAGIC_NUMBER', 0)
    
    bot_deals = []
    all_deals = []
    
    for d in deals:
        deal_info = {
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
        }
        
        all_deals.append(deal_info)
        
        # Filter for bot trades
        if symbol in (d.symbol or "") and d.magic == magic:
            bot_deals.append(deal_info)
    
    print(f"[INFO] Found {len(bot_deals)} bot deals (magic: {magic}, symbol: {symbol})")
    
    mt5.shutdown()
    return {"all_deals": all_deals, "bot_deals": bot_deals, "magic": magic, "symbol": symbol}

def process_deals_to_trades(deals):
    """Convert deals to closed trades (same logic as MT5Bridge)."""
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
        })
    
    closed.sort(key=lambda t: t["close_time"])
    return closed

def load_dashboard_data():
    """Load current dashboard/tracking data."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load trade history
    history_file = os.path.join(base_dir, "trade_history.json")
    trade_history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                trade_history = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load trade_history.json: {e}")
    
    # Load open trades
    open_file = os.path.join(base_dir, "open_trades.json")
    open_trades = {}
    if os.path.exists(open_file):
        try:
            with open(open_file, "r") as f:
                open_trades = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load open_trades.json: {e}")
    
    return {
        "trade_history": trade_history,
        "open_trades": open_trades
    }

def compare_data():
    """Main comparison function."""
    print("=" * 60)
    print("MT5 HISTORY vs DASHBOARD COMPARISON")
    print("=" * 60)
    
    # Fetch MT5 data
    mt5_data = fetch_mt5_history()
    if mt5_data is None:
        print("[ERROR] Could not fetch MT5 data")
        return
    
    # Process MT5 deals into closed trades
    mt5_closed_trades = process_deals_to_trades(mt5_data["bot_deals"])
    
    # Load dashboard data
    dashboard_data = load_dashboard_data()
    
    print(f"\n[MT5 DATA]")
    print(f"Total deals: {len(mt5_data['all_deals'])}")
    print(f"Bot deals: {len(mt5_data['bot_deals'])} (magic: {mt5_data['magic']})")
    print(f"Closed trades: {len(mt5_closed_trades)}")
    
    print(f"\n[DASHBOARD DATA]")
    print(f"Trade history entries: {len(dashboard_data['trade_history'])}")
    print(f"Open trades: {len(dashboard_data['open_trades'])}")
    
    # Compare closed trades
    print(f"\n[CLOSED TRADES COMPARISON]")
    print(f"MT5 closed trades: {len(mt5_closed_trades)}")
    print(f"Dashboard history: {len(dashboard_data['trade_history'])}")
    
    if mt5_closed_trades:
        mt5_tickets = {t["ticket"] for t in mt5_closed_trades}
        dashboard_tickets = {t.get("ticket") for t in dashboard_data["trade_history"] if t.get("ticket")}
        
        missing_in_dashboard = mt5_tickets - dashboard_tickets
        extra_in_dashboard = dashboard_tickets - mt5_tickets
        
        print(f"\nMissing in dashboard: {len(missing_in_dashboard)} trades")
        if missing_in_dashboard:
            print("Missing tickets:", sorted(missing_in_dashboard))
        
        print(f"Extra in dashboard: {len(extra_in_dashboard)} trades")
        if extra_in_dashboard:
            print("Extra tickets:", sorted(extra_in_dashboard))
        
        # Show recent MT5 trades
        print(f"\n[RECENT MT5 CLOSED TRADES]")
        for trade in mt5_closed_trades[-10:]:  # Last 10
            print(f"#{trade['ticket']}: {trade['direction']} {trade['volume']} @ {trade['exit_price']} | "
                  f"P&L: ${trade['pnl']:+.2f} | {trade['close_time'][:19]}")
        
        # Show recent dashboard trades
        print(f"\n[RECENT DASHBOARD TRADES]")
        for trade in dashboard_data["trade_history"][-10:]:  # Last 10
            ticket = trade.get("ticket", "N/A")
            pnl = trade.get("pnl", 0)
            time_str = trade.get("time", "")[:19] if trade.get("time") else "N/A"
            print(f"#{ticket}: P&L: ${pnl:+.2f} | {time_str}")
    
    # Calculate P&L comparison
    mt5_total_pnl = sum(t["pnl"] for t in mt5_closed_trades)
    dashboard_total_pnl = sum(t.get("pnl", 0) for t in dashboard_data["trade_history"])
    
    print(f"\n[P&L COMPARISON]")
    print(f"MT5 total P&L: ${mt5_total_pnl:+.2f}")
    print(f"Dashboard total P&L: ${dashboard_total_pnl:+.2f}")
    print(f"Difference: ${mt5_total_pnl - dashboard_total_pnl:+.2f}")
    
    # Show bot deal details if available
    if mt5_data["bot_deals"]:
        print(f"\n[BOT DEAL DETAILS - Last 5]")
        for deal in mt5_data["bot_deals"][-5:]:
            entry_type = "IN" if deal["entry"] == 0 else "OUT" if deal["entry"] == 1 else "INOUT"
            print(f"Deal #{deal['ticket']}: Pos #{deal['position_id']} | {entry_type} | "
                  f"{deal['type']} {deal['volume']} @ {deal['price']} | "
                  f"P&L: ${deal['net_profit']:+.2f} | {deal['time'][:19]}")

if __name__ == "__main__":
    compare_data()