#!/usr/bin/env python3
"""
Fix Trade History - Rebuild from MT5 as single source of truth
"""
import json
import os
import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta
import config as cfg

def fix_trade_history():
    """Rebuild trade history from MT5 deal history (authoritative source)."""
    
    # Connect to MT5
    kwargs = {}
    if cfg.MT5_PATH:
        kwargs["path"] = cfg.MT5_PATH
    if cfg.MT5_LOGIN:
        kwargs["login"] = cfg.MT5_LOGIN
        kwargs["password"] = cfg.MT5_PASSWORD or ""
        kwargs["server"] = cfg.MT5_SERVER or ""

    if not mt5.initialize(**kwargs):
        print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
        return False

    print("[INFO] Connected to MT5, fetching deal history...")
    
    # Get deal history for last 30 days
    from_date = datetime.now(timezone.utc) - timedelta(days=30)
    to_date = datetime.now(timezone.utc) + timedelta(hours=1)
    deals = mt5.history_deals_get(from_date, to_date)
    
    if not deals:
        print("[WARN] No deals found in MT5 history")
        mt5.shutdown()
        return False
    
    print(f"[INFO] Found {len(deals)} deals in MT5 history")
    
    # Process deals into closed trades
    entries = {}   # position_id -> IN deal
    exits = {}     # position_id -> aggregated exit info
    
    for d in deals:
        # Filter by symbol
        if cfg.SYMBOL not in (d.symbol or ""):
            continue
            
        pid = d.position_id
        if d.entry == 0:  # IN
            entries[pid] = d
        elif d.entry in (1, 2):  # OUT or IN/OUT reversal
            if pid not in exits:
                exits[pid] = {"pnl": 0.0, "volume": 0.0, "last_deal": d}
            exits[pid]["pnl"] += d.profit + d.swap + d.commission
            exits[pid]["volume"] += d.volume
            # Keep the latest exit deal for close_time/price
            if d.time >= exits[pid]["last_deal"].time:
                exits[pid]["last_deal"] = d
    
    # Build closed trades list
    closed_trades = []
    for pid, ex in exits.items():
        e = entries.get(pid)
        d = ex["last_deal"]
        pnl = round(ex["pnl"], 2)
        
        trade_record = {
            "ticket": pid,
            "pnl": pnl,
            "won": pnl > 0,
            "time": datetime.fromtimestamp(d.time, tz=timezone.utc).isoformat(),
            "symbol": d.symbol,
            "direction": e.type if e else ("BUY" if d.type == 1 else "SELL"),
            "volume": round(ex["volume"], 2),
            "entry_price": e.price if e else 0,
            "exit_price": d.price,
            "open_time": datetime.fromtimestamp(e.time, tz=timezone.utc).isoformat() if e else "",
            "close_time": datetime.fromtimestamp(d.time, tz=timezone.utc).isoformat(),
            "reason": d.comment,
            "magic": d.magic,
        }
        closed_trades.append(trade_record)
    
    # Sort by close time
    closed_trades.sort(key=lambda t: t["close_time"])
    
    print(f"[INFO] Processed {len(closed_trades)} closed trades from MT5")
    
    # Backup existing file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    history_file = os.path.join(base_dir, "trade_history.json")
    
    if os.path.exists(history_file):
        backup_file = os.path.join(base_dir, "trade_history_backup.json")
        os.rename(history_file, backup_file)
        print(f"[INFO] Backed up existing history to {backup_file}")
    
    # Write new history from MT5
    with open(history_file, "w") as f:
        json.dump(closed_trades, f, indent=2, default=str)
    
    print(f"[SUCCESS] Rebuilt trade_history.json with {len(closed_trades)} trades from MT5")
    
    # Show summary
    total_pnl = sum(t["pnl"] for t in closed_trades)
    wins = sum(1 for t in closed_trades if t["won"])
    losses = len(closed_trades) - wins
    win_rate = (wins / len(closed_trades) * 100) if closed_trades else 0
    
    print(f"[SUMMARY] Total P&L: ${total_pnl:+.2f} | Trades: {len(closed_trades)} | Win Rate: {win_rate:.1f}% ({wins}W/{losses}L)")
    
    mt5.shutdown()
    return True

if __name__ == "__main__":
    fix_trade_history()