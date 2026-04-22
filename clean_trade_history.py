#!/usr/bin/env python3
"""
Clean Trade History - Remove invalid entries and prepare for MT5 sync
"""
import json
import os
from datetime import datetime

def clean_trade_history():
    """Clean up trade history file and prepare for MT5 sync."""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    history_file = os.path.join(base_dir, "trade_history.json")
    
    if not os.path.exists(history_file):
        print("[INFO] No trade_history.json found, creating empty file")
        with open(history_file, "w") as f:
            json.dump([], f)
        return
    
    # Load existing history
    with open(history_file, "r") as f:
        trades = json.load(f)
    
    print(f"[INFO] Found {len(trades)} trades in history")
    
    # Backup original
    backup_file = os.path.join(base_dir, f"trade_history_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(backup_file, "w") as f:
        json.dump(trades, f, indent=2)
    print(f"[INFO] Backed up original to {backup_file}")
    
    # Filter out trades with 0.0 P&L (likely invalid)
    valid_trades = [t for t in trades if t.get("pnl", 0) != 0.0]
    invalid_count = len(trades) - len(valid_trades)
    
    if invalid_count > 0:
        print(f"[INFO] Removed {invalid_count} trades with 0.0 P&L")
    
    # Write cleaned history
    with open(history_file, "w") as f:
        json.dump(valid_trades, f, indent=2, default=str)
    
    # Show summary
    if valid_trades:
        total_pnl = sum(t.get("pnl", 0) for t in valid_trades)
        wins = sum(1 for t in valid_trades if t.get("pnl", 0) > 0)
        losses = len(valid_trades) - wins
        win_rate = (wins / len(valid_trades) * 100) if valid_trades else 0
        
        print(f"[SUMMARY] Cleaned history: {len(valid_trades)} trades")
        print(f"[SUMMARY] Total P&L: ${total_pnl:+.2f} | Win Rate: {win_rate:.1f}% ({wins}W/{losses}L)")
    else:
        print("[SUMMARY] No valid trades found")
    
    print("[SUCCESS] Trade history cleaned. MT5 will be the authoritative source going forward.")

if __name__ == "__main__":
    clean_trade_history()