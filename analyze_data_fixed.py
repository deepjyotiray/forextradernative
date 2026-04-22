#!/usr/bin/env python3
"""
Analyze Current Trade Data Structure
Identify issues with trade tracking without requiring MT5 connection
"""
import json
import os
from datetime import datetime
from collections import defaultdict

def analyze_trade_data():
    """Analyze current trade data files and identify issues."""
    print("=" * 60)
    print("TRADE DATA ANALYSIS")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load trade history
    history_file = os.path.join(base_dir, "trade_history.json")
    trade_history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                trade_history = json.load(f)
            print(f"[OK] Loaded {len(trade_history)} trades from trade_history.json")
        except Exception as e:
            print(f"[ERR] Failed to load trade_history.json: {e}")
            return
    else:
        print("[ERR] trade_history.json not found")
        return
    
    # Load open trades
    open_file = os.path.join(base_dir, "open_trades.json")
    open_trades = {}
    if os.path.exists(open_file):
        try:
            with open(open_file, "r") as f:
                open_trades = json.load(f)
            print(f"[OK] Loaded {len(open_trades)} open trades from open_trades.json")
        except Exception as e:
            print(f"[ERR] Failed to load open_trades.json: {e}")
    
    print(f"\n[TRADE HISTORY ANALYSIS]")
    print(f"Total trades: {len(trade_history)}")
    
    if not trade_history:
        print("[!] No trade history to analyze")
        return
    
    # Analyze P&L distribution
    pnl_values = [t.get("pnl", 0) for t in trade_history]
    zero_pnl = sum(1 for p in pnl_values if p == 0.0)
    positive_pnl = sum(1 for p in pnl_values if p > 0)
    negative_pnl = sum(1 for p in pnl_values if p < 0)
    total_pnl = sum(pnl_values)
    
    print(f"P&L Distribution:")
    print(f"  Zero P&L: {zero_pnl} trades ({zero_pnl/len(trade_history)*100:.1f}%)")
    print(f"  Positive: {positive_pnl} trades ({positive_pnl/len(trade_history)*100:.1f}%)")
    print(f"  Negative: {negative_pnl} trades ({negative_pnl/len(trade_history)*100:.1f}%)")
    print(f"  Total P&L: ${total_pnl:+.2f}")
    
    # Check for missing fields that MT5 should provide
    print(f"\n[FIELD ANALYSIS]")
    required_mt5_fields = ['ticket', 'pnl', 'won', 'time']
    optional_mt5_fields = ['symbol', 'direction', 'volume', 'entry_price', 'exit_price', 'open_time', 'close_time', 'magic']
    
    field_coverage = {}
    for field in required_mt5_fields + optional_mt5_fields:
        count = sum(1 for t in trade_history if field in t and t[field] is not None and t[field] != "")
        field_coverage[field] = count
        status = "OK" if count == len(trade_history) else "WARN" if count > 0 else "MISS"
        print(f"  {status} {field}: {count}/{len(trade_history)} ({count/len(trade_history)*100:.1f}%)")
    
    # Analyze ticket patterns
    print(f"\n[TICKET ANALYSIS]")
    tickets = [t.get("ticket") for t in trade_history if t.get("ticket")]
    if tickets:
        print(f"  Tickets found: {len(tickets)}")
        print(f"  Unique tickets: {len(set(tickets))}")
        print(f"  Ticket range: {min(tickets)} - {max(tickets)}")
        
        # Check for duplicates
        ticket_counts = defaultdict(int)
        for ticket in tickets:
            ticket_counts[ticket] += 1
        duplicates = {t: c for t, c in ticket_counts.items() if c > 1}
        if duplicates:
            print(f"  [!] Duplicate tickets found: {duplicates}")
        else:
            print(f"  [OK] No duplicate tickets")
    else:
        print(f"  [ERR] No tickets found in trade history")
    
    # Show recent trades structure
    print(f"\n[RECENT TRADES STRUCTURE]")
    for i, trade in enumerate(trade_history[-5:], 1):
        print(f"\nTrade {len(trade_history)-5+i}:")
        for key, value in trade.items():
            if isinstance(value, dict):
                print(f"  {key}: {len(value)} items")
            else:
                print(f"  {key}: {value}")
    
    # Analyze what MT5 structure should look like
    print(f"\n[EXPECTED MT5 STRUCTURE]")
    print("Each closed trade from MT5 should have:")
    print("  - ticket: Position ID (unique)")
    print("  - symbol: Trading symbol (e.g., XAUUSD)")
    print("  - direction: BUY or SELL")
    print("  - volume: Trade size in lots")
    print("  - entry_price: Opening price")
    print("  - exit_price: Closing price")
    print("  - pnl: Net profit/loss (profit + swap + commission)")
    print("  - won: Boolean (pnl > 0)")
    print("  - open_time: ISO timestamp")
    print("  - close_time: ISO timestamp")
    print("  - magic: Magic number (234000)")
    print("  - reason: Close reason/comment")
    
    # Check current open trades
    if open_trades:
        print(f"\n[OPEN TRADES ANALYSIS]")
        print(f"Currently open: {len(open_trades)} positions")
        for ticket, trade in open_trades.items():
            direction = trade.get('direction', 'N/A')
            volume = trade.get('volume', 'N/A')
            entry = trade.get('entry', 'N/A')
            pnl = trade.get('live_pnl', 0)
            print(f"  #{ticket}: {direction} {volume} @ {entry} | Live P&L: ${pnl:+.2f}")
    
    # Recommendations
    print(f"\n[RECOMMENDATIONS]")
    if zero_pnl > 0:
        print(f"  [!] {zero_pnl} trades with 0.0 P&L - these should be fetched from MT5 history")
    
    missing_fields = [f for f in required_mt5_fields if field_coverage[f] < len(trade_history)]
    if missing_fields:
        print(f"  [!] Missing required fields: {missing_fields}")
    
    incomplete_fields = [f for f in optional_mt5_fields if field_coverage[f] < len(trade_history) * 0.8]
    if incomplete_fields:
        print(f"  [!] Incomplete optional fields: {incomplete_fields}")
    
    print(f"\n[NEXT STEPS]")
    print("1. Run the trader with MT5 connected")
    print("2. The system will automatically sync with MT5 history")
    print("3. Dashboard will show MT5-sourced data as authoritative")
    print("4. All future trades will be tracked from MT5 deals")

if __name__ == "__main__":
    analyze_trade_data()