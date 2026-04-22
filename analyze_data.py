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
            print(f"[✓] Loaded {len(trade_history)} trades from trade_history.json")
        except Exception as e:
            print(f"[✗] Failed to load trade_history.json: {e}")
            return
    else:
        print("[✗] trade_history.json not found")
        return
    
    # Load open trades
    open_file = os.path.join(base_dir, "open_trades.json")
    open_trades = {}
    if os.path.exists(open_file):
        try:
            with open(open_file, "r") as f:
                open_trades = json.load(f)
            print(f"[✓] Loaded {len(open_trades)} open trades from open_trades.json")
        except Exception as e:
            print(f"[✗] Failed to load open_trades.json: {e}")
    
    print(f"\\n[TRADE HISTORY ANALYSIS]")
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
    print(f"  Total P&L: ${total_pnl:+.2f}")ds:\n        count = sum(1 for t in trade_history if field in t and t[field] is not None and t[field] != \"\")\n        field_coverage[field] = count\n        status = \"✓\" if count == len(trade_history) else \"⚠\" if count > 0 else \"✗\"\n        print(f\"  {status} {field}: {count}/{len(trade_history)} ({count/len(trade_history)*100:.1f}%)\")\n    \n    # Analyze ticket patterns\n    print(f\"\\n[TICKET ANALYSIS]\")\n    tickets = [t.get(\"ticket\") for t in trade_history if t.get(\"ticket\")]\n    if tickets:\n        print(f\"  Tickets found: {len(tickets)}\")\n        print(f\"  Unique tickets: {len(set(tickets))}\")\n        print(f\"  Ticket range: {min(tickets)} - {max(tickets)}\")\n        \n        # Check for duplicates\n        ticket_counts = defaultdict(int)\n        for ticket in tickets:\n            ticket_counts[ticket] += 1\n        duplicates = {t: c for t, c in ticket_counts.items() if c > 1}\n        if duplicates:\n            print(f\"  [!] Duplicate tickets found: {duplicates}\")\n        else:\n            print(f\"  [✓] No duplicate tickets\")\n    else:\n        print(f\"  [✗] No tickets found in trade history\")\n    \n    # Show recent trades structure\n    print(f\"\\n[RECENT TRADES STRUCTURE]\")\n    for i, trade in enumerate(trade_history[-5:], 1):\n        print(f\"\\nTrade {len(trade_history)-5+i}:\")\n        for key, value in trade.items():\n            if isinstance(value, dict):\n                print(f\"  {key}: {len(value)} items\")\n            else:\n                print(f\"  {key}: {value}\")\n    \n    # Analyze what MT5 structure should look like\n    print(f\"\\n[EXPECTED MT5 STRUCTURE]\")\n    print(\"Each closed trade from MT5 should have:\")\n    print(\"  - ticket: Position ID (unique)\")\n    print(\"  - symbol: Trading symbol (e.g., XAUUSD)\")\n    print(\"  - direction: BUY or SELL\")\n    print(\"  - volume: Trade size in lots\")\n    print(\"  - entry_price: Opening price\")\n    print(\"  - exit_price: Closing price\")\n    print(\"  - pnl: Net profit/loss (profit + swap + commission)\")\n    print(\"  - won: Boolean (pnl > 0)\")\n    print(\"  - open_time: ISO timestamp\")\n    print(\"  - close_time: ISO timestamp\")\n    print(\"  - magic: Magic number (234000)\")\n    print(\"  - reason: Close reason/comment\")\n    \n    # Check current open trades\n    if open_trades:\n        print(f\"\\n[OPEN TRADES ANALYSIS]\")\n        print(f\"Currently open: {len(open_trades)} positions\")\n        for ticket, trade in open_trades.items():\n            direction = trade.get('direction', 'N/A')\n            volume = trade.get('volume', 'N/A')\n            entry = trade.get('entry', 'N/A')\n            pnl = trade.get('live_pnl', 0)\n            print(f\"  #{ticket}: {direction} {volume} @ {entry} | Live P&L: ${pnl:+.2f}\")\n    \n    # Recommendations\n    print(f\"\\n[RECOMMENDATIONS]\")\n    if zero_pnl > 0:\n        print(f\"  [!] {zero_pnl} trades with 0.0 P&L - these should be fetched from MT5 history\")\n    \n    missing_fields = [f for f in required_mt5_fields if field_coverage[f] < len(trade_history)]\n    if missing_fields:\n        print(f\"  [!] Missing required fields: {missing_fields}\")\n    \n    incomplete_fields = [f for f in optional_mt5_fields if field_coverage[f] < len(trade_history) * 0.8]\n    if incomplete_fields:\n        print(f\"  [!] Incomplete optional fields: {incomplete_fields}\")\n    \n    print(f\"\\n[NEXT STEPS]\")\n    print(\"1. Run the trader with MT5 connected\")\n    print(\"2. The system will automatically sync with MT5 history\")\n    print(\"3. Dashboard will show MT5-sourced data as authoritative\")\n    print(\"4. All future trades will be tracked from MT5 deals\")\n\nif __name__ == \"__main__\":\n    analyze_trade_data()