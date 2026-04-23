#!/usr/bin/env python3
"""
Simple analysis of trading losses around 6:30 PM IST
"""
import json
from datetime import datetime, timezone, timedelta

# Load trade history
with open('trade_history.json', 'r') as f:
    trades = json.load(f)

print("=== TRADING LOSS ANALYSIS FOR TODAY ===\n")

# Get today's date (assuming we're analyzing April 23, 2026 based on the data)
target_date = "2026-04-23"
target_time_630pm = "18:30:00"  # 6:30 PM

print(f"Analyzing trades from {target_date}")
print(f"Focus time: {target_time_630pm} IST (6:30 PM)\n")

# Filter trades from today
today_trades = []
for trade in trades:
    trade_time = trade.get('time', '')
    if target_date in trade_time:
        today_trades.append(trade)

print(f"Total trades today: {len(today_trades)}")

# Sort by time
today_trades.sort(key=lambda x: x.get('time', ''))

# Analyze trades around 6:30 PM IST (which would be around 13:00 UTC)
# Looking for trades between 5:30 PM and 8:30 PM IST (12:00-15:00 UTC)
problem_window_trades = []
cumulative_pnl = 0
total_pnl_today = 0

print("\n=== ALL TRADES TODAY (chronological order) ===")
for i, trade in enumerate(today_trades, 1):
    pnl = trade.get('pnl', 0)
    total_pnl_today += pnl
    cumulative_pnl += pnl
    
    time_str = trade.get('time', '')
    # Extract time part
    if 'T' in time_str:
        time_part = time_str.split('T')[1][:8]
    else:
        time_part = time_str
    
    direction = trade.get('direction', '')
    entry = trade.get('entry_price', 0)
    exit_price = trade.get('exit_price', 0)
    comment = trade.get('comment', '')
    
    status = "WIN" if pnl > 0 else "LOSS"
    
    print(f"{i:3d}. {time_part} | {direction:4s} | Entry: {entry:8.2f} | Exit: {exit_price:8.2f} | "
          f"PnL: ${pnl:7.2f} | Running: ${cumulative_pnl:7.2f} | {status}")
    
    # Check if this is in the problem window (around 6:30 PM IST)
    # 6:30 PM IST = 13:00 UTC, so looking for trades between 12:00-15:00 UTC
    hour = int(time_part.split(':')[0])
    if 12 <= hour <= 15:  # UTC hours corresponding to 5:30-8:30 PM IST
        problem_window_trades.append(trade)
        if comment:
            print(f"     Comment: {comment}")

print(f"\nTotal PnL today: ${total_pnl_today:.2f}")

# Analyze the problem window
print(f"\n=== PROBLEM WINDOW ANALYSIS (12:00-15:00 UTC / 5:30-8:30 PM IST) ===")
print(f"Trades in problem window: {len(problem_window_trades)}")

if problem_window_trades:
    window_pnl = sum(t.get('pnl', 0) for t in problem_window_trades)
    losses = [t for t in problem_window_trades if t.get('pnl', 0) < 0]
    wins = [t for t in problem_window_trades if t.get('pnl', 0) > 0]
    
    print(f"Window PnL: ${window_pnl:.2f}")
    print(f"Losses: {len(losses)} trades, ${sum(t.get('pnl', 0) for t in losses):.2f}")
    print(f"Wins: {len(wins)} trades, ${sum(t.get('pnl', 0) for t in wins):.2f}")
    
    if losses:
        print(f"\nLargest losses in window:")
        sorted_losses = sorted(losses, key=lambda x: x.get('pnl', 0))
        for i, trade in enumerate(sorted_losses[:5], 1):
            time_str = trade.get('time', '')
            time_part = time_str.split('T')[1][:8] if 'T' in time_str else time_str
            print(f"  {i}. {time_part} | ${trade.get('pnl', 0):.2f} | {trade.get('comment', '')}")

# Look for patterns
print(f"\n=== PATTERN ANALYSIS ===")

# Analyze by direction
sell_trades = [t for t in today_trades if t.get('direction') == 'SELL']
buy_trades = [t for t in today_trades if t.get('direction') == 'BUY']

sell_pnl = sum(t.get('pnl', 0) for t in sell_trades)
buy_pnl = sum(t.get('pnl', 0) for t in buy_trades)

print(f"SELL trades: {len(sell_trades)} trades, ${sell_pnl:.2f} PnL")
print(f"BUY trades: {len(buy_trades)} trades, ${buy_pnl:.2f} PnL")

# Analyze stop loss hits
sl_hits = [t for t in today_trades if 'sl' in t.get('comment', '').lower()]
tp_hits = [t for t in today_trades if 'tp' in t.get('comment', '').lower()]

print(f"\nStop Loss hits: {len(sl_hits)}")
print(f"Take Profit hits: {len(tp_hits)}")

# Show worst performing hour
hourly_pnl = {}
for trade in today_trades:
    time_str = trade.get('time', '')
    if 'T' in time_str:
        hour = int(time_str.split('T')[1][:2])
        hourly_pnl[hour] = hourly_pnl.get(hour, 0) + trade.get('pnl', 0)

print(f"\n=== HOURLY PnL BREAKDOWN (UTC) ===")
for hour in sorted(hourly_pnl.keys()):
    ist_hour = (hour + 5) % 24  # Convert UTC to IST (rough)
    if hour + 5 >= 24:
        ist_hour = hour + 5 - 24
    print(f"{hour:2d}:00 UTC ({ist_hour:2d}:30 IST): ${hourly_pnl[hour]:7.2f}")

# Find the worst hour
worst_hour = min(hourly_pnl.keys(), key=lambda h: hourly_pnl[h])
worst_pnl = hourly_pnl[worst_hour]
worst_ist = (worst_hour + 5) % 24
if worst_hour + 5 >= 24:
    worst_ist = worst_hour + 5 - 24

print(f"\nWorst performing hour: {worst_hour:02d}:00 UTC ({worst_ist:02d}:30 IST) with ${worst_pnl:.2f}")

print(f"\n=== SUMMARY ===")
print(f"The major losses appear to have occurred around {worst_ist:02d}:30 IST")
print(f"This corresponds to the time window you mentioned (around 6:30 PM IST)")
print(f"Total loss in that period: ${worst_pnl:.2f}")
print(f"This represents {abs(worst_pnl)/abs(total_pnl_today)*100:.1f}% of today's total losses")